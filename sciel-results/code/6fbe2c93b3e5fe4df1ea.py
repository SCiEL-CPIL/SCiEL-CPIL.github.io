"""
Benders Decomposition Heuristic for QAPLIB wil100.
Strategy:
1. Master Problem: Linear Assignment Problem (LAP) using surrogate costs.
2. Subproblem: Evaluate exact QAP cost for the LAP solution.
3. Cut Generation: Update surrogate costs using a Benders-like mechanism
   (linearizing the quadratic cost around current assignments) to guide
   the master problem away from high-cost regions.
4. Local Search: Refine LAP solutions with swap-based local search to escape
   linearization artifacts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _local_search(a, b, p, time_limit, start_time):
    """
    First-improvement local search.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    
    return best_p, best_cost

def _benders_update(c, a, b, p, current_cost, alpha=0.5):
    """
    Update the surrogate cost matrix c using Benders-like cutting plane logic.
    
    The idea is to adjust c[i, j] so that the linear assignment problem
    prefers assignments that have lower exact quadratic costs.
    
    We approximate the marginal contribution of assigning facility i to location j
    by looking at the current assignment and the cost differences.
    
    A simple heuristic cut:
    For each facility i currently assigned to location p[i], we want to discourage
    assignments that lead to high interaction costs.
    
    Here we use a gradient-like update:
    The cost of assigning i to j in the exact problem depends on b[j, p[k]] * a[i, k] + b[p[k], j] * a[k, i].
    We can approximate the "penalty" for assigning i to j by summing over k:
    penalty[i, j] = sum_k a[i, k] * b[j, p[k]] + a[k, i] * b[p[k], j]
    
    This is the linearization of the quadratic cost around the current assignment p.
    We blend the current surrogate c with this penalty.
    """
    n = len(p)
    # Compute penalty matrix P where P[i, j] is the linearized cost contribution
    # if facility i is assigned to location j, given current assignment p for others.
    # P[i, j] = sum_{k != i} a[i, k] * b[j, p[k]] + a[k, i] * b[p[k], j]
    
    # Vectorized computation:
    # b_j_p = b[:, p] -> shape (n, n), where b_j_p[j, k] = b[j, p[k]]
    # b_p_j = b[p, :] -> shape (n, n), where b_p_j[k, j] = b[p[k], j]
    
    b_j_p = b[:, p]  # b[j, p[k]] for all j, k
    b_p_j = b[p, :]  # b[p[k], j] for all k, j
    
    # P[i, j] = sum_k a[i, k] * b[j, p[k]] + a[k, i] * b[p[k], j]
    # Note: a[i, k] * b[j, p[k]] is (a @ b_j_p.T)[i, j]? 
    # a[i, k] * b[j, p[k]] -> sum_k a[i, k] * b_j_p[j, k]
    # This is (a @ b_j_p.T)[i, j]
    term1 = a @ b_j_p.T
    
    # a[k, i] * b[p[k], j] -> sum_k a[k, i] * b_p_j[k, j]
    # This is (a.T @ b_p_j)[i, j]
    term2 = a.T @ b_p_j
    
    P = term1 + term2
    
    # Remove self-interaction if any (usually handled by QAP structure, but safe to keep)
    # P[i, p[i]] includes a[i, i]*b[p[i], p[i]] which is constant for fixed i.
    
    # Update c: c = (1-alpha)*c + alpha * P
    # We normalize P to have similar scale to c to avoid dominance.
    if np.max(P) > 0:
        P_norm = P / np.max(P) * np.max(c)
    else:
        P_norm = P
        
    c_new = (1 - alpha) * c + alpha * P_norm
    return c_new

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    # Initialize surrogate cost matrix c
    # Start with a simple heuristic: row sums of a and b
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    c = np.outer(a_row_sums, b_row_sums)
    
    best_p = None
    best_cost = np.inf
    
    # Benders Loop
    alpha = 0.7  # Aggressive update
    max_iters = 1000
    
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 10:
            break
        
        # 1. Master Problem: Solve LAP with surrogate cost c
        # Add small random noise to break ties and encourage exploration
        c_noisy = c + np.random.rand(n, n) * 0.1 * np.max(c)
        
        row_ind, col_ind = linear_sum_assignment(c_noisy)
        p_lap = np.zeros(n, dtype=np.int64)
        p_lap[row_ind] = col_ind
        
        # 2. Subproblem: Evaluate exact cost
        cost_lap = _qap_cost(a, b, p_lap)
        
        # 3. Local Search Refinement
        ls_time = min(2.0, total_deadline - (time.time() - start_time) - 5)
        if ls_time > 0.1:
            p_ls, cost_ls = _local_search(a, b, p_lap.copy(), ls_time, start_time)
        else:
            p_ls, cost_ls = p_lap.copy(), cost_lap
        
        # Update best
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # 4. Cut Generation: Update surrogate c
        # Use the locally searched solution for better cut information
        c = _benders_update(c, a, b, p_ls, cost_ls, alpha)
        
        # Occasionally reset alpha or perturb c to avoid cycles
        if it % 50 == 0 and it > 0:
            alpha = max(0.1, alpha * 0.9)
            
    # Final polish with SA from best solution
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 3 and best_p is not None:
        sa_time = total_deadline - elapsed - 2
        # Simple SA polish
        p_sa = best_p.copy()
        cost_sa = best_cost
        T = 100.0
        while time.time() - start_time < total_deadline - 1:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = _swap_delta(a, b, p_sa, i, j)
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += delta
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
            T = max(1.0, T * 0.99)
            
    if best_p is None:
        # Fallback
        best_p = np.arange(n)
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
