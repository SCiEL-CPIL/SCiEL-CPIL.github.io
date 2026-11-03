"""
Combined Iterated Local Search + Simulated Annealing solver for QAPLIB wil100.
Strategy:
1. Fast O(n) incremental delta evaluation for 2-opt swaps using precomputed vectors.
2. Multi-start: greedy initial + random restarts.
3. Iterated Local Search:
   - First-improvement local search until local optimum.
   - Perturbation (k random swaps) to escape local optimum.
   - Accept if perturbed solution improves best, or with SA probability.
4. Final polish with SA from the best solution found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_swap_deltas(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    Uses precomputed row/col difference vectors for efficiency.
    """
    pi = p[i]
    pj = p[j]
    
    # Precompute the difference vectors once per call
    # These are the same for all swaps involving i or j
    
    # For swap (i, j):
    # delta = sum_k (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    #       + sum_k (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    #       - corrections for k=i and k=j
    
    diff_a_row = a[i, :] - a[j, :]
    diff_a_col = a[:, i] - a[:, j]
    
    # Term 1: dot product with b[pj, p] - b[pi, p]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Term 2: dot product with b[p, pj] - b[p, pi]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Corrections for k=i:
    # T1: (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
    # T2: (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
    corr_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    corr_i += (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    
    # Corrections for k=j:
    # T1: (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
    # T2: (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
    corr_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    corr_j += (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta = sum_T1 + sum_T2 - corr_i - corr_j
    
    # Add direct terms
    delta += (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    return int(delta)

def _greedy_initial(a, b):
    """Create an initial permutation using rank-matching heuristic."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _local_search(a, b, p, time_limit, start_time):
    """
    First-improvement local search: repeatedly pick a random swap,
    accept if it improves. Continue until time runs out or no improvement
    found after many tries.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    no_improve = 0
    max_no_improve = 300  # Reduced from 500 to detect local optima faster
    
    # Precompute some useful arrays
    b_rows = b  # alias for clarity
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_deltas(a, b, p, i, j)
        
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

def _perturb(p, k):
    """Apply k random swaps to p, returning a copy."""
    n = len(p)
    p_new = p.copy()
    for _ in range(k):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new

def _sa_polish(a, b, p, time_limit, start_time):
    """
    Simulated Annealing polish from a good starting point.
    Uses time-based cooling schedule.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    T_start = 300.0
    T_end = 1.0
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(1.0, elapsed / max(time_limit, 1.0))
        T = T_start * (T_end / T_start) ** frac
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_deltas(a, b, p, i, j)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)  # Fixed seed for reproducibility within a run
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multi-start with local search
    # Start with greedy
    p = _greedy_initial(a, b)
    p, cost = _local_search(a, b, p, 5.0, start_time)
    best_p = p.copy()
    best_cost = cost
    
    # Try several random restarts with local search
    num_restarts = 15
    for r in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 30:
            break
        
        p_rand = np.random.permutation(n).astype(np.int64)
        # Short local search from random start
        ls_time = min(3.0, total_deadline - elapsed - 25)
        if ls_time < 0.5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_rand.copy(), ls_time, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: Iterated Local Search from best solution
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 20:
        p_ils = best_p.copy()
        cost_ils = best_cost
        perturbation_k = 3
        
        while time.time() - start_time < total_deadline - 15:
            # Perturb
            p_pert = _perturb(p_ils, perturbation_k)
            
            # Local search from perturbed
            ls_time = min(5.0, total_deadline - (time.time() - start_time) - 10)
            if ls_time < 0.5:
                break
            
            p_ls, cost_ls = _local_search(a, b, p_pert, ls_time, start_time)
            
            # Acceptance: if better than current ILS solution, or with SA probability
            if cost_ls < cost_ils:
                p_ils = p_ls
                cost_ils = cost_ls
                # Reduce perturbation on improvement
                perturbation_k = max(2, perturbation_k - 1)
            else:
                # SA acceptance: accept worse solution with some probability
                # to escape local minima
                delta = cost_ls - cost_ils
                T = 100.0  # Fixed temperature for ILS acceptance
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    p_ils = p_ls
                    cost_ils = cost_ls
                # Increase perturbation on stagnation
                perturbation_k = min(10, perturbation_k + 1)
            
            if cost_ils < best_cost:
                best_cost = cost_ils
                best_p = p_ils.copy()
    
    # Phase 3: Final SA polish from best solution
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 3:
        sa_time = total_deadline - elapsed - 2
        p_final, cost_final = _sa_polish(a, b, best_p.copy(), sa_time, start_time)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
