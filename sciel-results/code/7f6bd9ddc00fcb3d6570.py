"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute the QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return np.sum(a * b[np.ix_(p, p)])


def _delta_swap(a, b, p, i, j, b_p=None, b_pT=None):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Uses precomputed b_p = b[p,:] and b_pT = b[:,p] for speed.
    Returns delta = cost_after - cost_before.
    """
    pi = p[i]
    pj = p[j]
    
    if b_p is None:
        b_p = b[p, :]
    if b_pT is None:
        b_pT = b[:, p]
    
    # diff vectors
    diff_row = b_p[pj, :] - b_p[pi, :]  # b[pj, p[k]] - b[pi, p[k]] for all k
    diff_col = b_pT[:, pj] - b_pT[:, pi]  # b[p[k], pj] - b[p[k], pi] for all k
    
    # Full sums over all k
    S1 = np.dot(a[i, :], diff_row)
    S2 = np.dot(a[:, i], diff_col)
    S3 = -np.dot(a[j, :], diff_row)
    S4 = -np.dot(a[:, j], diff_col)
    
    # Subtract k=i and k=j terms
    # For S1: k=i -> a[i,i]*diff_row[pi], k=j -> a[i,j]*diff_row[pj]
    # For S2: k=i -> a[i,i]*diff_col[pi], k=j -> a[j,i]*diff_col[pj]
    # For S3: k=i -> -a[j,i]*diff_row[pi], k=j -> -a[j,j]*diff_row[pj]
    # For S4: k=i -> -a[i,j]*diff_col[pi], k=j -> -a[j,j]*diff_col[pj]
    
    sub_S1 = a[i, i] * diff_row[pi] + a[i, j] * diff_row[pj]
    sub_S2 = a[i, i] * diff_col[pi] + a[j, i] * diff_col[pj]
    sub_S3 = -a[j, i] * diff_row[pi] - a[j, j] * diff_row[pj]
    sub_S4 = -a[i, j] * diff_col[pi] - a[j, j] * diff_col[pj]
    
    sum_S1 = S1 - sub_S1
    sum_S2 = S2 - sub_S2
    sum_S3 = S3 - sub_S3
    sum_S4 = S4 - sub_S4
    
    # Cross terms
    b_pi_pi = b[pi, pi]
    b_pj_pj = b[pj, pj]
    b_pi_pj = b[pi, pj]
    b_pj_pi = b[pj, pi]
    
    cross = (a[i, i] * (b_pj_pj - b_pi_pi) +
             a[j, j] * (b_pi_pi - b_pj_pj) +
             a[i, j] * (b_pj_pi - b_pi_pj) +
             a[j, i] * (b_pi_pj - b_pj_pi))
    
    delta = sum_S1 + sum_S2 + sum_S3 + sum_S4 + cross
    return delta


def _local_search(a, b, p, current_cost, deadline, max_passes=20):
    """
    Vectorized 2-swap local search.
    Performs full passes over all pairs (i,j) with i<j, accepting improving swaps.
    """
    n = len(p)
    p = p.copy()
    b_p = b[p, :]
    b_pT = b[:, p]
    
    passes = 0
    while passes < max_passes and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                pi = p[i]
                pj = p[j]
                
                # Quick delta computation using precomputed arrays
                diff_row = b_p[pj, :] - b_p[pi, :]
                diff_col = b_pT[:, pj] - b_pT[:, pi]
                
                S1 = np.dot(a[i, :], diff_row)
                S2 = np.dot(a[:, i], diff_col)
                S3 = -np.dot(a[j, :], diff_row)
                S4 = -np.dot(a[:, j], diff_col)
                
                sub_S1 = a[i, i] * diff_row[pi] + a[i, j] * diff_row[pj]
                sub_S2 = a[i, i] * diff_col[pi] + a[j, i] * diff_col[pj]
                sub_S3 = -a[j, i] * diff_row[pi] - a[j, j] * diff_row[pj]
                sub_S4 = -a[i, j] * diff_col[pi] - a[j, j] * diff_col[pj]
                
                cross = (a[i, i] * (b[pj, pj] - b[pi, pi]) +
                         a[j, j] * (b[pi, pi] - b[pj, pj]) +
                         a[i, j] * (b[pj, pi] - b[pi, pj]) +
                         a[j, i] * (b[pi, pj] - b[pj, pi]))
                
                delta = (S1 - sub_S1) + (S2 - sub_S2) + (S3 - sub_S3) + (S4 - sub_S4) + cross
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    # Update b_p and b_pT
                    b_p[pi, :] = b[pj, :]
                    b_p[pj, :] = b[pi, :]
                    b_pT[:, pi] = b[:, pj]
                    b_pT[:, pj] = b[:, pi]
                    improved = True
        
        if not improved:
            break
        passes += 1
    
    return p, current_cost


def _hungarian_init(a, b, rng):
    """Hungarian algorithm initialization with optional perturbation."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    cost_mat = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def _perturb(p, rng, num_swaps=5):
    """Perturb a permutation with random swaps."""
    p = p.copy()
    n = len(p)
    for _ in range(num_swaps):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multiple diverse starts with local search
    best_p = None
    best_cost = np.inf
    
    # Generate diverse initial solutions
    base_p = _hungarian_init(a, b, rng)
    
    # Try the base solution and several perturbations
    candidates = [base_p]
    for k in range(1, 6):
        candidates.append(_perturb(base_p, rng, num_swaps=k * 3))
    
    # Also try a few random starts
    for _ in range(3):
        candidates.append(rng.permutation(n).astype(np.int64))
    
    # Local search on each candidate
    for p_init in candidates:
        if time.time() >= deadline - 30.0:  # Leave time for SA
            break
        p_ls, cost_ls = _local_search(a, b, p_init, _compute_cost(a, b, p_init), deadline - 30.0, max_passes=5)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    if best_p is None:
        best_p = base_p
        best_cost = _compute_cost(a, b, best_p)
    
    # Phase 2: Simulated Annealing refinement on the best solution
    if time.time() < deadline - 5.0:
        p = best_p.copy()
        current_cost = best_cost
        b_p = b[p, :]
        b_pT = b[:, p]
        
        T_start = max(100.0, best_cost * 0.01)
        T_end = 1.0
        T = T_start
        cooling_rate = 0.9995
        
        while time.time() < deadline - 2.0:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            pi = p[i]
            pj = p[j]
            
            # Fast delta using precomputed arrays
            diff_row = b_p[pj, :] - b_p[pi, :]
            diff_col = b_pT[:, pj] - b_pT[:, pi]
            
            S1 = np.dot(a[i, :], diff_row)
            S2 = np.dot(a[:, i], diff_col)
            S3 = -np.dot(a[j, :], diff_row)
            S4 = -np.dot(a[:, j], diff_col)
            
            sub_S1 = a[i, i] * diff_row[pi] + a[i, j] * diff_row[pj]
            sub_S2 = a[i, i] * diff_col[pi] + a[j, i] * diff_col[pj]
            sub_S3 = -a[j, i] * diff_row[pi] - a[j, j] * diff_row[pj]
            sub_S4 = -a[i, j] * diff_col[pi] - a[j, j] * diff_col[pj]
            
            cross = (a[i, i] * (b[pj, pj] - b[pi, pi]) +
                     a[j, j] * (b[pi, pi] - b[pj, pj]) +
                     a[i, j] * (b[pj, pi] - b[pi, pj]) +
                     a[j, i] * (b[pi, pj] - b[pj, pi]))
            
            delta = (S1 - sub_S1) + (S2 - sub_S2) + (S3 - sub_S3) + (S4 - sub_S4) + cross
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                # Update precomputed arrays
                b_p[pi, :] = b[pj, :]
                b_p[pj, :] = b[pi, :]
                b_pT[:, pi] = b[:, pj]
                b_pT[:, pj] = b[:, pi]
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= cooling_rate
            if T < T_end:
                T = T_end
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
