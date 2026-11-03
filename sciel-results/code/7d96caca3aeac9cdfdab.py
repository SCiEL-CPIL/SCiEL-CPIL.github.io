import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))

def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    return int(delta_k + delta_self)

def _sa_solve(a, b, p, deadline, T0=500.0, cooling=0.999):
    """
    Simulated Annealing solver.
    """
    n = len(p)
    p = p.copy()
    cost = _cost(a, b, p)
    best_p = p.copy()
    best_cost = cost
    
    T = T0
    step = 0
    while time.time() < deadline:
        # Pick a random swap
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
            
        delta = _delta_swap(a, b, p, i, j)
        
        # Acceptance
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            if T > 1e-6:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
        
        T *= cooling
        step += 1
        
    return best_p, best_cost

def _spectral_init(a, b):
    """Spectral initialization."""
    n = a.shape[0]
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        k = min(10, n)
        idx_a = np.argsort(evals_a)
        fac_emb = evecs_a[:, idx_a[:k]]
        
        idx_b = np.argsort(-evals_b)
        loc_emb = evecs_b[:, idx_b[:k]]
        
        fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
        fac_norm[fac_norm == 0] = 1
        fac_emb = fac_emb / fac_norm
        
        loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
        loc_norm[loc_norm == 0] = 1
        loc_emb = loc_emb / loc_norm
        
        cost_matrix = -fac_emb @ loc_emb.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except:
        return np.random.permutation(n)

def _local_search_first_improvement(a, b, p, deadline):
    """
    First-improvement 2-opt local search.
    """
    n = len(p)
    p = p.copy()
    cost = _cost(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break
            if improved:
                break
    return p, cost

def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 145.0 # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    def _update_best(p, cost):
        nonlocal best_p, best_cost
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # 1. Spectral Init
    p_spec = _spectral_init(a, b)
    cost_spec = _cost(a, b, p_spec)
    _update_best(p_spec, cost_spec)
    
    # 2. Local Search (First Improvement 2-opt) to polish spectral init
    if time.time() < deadline - 20:
        p_ls, cost_ls = _local_search_first_improvement(a, b, p_spec, deadline - 20)
        _update_best(p_ls, cost_ls)
    else:
        p_ls = p_spec
        cost_ls = cost_spec

    # 3. Simulated Annealing from the polished solution
    if time.time() < deadline - 5:
        # Adjust T0 based on problem size?
        # wil100 costs are around 270k. Deltas are in thousands.
        # T0=5000 might be better.
        p_sa, cost_sa = _sa_solve(a, b, p_ls, deadline - 5, T0=2000.0, cooling=0.999)
        _update_best(p_sa, cost_sa)
        
    # 4. If time remains, do another SA run with a random start
    if time.time() < deadline - 10 and best_p is not None:
        p_rand = np.random.permutation(n)
        # Quick local search on random start?
        # Or just SA directly?
        # SA from random start is slow to converge.
        # Let's do a quick local search on random start first.
        if time.time() < deadline - 15:
             p_rand_ls, cost_rand_ls = _local_search_first_improvement(a, b, p_rand, deadline - 15)
             _update_best(p_rand_ls, cost_rand_ls)
             if time.time() < deadline - 5:
                 p_sa2, cost_sa2 = _sa_solve(a, b, p_rand_ls, deadline - 5, T0=1000.0, cooling=0.999)
                 _update_best(p_sa2, cost_sa2)
             
    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
