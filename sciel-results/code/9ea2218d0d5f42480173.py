"""QAP solver for wil100: Iterated Local Search with Best-Improvement 2-opt."""
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i -> location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute change in QAP cost when swapping facilities i and j in assignment p.
    
    Args:
        a: Flow matrix (n, n)
        b: Distance matrix (n, n)
        p: Permutation (n,)
        i, j: Indices to swap
        
    Returns:
        Change in cost (integer)
    """
    if i == j:
        return 0
        
    u, v = p[i], p[j]
    
    # Precompute differences
    # diff_row[k] = b[v, p[k]] - b[u, p[k]]
    diff_row = b[v, p] - b[u, p]
    # diff_col[k] = b[p[k], v] - b[p[k], u]
    diff_col = b[p, v] - b[p, u]
    
    # Main sum over all k
    # Term 1: sum_k (a[i,k] - a[j,k]) * diff_row[k]
    term1 = np.dot(a[i, :] - a[j, :], diff_row)
    # Term 2: sum_k (a[k,i] - a[k,j]) * diff_col[k]
    term2 = np.dot(a[:, i] - a[:, j], diff_col)
    
    # The above sums include k=i and k=j, which need correction
    # because p[i] and p[j] change, and the terms involving a[i,i], a[j,j], a[i,j], a[j,i]
    # are handled differently in the exact formula.
    
    # Let's derive the exact delta from scratch to be safe.
    # Old cost contributions involving i or j:
    # C_old(i,j) = sum_{k!=i,j} (a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] + a[j,k]b[v,p[k]] + a[k,j]b[p[k],v]) 
    #              + a[i,j]b[u,v] + a[j,i]b[v,u]
    #              + a[i,i]b[u,u] + a[j,j]b[v,v]
    
    # New cost contributions involving i or j (after swap, p[i]=v, p[j]=u):
    # C_new(i,j) = sum_{k!=i,j} (a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] + a[j,k]b[u,p[k]] + a[k,j]b[p[k],u]) 
    #              + a[i,j]b[v,u] + a[j,i]b[u,v]
    #              + a[i,i]b[v,v] + a[j,j]b[u,u]
    
    # Delta = C_new - C_old
    # Terms for k != i, j:
    # sum_{k!=i,j} [ a[i,k](b[v,p[k]] - b[u,p[k]]) + a[k,i](b[p[k],v] - b[p[k],u]) 
    #               + a[j,k](b[u,p[k]] - b[v,p[k]]) + a[k,j](b[p[k],u] - b[p[k],v]) ]
    # = sum_{k!=i,j} [ (a[i,k] - a[j,k])(b[v,p[k]] - b[u,p[k]]) 
    #               + (a[k,i] - a[k,j])(b[p[k],v] - b[p[k],u]) ]
    # = sum_{k!=i,j} [ (a[i,k] - a[j,k]) diff_row[k] + (a[k,i] - a[k,j]) diff_col[k] ]
    
    # We computed term1 = sum_{k} (a[i,k] - a[j,k]) diff_row[k]
    # and term2 = sum_{k} (a[k,i] - a[k,j]) diff_col[k]
    
    # So the sum for k != i,j is (term1 - k=i part - k=j part) + (term2 - k=i part - k=j part)
    
    # k=i part in term1: (a[i,i] - a[j,i]) * diff_row[i]
    # diff_row[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
    # k=j part in term1: (a[i,j] - a[j,j]) * diff_row[j]
    # diff_row[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
    
    # k=i part in term2: (a[i,i] - a[i,j]) * diff_col[i]
    # diff_col[i] = b[p[i], v] - b[p[i], u] = b[u, v] - b[u, u]
    # k=j part in term2: (a[j,i] - a[j,j]) * diff_col[j]
    # diff_col[j] = b[p[j], v] - b[p[j], u] = b[v, v] - b[v, u]
    
    sum_k_neq = (term1 - (a[i, i] - a[j, i]) * (b[v, u] - b[u, u]) - (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])) \
               + (term2 - (a[i, i] - a[i, j]) * (b[u, v] - b[u, u]) - (a[j, i] - a[j, j]) * (b[v, v] - b[v, u]))
    
    # Terms for i,j interaction and self loops:
    # Old: a[i,j]b[u,v] + a[j,i]b[v,u] + a[i,i]b[u,u] + a[j,j]b[v,v]
    # New: a[i,j]b[v,u] + a[j,i]b[u,v] + a[i,i]b[v,v] + a[j,j]b[u,u]
    
    delta_inter = (a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u]) 
                   + a[i, i] * (b[v, v] - b[u, u]) + a[j, j] * (b[u, u] - b[v, v]))
    
    total_delta = sum_k_neq + delta_inter
    return int(total_delta)


def _best_improvement_ls(a, b, p, deadline, best_cost, best_p):
    """Best-improvement 2-opt local search."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Find the best swap
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
                    
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cur += best_delta
            improved = True
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                
    return best_cost, best_p


def _greedy_start(a, b):
    """Greedy constructive heuristic."""
    n = len(a)
    p = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    # Sort facilities by total flow in descending order
    order = np.argsort(-a.sum(axis=1), kind="stable")
    
    for idx in order:
        if not np.any(used):
            p[idx] = int(np.argmin(b.sum(axis=1)))
        else:
            avail = np.where(~used)[0]
            # Score each available location
            min_score = np.inf
            best_loc = avail[0]
            for l in avail:
                s = 0.0
                for j in np.where(used)[0]:
                    s += a[idx, j] * b[l, p[j]] + a[j, idx] * b[p[j], l]
                if s < min_score:
                    min_score = s
                    best_loc = l
            p[idx] = best_loc
        used[p[idx]] = True
        
    return p


def _perturb(p, rng, k):
    """Perturb permutation p by k random swaps."""
    n = len(p)
    p = p.copy()
    for _ in range(k):
        i, j = rng.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill
    
    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Initial solution: Greedy
    p = _greedy_start(a, b)
    
    # Initial local search
    best_cost, best_p = _best_improvement_ls(a, b, p, deadline, best_cost, best_p)
    
    # Iterated Local Search
    current_p = best_p.copy()
    
    while time.time() < deadline - 5.0:
        # Perturb
        # Adaptive perturbation size: start small, increase if stuck
        # For simplicity, fixed small perturbation
        k_perturb = 5
        p_perturbed = _perturb(current_p, rng, k_perturb)
        
        # Local search
        new_cost, new_p = _best_improvement_ls(a, b, p_perturbed, deadline, best_cost, best_p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p
            current_p = new_p.copy()
        else:
            # With some probability, accept worse solution to escape
            # Or just keep current best
            # Standard ILS often accepts worse solutions with a cooling schedule
            # Here we just restart from best
            current_p = best_p.copy()
            
    return best_p.tolist()
