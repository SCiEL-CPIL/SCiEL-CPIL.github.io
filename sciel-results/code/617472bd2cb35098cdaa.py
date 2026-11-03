"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j
    in the permutation p (i.e., swapping p[i] and p[j]).
    """
    u = p[i]
    v = p[j]
    
    old_ii = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
    new_ii = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    k_indices = np.where(mask)[0]
    
    if len(k_indices) > 0:
        pk = p[k_indices]
        
        a_i_k = a[i, k_indices]
        a_j_k = a[j, k_indices]
        a_k_i = a[k_indices, i]
        a_k_j = a[k_indices, j]
        
        b_u_pk = b[u, pk]
        b_v_pk = b[v, pk]
        b_pk_u = b[pk, u]
        b_pk_v = b[pk, v]
        
        old_cross = np.sum(a_i_k * b_u_pk + a_j_k * b_v_pk + a_k_i * b_pk_u + a_k_j * b_pk_v)
        new_cross = np.sum(a_i_k * b_v_pk + a_j_k * b_u_pk + a_k_i * b_pk_v + a_k_j * b_pk_u)
    else:
        old_cross = 0
        new_cross = 0
    
    delta = (new_ii + new_cross) - (old_ii + old_cross)
    return int(delta)


def _two_opt_search(a, b, p, max_iterations=100000, time_limit=None, start_time=None):
    """
    Perform 2-opt local search on permutation p.
    Returns (best_p, best_cost).
    """
    if time_limit is None:
        time_limit = 100
    if start_time is None:
        start_time = time.time()
    
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time.time() - start_time > time_limit - 2:
            break
        
        for i in range(n):
            if time.time() - start_time > time_limit - 2:
                break
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit - 1:
                    break
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
    
    return best_p, best_cost


def _spectral_init(a, b):
    """
    Generate an initial permutation using spectral bisection heuristic.
    Sorts facilities and locations by their Fiedler vectors (eigenvectors of Laplacians).
    """
    n = len(a)
    
    # Compute Laplacian for A
    deg_a = a.sum(axis=1)
    L_a = np.diag(deg_a) - a
    
    # Compute Laplacian for B
    deg_b = b.sum(axis=1)
    L_b = np.diag(deg_b) - b
    
    # Get eigenvalues and eigenvectors
    # We want the Fiedler vector (index 1, 0-based)
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(L_a)
        eigvals_b, eigvecs_b = np.linalg.eigh(L_b)
        
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler_a = eigvecs_a[:, 1]
        fiedler_b = eigvecs_b[:, 1]
        
        # Sort indices by Fiedler vector values
        order_a = np.argsort(fiedler_a)
        order_b = np.argsort(fiedler_b)
        
        # Construct permutation: facility order_a[k] goes to location order_b[k]
        # p[facility] = location
        p1 = np.empty(n, dtype=np.int64)
        p1[order_a] = order_b
        
        # Also try reversed order for one of them
        order_b_rev = np.argsort(-fiedler_b)
        p2 = np.empty(n, dtype=np.int64)
        p2[order_a] = order_b_rev
        
        # Evaluate both
        cost1 = _qap_cost(a, b, p1)
        cost2 = _qap_cost(a, b, p2)
        
        if cost1 < cost2:
            return p1
        else:
            return p2
            
    except Exception:
        # Fallback to rank matching if spectral fails
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def _solve_with_spectral(a, b, time_budget=110):
    """
    Solve QAP using spectral initialization followed by 2-opt.
    """
    start_time = time.time()
    n = len(a)
    
    # 1. Spectral Initialization
    p_init = _spectral_init(a, b)
    current_cost = _qap_cost(a, b, p_init)
    
    best_p = p_init.copy()
    best_cost = current_cost
    
    # 2. 2-Opt Local Search
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        p_opt, cost_opt = _two_opt_search(a, b, p_init, time_limit=remaining_time - 2, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
            
    # 3. If time remains, try a few random restarts with shorter 2-opt
    rng = np.random.default_rng(42)
    while time.time() - start_time < time_budget - 10:
        p_rand = rng.permutation(n)
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        per_restart_time = min(remaining_time - 2, 10)
        if per_restart_time < 2:
            break
            
        p_opt, cost_opt = _two_opt_search(a, b, p_rand, time_limit=per_restart_time, start_time=time.time())
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
            
    return best_p, best_cost


def solve(a, b):
    """
    Solve the QAP instance tai256c.
    Returns a permutation p of 0..n-1 as a list of integers.
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: leave 10s margin for safety
    time_budget = 110
    
    best_p, best_cost = _solve_with_spectral(a, b, time_budget=time_budget)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
