import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))

def _precompute_deltas(a, b):
    """
    Precompute matrices for O(1) 2-opt delta calculation.
    AB = A @ B
    ATBT = A.T @ B.T
    """
    AB = a @ b
    ATBT = a.T @ b.T
    return AB, ATBT

def _delta_swap_fast(a, b, AB, ATBT, p, i, j):
    """
    Calculate delta for swapping facilities i and j in permutation p.
    Uses precomputed AB and ATBT for O(1) computation.
    """
    n = len(p)
    u = p[i]
    v = p[j]
    
    # Terms from the sum over k != i, j
    # R_i(v) = AB[i, v] - AB[i, u]
    # C_i(v) = ATBT[i, v] - ATBT[i, u]
    # R_j(v) = AB[j, v] - AB[j, u]
    # C_j(v) = ATBT[j, v] - ATBT[j, u]
    
    Ri_v = AB[i, v] - AB[i, u]
    Ci_v = ATBT[i, v] - ATBT[i, u]
    Rj_v = AB[j, v] - AB[j, u]
    Cj_v = ATBT[j, v] - ATBT[j, u]
    
    # The dot product sum included k=i and k=j. We must subtract their contributions.
    # Sum term S = sum_k (a_ik - a_jk)(b_vk - b_uk) + (a_ki - a_kj)(b_kv - b_ku)
    # S_corrected = S - Term(k=i) - Term(k=j)
    
    # Term(k=i):
    # (a_ii - a_ji)(b_vi - b_ui) + (a_ii - a_ji)(b_ivi - b_uui) ? 
    # Wait, p_i = u, p_j = v.
    # b_vk is b[v, p_k]. For k=i, p_k = u. So b[v, u].
    # b_uk is b[u, p_k]. For k=i, p_k = u. So b[u, u].
    # b_kv is b[p_k, v]. For k=i, p_k = u. So b[u, v].
    # b_ku is b[p_k, u]. For k=i, p_k = u. So b[u, u].
    
    term_k_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u]) + (a[i, i] - a[j, i]) * (b[u, v] - b[u, u])
    # Note: a[i,i] - a[j,i] is same for both parts? 
    # First part: (a_ik - a_jk). k=i -> a_ii - a_ji.
    # Second part: (a_ki - a_kj). k=i -> a_ii - a_ji.
    # Yes.
    
    # Term(k=j):
    # b_vk for k=j is b[v, v]. b_uk for k=j is b[u, v].
    # b_kv for k=j is b[v, v]. b_ku for k=j is b[v, u].
    
    term_k_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v]) + (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
    
    sum_part = (Ri_v - Rj_v + Ci_v - Cj_v) - term_k_i - term_k_j
    
    # Internal terms (i,j) and (i,i), (j,j)
    # Old: a_ii b_uu + a_jj b_vv + a_ij b_uv + a_ji b_vu
    # New: a_ii b_vv + a_jj b_uu + a_ij b_vu + a_ji b_uv
    
    old_internal = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
    new_internal = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    internal_delta = new_internal - old_internal
    
    return int(sum_part + internal_delta)

def _local_search_2opt(a, b, p, deadline):
    """
    Best-improvement 2-opt local search with O(1) delta calculation.
    """
    n = len(p)
    AB, ATBT = _precompute_deltas(a, b)
    cost = _qap_cost(a, b, p)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            u = p[i]
            # Precompute diffs for i
            # Ri(v) = AB[i, v] - AB[i, u]
            # Ci(v) = ATBT[i, v] - ATBT[i, u]
            Ri_v = AB[i, :] - AB[i, u]
            Ci_v = ATBT[i, :] - ATBT[i, u]
            
            for j in range(i + 1, n):
                v = p[j]
                
                # Calculate delta
                # Rj(v) = AB[j, v] - AB[j, u]
                # Cj(v) = ATBT[j, v] - ATBT[j, u]
                Rj_v = AB[j, v] - AB[j, u]
                Cj_v = ATBT[j, v] - ATBT[j, u]
                
                term_k_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u]) + (a[i, i] - a[j, i]) * (b[u, v] - b[u, u])
                term_k_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v]) + (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                
                sum_part = (Ri_v[v] - Rj_v + Ci_v[v] - Cj_v) - term_k_i - term_k_j
                
                old_internal = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
                new_internal = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
                internal_delta = new_internal - old_internal
                
                delta = int(sum_part + internal_delta)
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i != -1:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            improved = True
    
    return p.copy(), cost

def _spectral_init(a, b, k=10, rng=None):
    """Spectral initialization."""
    n = a.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
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
        
        noise_scale = rng.uniform(0.01, 0.05)
        fac_emb = fac_emb + noise_scale * rng.standard_normal(fac_emb.shape)
        loc_emb = loc_emb + noise_scale * rng.standard_normal(loc_emb.shape)
        
        cost_matrix = -fac_emb @ loc_emb.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    rng_main = np.random.default_rng(42)
    
    def _update_best(p, cost):
        nonlocal best_p, best_cost
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # 1. Multi-Start Spectral Initialization
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        p_spec = _spectral_init(a, b, k=min(10, n), rng=rng_main)
        cost_spec = _qap_cost(a, b, p_spec)
        _update_best(p_spec, cost_spec)
        
        if time.time() < deadline - 25:
            p_ls, cost_ls = _local_search_2opt(a, b, p_spec.copy(), deadline - 25)
            _update_best(p_ls, cost_ls)
    
    # 2. Linear Assignment Initialization
    if time.time() < deadline - 20:
        surrogate = a @ b + a.T @ b
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_lap = np.empty(n, dtype=np.int64)
        p_lap[row_ind] = col_ind
        cost_lap = _qap_cost(a, b, p_lap)
        _update_best(p_lap, cost_lap)
        
        if time.time() < deadline - 15:
            p_ls, cost_ls = _local_search_2opt(a, b, p_lap.copy(), deadline - 15)
            _update_best(p_ls, cost_ls)
    
    # 3. Iterated Local Search (ILS)
    if best_p is not None and time.time() < deadline - 10:
        current_p = best_p.copy()
        current_cost = best_cost
        
        while time.time() < deadline - 5:
            # Perturb: Random 2-opt swap or segment reverse
            if rng_main.random() < 0.5:
                i = rng_main.integers(0, n)
                j = rng_main.integers(0, n)
                if i != j:
                    current_p[i], current_p[j] = current_p[j], current_p[i]
            else:
                i = rng_main.integers(0, n - 1)
                j = rng_main.integers(i + 1, n)
                current_p[i:j+1] = current_p[i:j+1][::-1]
            
            # Local Search
            p_ls, cost_ls = _local_search_2opt(a, b, current_p.copy(), deadline - 5)
            
            if cost_ls < current_cost:
                current_p = p_ls
                current_cost = cost_ls
                _update_best(current_p, current_cost)
            else:
                # If no improvement, keep current_p but maybe accept with prob? 
                # Standard ILS: accept if better, else reject.
                # Or accept with prob exp(-(cost_ls - current_cost)/T)
                pass
    
    # 4. Final Polish
    if best_p is not None and time.time() < deadline - 1:
        p_polish, cost_polish = _local_search_2opt(a, b, best_p.copy(), deadline - 1)
        _update_best(p_polish, cost_polish)
    
    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
