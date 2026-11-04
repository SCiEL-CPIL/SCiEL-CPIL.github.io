"""Combined QAP solver: spectral init, vectorized delta-matrix steepest descent, ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j in permutation p.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    M1 = a.T @ Bp
    
    # M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    M2 = a @ Bp
    
    # Term 1 matrix: T1[i, j] = M1[i, j] - M1[i, i] - M1[j, j] + M1[j, i]
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # corr_i[i, j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    
    # corr_j[i, j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    # a[j,i] is a.T[i, j], a[j,j] is diag_a[j], Bp[j,j] is diag_Bp[j], Bp[j,i] is Bp.T[i, j]
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2 matrix: T2[i, j] = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    # corr_i for T2: (a[i,i] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[i]])
    # = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    
    # corr_j for T2: (a[i,j] - a[j,j]) * (b[p[j],p[j]] - b[p[i],p[j]])
    # = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    # Term 3 matrix: T3[i, j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    # = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    # The diagonal should be 0 (swapping with self).
    np.fill_diagonal(D, 0)
    
    return D

def _local_search_fast(a, b, p, current_cost, time_limit, start_time):
    """
    Perform steepest descent local search using the full delta matrix.
    Returns updated permutation and cost.
    """
    n = len(p)
    improved = True
    
    while improved:
        if time.time() - start_time > time_limit:
            break
            
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best improving move
        min_val = np.min(D)
        
        if min_val < 0:
            # Find the indices of the minimum
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            improved = True
        else:
            improved = False
            
    return p, current_cost

def _spectral_init(a, b):
    """
    Spectral initialization: align principal eigenvectors.
    """
    n = len(a)
    try:
        eigvals_a, eigvecs_a = eigh(a)
        eigvals_b, eigvecs_b = eigh(b)
        
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        facility_scores = v_a
        location_scores = v_b
        
        heavy_facilities = np.argsort(-facility_scores, kind="stable")
        central_locations = np.argsort(-location_scores, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        return p
    except Exception:
        # Fallback to rank matching
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        return p

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit
    
    # Spectral Initialization
    p = _spectral_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # ILS Loop
    while time.time() - start_time < timeout:
        # Local Search
        p, current_cost = _local_search_fast(a, b, p, current_cost, timeout, start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time
        if time.time() - start_time > timeout:
            break
            
        # Perturbation
        # Perform a few random swaps to escape local optimum
        # Scale perturbation size with n
        num_swaps = max(2, n // 10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Calculate delta to update cost correctly
            delta = _compute_delta_matrix(a, b, p)[i, j]
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
