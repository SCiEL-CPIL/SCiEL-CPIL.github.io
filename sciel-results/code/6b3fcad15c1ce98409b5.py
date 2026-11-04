"""Combined QAP solver: vectorized steepest-descent local search + ILS with spectral init."""
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
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    M1 = a.T @ Bp
    
    # M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    M2 = a @ Bp
    
    # Term 1: T1[i, j] = sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    # Corrections for T1:
    # k=i: (a[i,i] - a[i,j]) * (b[p[i],p[j]] - b[p[i],p[i]])
    #      = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    # k=j: (a[j,i] - a[j,j]) * (b[p[j],p[j]] - b[p[j],p[i]])
    #      = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # C1[i, j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    
    # C2[i, j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    # a[j,i] is a.T[i, j]
    # Bp[j,i] is Bp.T[i, j] (Bp is symmetric)
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2: T2[i, j] = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    # Corrections for T2:
    # k=i: (a[i,i] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[i]])
    #      = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    # k=j: (a[i,j] - a[j,j]) * (b[p[j],p[j]] - b[p[i],p[j]])
    #      = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    
    # C3[i, j] = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    
    # C4[i, j] = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    # Term 3: T3[i, j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    # = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    # The diagonal should be 0 (swapping with self).
    np.fill_diagonal(D, 0)
    
    return D

def _local_search_fast(a, b, p, max_iter=None, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the full delta matrix.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 1000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() - start_time > time_limit:
            break
            
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best improving move
        min_val = np.min(D)
        
        if min_val < 0:
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            improved = True
            iter_count += 1
        else:
            improved = False
            
    return p, current_cost

def _spectral_initial(a, b):
    """
    Spectral initialization using eigenvectors.
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

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start ILS
    best_p = None
    best_cost = float('inf')
    
    # Initial solution from spectral method
    p = _spectral_initial(a, b)
    
    # Track number of iterations to adapt perturbation strength
    iteration = 0
    
    while time.time() - start_time < total_deadline:
        # Local Search
        p, current_cost = _local_search_fast(a, b, p, time_limit=total_deadline, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time
        if time.time() - start_time > total_deadline:
            break
        
        # Adaptive perturbation: start with larger perturbations, decrease over time
        # This helps escape local optima early and refine later
        progress = (time.time() - start_time) / total_deadline
        if progress < 0.3:
            num_swaps = max(3, n // 8)  # Larger perturbations early
        elif progress < 0.6:
            num_swaps = max(2, n // 12)  # Medium perturbations
        else:
            num_swaps = max(1, n // 15)  # Smaller perturbations late
        
        # Compute delta matrix once for all perturbations
        D = _compute_delta_matrix(a, b, p)
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = D[i, j]
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            # Recompute D after swap (only needed if we do multiple swaps)
            D = _compute_delta_matrix(a, b, p)
        
        iteration += 1
    
    # Final polish pass: if we have time left, do more local search iterations
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 5.0 and best_p is not None:
        p_polish = best_p.copy()
        p_polish, polish_cost = _local_search_fast(
            a, b, p_polish, 
            max_iter=500,
            time_limit=total_deadline - 2.0,  # Leave 2s safety margin
            start_time=start_time
        )
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = p_polish
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
