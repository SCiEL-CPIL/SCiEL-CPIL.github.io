"""Combined QAP solver: QL init, vectorized delta-matrix steepest descent, ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

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

def _compute_swap_delta(a, b, p, i, j):
    """Compute the exact cost delta of swapping positions i and j in permutation p."""
    n = len(p)
    fi = p[i]
    fj = p[j]
    
    # Build vector of distance differences for all k != i,j
    # d[k] = b[p[k],p[j]] - b[p[k],p[i]] for k != i,j
    # For k=i: contribution involves b[p[i],p[j]] - b[p[i],p[i]]
    # For k=j: contribution involves b[p[j],p[i]] - b[p[j],p[j]]
    
    # Full delta = sum_k a[k,i]*(Bp[k,j]-Bp[k,i]) + sum_k a[i,k]*(Bp[k,j]-Bp[k,i])
    #             + (a[i,j]-a[j,i])*(Bp[j,i]-Bp[i,j])
    #             + a[i,i]*(Bp[i,j]-Bp[i,i]) + a[j,j]*(Bp[j,i]-Bp[j,j])
    #             - a[i,i]*(Bp[i,i]-Bp[i,i]) - a[j,j]*(Bp[j,j]-Bp[j,j])  [these are 0]
    #             - sum_{k!=i,j} a[k,i]*Bp[k,i] - sum_{k!=i,j} a[i,k]*Bp[k,i]  [wait, need to be careful]
    
    # Let me use the formula from the delta matrix derivation:
    # D[i,j] = sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
    #        + sum_k a[i,k]*(b[p[k],p[j]] - b[p[k],p[i]])
    #        + (a[i,j] - a[j,i])*(b[p[j],p[i]] - b[p[i],p[j]])
    #        + a[i,i]*(b[p[i],p[j]] - b[p[i],p[i]])  [from T1 corr_i for k=i]
    #        + a[j,j]*(b[p[j],p[i]] - b[p[j],p[j]])  [from T1 corr_j for k=j]
    #        - a[i,i]*(b[p[i],p[i]] - b[p[i],p[i]])  [these diagonal terms cancel in the sum]
    #        - a[j,j]*(b[p[j],p[j]] - b[p[j],p[j]])
    #        + a[i,i]*(b[p[j],p[i]] - b[p[i],p[i]])  [from T2 corr_i]
    #        + a[j,j]*(b[p[j],p[j]] - b[p[i],p[j]])  [from T2 corr_j]
    #        - a[i,i]*(b[p[i],p[i]] - b[p[i],p[i]])
    #        - a[j,j]*(b[p[j],p[j]] - b[p[j],p[j]])
    
    # Actually, let me just compute it directly using the full formula:
    Bp = b[np.ix_(p, p)]
    
    # Term 1: sum_k a[k,i]*(Bp[k,j] - Bp[k,i])
    T1 = np.dot(a[:, i], Bp[:, j] - Bp[:, i])
    
    # Term 2: sum_k a[i,k]*(Bp[k,j] - Bp[k,i])
    T2 = np.dot(a[i, :], Bp[:, j] - Bp[:, i])
    
    # Term 3: (a[i,j] - a[j,i])*(Bp[j,i] - Bp[i,j])
    T3 = (a[i, j] - a[j, i]) * (Bp[j, i] - Bp[i, j])
    
    # Correction terms for T1:
    # corr_i: (a[i,i] - a[i,j])*(Bp[i,j] - Bp[i,i])
    C1 = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # corr_j: (a[j,i] - a[j,j])*(Bp[j,j] - Bp[j,i])
    C2 = (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    
    # Correction terms for T2:
    # corr_i: (a[i,i] - a[j,i])*(Bp[j,i] - Bp[i,i])
    C3 = (a[i, i] - a[j, i]) * (Bp[j, i] - Bp[i, i])
    # corr_j: (a[i,j] - a[j,j])*(Bp[j,j] - Bp[i,j])
    C4 = (a[i, j] - a[j, j]) * (Bp[j, j] - Bp[i, j])
    
    return int(T1 + T2 + T3 - C1 - C2 - C3 - C4)

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
            current_cost += int(min_val)
            improved = True
        else:
            improved = False
            
    return p, current_cost

def _sa_polish(a, b, p, current_cost, time_limit, start_time):
    """
    Simulated annealing polish with O(n) delta evaluation.
    Uses a cooling schedule to escape local minima.
    Returns updated permutation and cost.
    """
    n = len(p)
    T = 1000.0  # Initial temperature
    T_min = 1.0
    alpha = 0.95  # Cooling rate
    
    Bp = b[np.ix_(p, p)]
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Try a random swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        # Compute delta using O(n) method
        delta = _compute_swap_delta(a, b, p, i, j)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept the move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            # Update Bp
            fi, fj = p[i], p[j]
            Bp[i, :] = b[fi, p]
            Bp[:, i] = b[p, fi]
            Bp[j, :] = b[fj, p]
            Bp[:, j] = b[p, fj]
        
        T *= alpha
    
    return p, current_cost

def _ql_initialization(a, b, n_components=10, time_limit=30.0, start_time=None):
    """
    Quadratic Linearization (QL) initialization.
    Decompose flow matrix into dominant eigenvector outer products,
    solve linear assignment problems for each, and combine.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(a)
    
    try:
        eigvals, eigvecs = eigh(a)
    except Exception:
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        p, _ = linear_sum_assignment(np.abs(flow_sums[:, None] - dist_sums[None, :]))
        return p
        
    idx = np.argsort(-np.abs(eigvals))[:n_components]
    
    best_p = None
    best_cost = np.inf
    
    for r in range(n_components):
        if time.time() - start_time > time_limit:
            break
            
        v = eigvecs[:, idx[r]]
        lam = eigvals[idx[r]]
        
        C = lam * np.outer(v, v)
        
        try:
            row_ind, col_ind = linear_sum_assignment(C)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            
            cost = _qap_cost(a, b, p)
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        except Exception:
            continue
    
    if time.time() - start_time < time_limit:
        combined_C = np.zeros((n, n))
        for r in range(min(3, n_components)):
            v = eigvecs[:, idx[r]]
            lam = eigvals[idx[r]]
            combined_C += lam * np.outer(v, v)
        
        try:
            row_ind, col_ind = linear_sum_assignment(combined_C)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            
            cost = _qap_cost(a, b, p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        except Exception:
            pass
    
    if best_p is not None:
        return best_p
    else:
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        p, _ = linear_sum_assignment(np.abs(flow_sums[:, None] - dist_sums[None, :]))
        return p

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit
    
    # QL Initialization
    p = _ql_initialization(a, b, n_components=8, time_limit=25.0, start_time=start_time)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # ILS Loop
    consecutive_no_improve = 0
    perturbation_size = 5
    
    while time.time() - start_time < timeout:
        # Local Search
        p, current_cost = _local_search_fast(a, b, p, current_cost, timeout, start_time)
        
        # SA Polish
        p, current_cost = _sa_polish(a, b, p, current_cost, timeout, start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        # Check time
        if time.time() - start_time > timeout:
            break
            
        # Adaptive Perturbation
        # Increase perturbation if stuck, decrease if improving
        if consecutive_no_improve >= 3:
            perturbation_size = min(perturbation_size + 2, n // 3)
        elif consecutive_no_improve == 0:
            perturbation_size = max(2, perturbation_size - 1)
        
        # Perform random swaps to escape local optimum
        for _ in range(perturbation_size):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Calculate delta using O(n) method
            delta = _compute_swap_delta(a, b, p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
