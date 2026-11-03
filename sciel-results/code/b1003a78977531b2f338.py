"""Spectral initialization + Iterated Local Search for QAP wil100."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute QAP cost: sum a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _spectral_init(a, b):
    """Spectral initialization: align high-flow facilities with high-distance locations.
    
    For QAP, we want high flows to correspond to short distances.
    We use the leading eigenvectors of A and B to create a geometric embedding,
    then match facilities to locations based on alignment.
    """
    n = len(a)
    # Symmetrize for eigendecomposition
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Get leading eigenvectors (top k)
    k = min(10, n)
    
    # Use scipy for efficiency if available, otherwise numpy
    try:
        from scipy.linalg import eigh
        # For A: we want to capture the structure; use top eigenvectors
        wA, vA = eigh(A_sym, subset_by_index=[n-k, n-1])
        wB, vB = eigh(B_sym, subset_by_index=[n-k, n-1])
    except ImportError:
        wA, vA = np.linalg.eigh(A_sym)
        vA = vA[:, -k:]
        wA = wA[-k:]
        wB, vB = np.linalg.eigh(B_sym)
        vB = vB[:, -k:]
        wB = wB[-k:]
    
    # Project facilities and locations into spectral space
    # Facilities: rows of A projected
    fac_proj = vA @ (vA.T @ np.ones(n))  # Not quite right, let's use direct projection
    # Actually, use the eigenvector coordinates directly
    fac_coords = vA  # shape (n, k)
    loc_coords = vB  # shape (n, k)
    
    # For QAP minimization: high flow should go to short distance
    # So we want to match high A-eigenvalue directions with low B-eigenvalue directions
    # Use a cost matrix in spectral space: distance between facility i and location j
    # in the spectral embedding, weighted by eigenvalues
    
    # Compute pairwise spectral distances
    # fac_coords[i] is the spectral position of facility i
    # loc_coords[j] is the spectral position of location j
    # We want to minimize sum of (spectral distance)^2 weighted by eigenvalues
    
    # Simple approach: compute a cost matrix in spectral space
    # cost[i,j] = sum_k (fac_coords[i,k]*sqrt(wA[k]) - loc_coords[j,k]*sqrt(wB[k]))^2
    sqrt_wA = np.sqrt(np.abs(wA))
    sqrt_wB = np.sqrt(np.abs(wB))
    
    fac_scaled = fac_coords * sqrt_wA[np.newaxis, :]
    loc_scaled = loc_coords * sqrt_wB[np.newaxis, :]
    
    # Compute squared distances: ||f_i - l_j||^2 = ||f_i||^2 + ||l_j||^2 - 2 f_i . l_j
    fac_norms = np.sum(fac_scaled**2, axis=1)
    loc_norms = np.sum(loc_scaled**2, axis=1)
    dot = fac_scaled @ loc_scaled.T
    
    spectral_cost = fac_norms[:, np.newaxis] + loc_norms[np.newaxis, :] - 2 * dot
    
    # Find assignment that minimizes spectral cost
    # Use Hungarian algorithm or greedy with local search
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(spectral_cost)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
    except ImportError:
        # Greedy fallback
        p = np.empty(n, dtype=np.int64)
        for i in range(n):
            remaining = np.where(p == -1)[0] if hasattr(p, '__len__') else None
            # Simple greedy: pick best unused location
            if i == 0:
                p[i] = np.argmin(spectral_cost[0])
            else:
                used = set(p[:i].tolist())
                avail = [j for j in range(n) if j not in used]
                p[i] = min(avail, key=lambda j: spectral_cost[i, j])
    
    return p


def _compute_swap_deltas(a, b, p, i, j):
    """Compute cost delta for swapping p[i] and p[j].
    
    Delta = sum over all k of a[i,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[j,k]*(b[p[j],p[k]] - b[p[i],p[k]])
          + a[k,i]*(b[p[k],p[i]] - b[p[k],p[j]]) + a[k,j]*(b[p[k],p[j]] - b[p[k],p[i]])
          + a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]]) + a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    
    Simplified: For swap of locations li=p[i], lj=p[j]:
    delta = sum_{k != i,j} [a[i,k]*(b[li,p[k]] - b[lj,p[k]]) + a[j,k]*(b[lj,p[k]] - b[li,p[k]])
                            + a[k,i]*(b[p[k],li] - b[p[k],lj]) + a[k,j]*(b[p[k],lj] - b[p[k],li])]
            + a[i,j]*(b[li,lj] - b[lj,li]) + a[j,i]*(b[lj,li] - b[li,lj])
    """
    li = p[i]
    lj = p[j]
    
    # Terms involving i and j with all other k
    # a[i,:] * (b[li,:] - b[lj,:]) - exclude i,j
    # a[j,:] * (b[lj,:] - b[li,:]) - exclude i,j
    # a[:,i] * (b[:,li] - b[:,lj]) - exclude i,j
    # a[:,j] * (b[:,lj] - b[:,li]) - exclude i,j
    
    diff_b_row = b[li, :] - b[lj, :]
    diff_b_col = b[:, li] - b[:, lj]
    
    delta = 0.0
    
    # a[i,k] * (b[li,k] - b[lj,k]) for k != i,j
    delta += np.sum(a[i, :] * diff_b_row) - a[i, i] * diff_b_row[i] - a[i, j] * diff_b_row[j]
    
    # a[j,k] * (b[lj,k] - b[li,k]) for k != i,j
    delta += np.sum(a[j, :] * (-diff_b_row)) - a[j, i] * (-diff_b_row[i]) - a[j, j] * (-diff_b_row[j])
    
    # a[k,i] * (b[k,li] - b[k,lj]) for k != i,j
    delta += np.sum(a[:, i] * diff_b_col) - a[i, i] * diff_b_col[i] - a[j, i] * diff_b_col[j]
    
    # a[k,j] * (b[k,lj] - b[k,li]) for k != i,j
    delta += np.sum(a[:, j] * (-diff_b_col)) - a[i, j] * (-diff_b_col[i]) - a[j, j] * (-diff_b_col[j])
    
    # Cross terms between i and j
    # a[i,j] * (b[li,lj] - b[lj,li]) + a[j,i] * (b[lj,li] - b[li,lj])
    cross = a[i, j] * (b[li, lj] - b[lj, li]) + a[j, i] * (b[lj, li] - b[li, lj])
    delta += cross
    
    return int(delta)


def _local_search(a, b, p, time_limit=None, start_time=None):
    """2-opt local search: repeatedly find best improving swap until no improvement.
    
    Uses vectorized delta computation for efficiency.
    """
    if start_time is None:
        start_time = time.time()
    n = len(p)
    cost = _compute_cost(a, b, p)
    improved = True
    
    while improved:
        if time_limit is not None and time.time() - start_time > time_limit - 2:
            break
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Vectorized computation of all swap deltas
        # For each pair (i,j), compute delta efficiently
        # This is O(n^2) per iteration but with vectorized inner loops
        
        # Precompute for efficiency
        # For swap (i,j) with li=p[i], lj=p[j]:
        # delta = sum_k [a[i,k]*(b[li,k]-b[lj,k]) + a[j,k]*(b[lj,k]-b[li,k]) 
        #                 + a[k,i]*(b[k,li]-b[k,lj]) + a[k,j]*(b[k,lj]-b[k,li])] + cross
        
        # Let's compute deltas for all pairs using matrix operations
        # delta(i,j) = a[i,:]·(b[li,:]-b[lj,:]) + a[j,:]·(b[lj,:]-b[li,:])
        #             + a[:,i]·(b[:,li]-b[:,lj]) + a[:,j]·(b[:,lj]-b[:,li]) + cross(i,j)
        
        # This is complex to fully vectorize. Use a semi-vectorized approach.
        # Iterate over i, vectorize over j
        
        for i in range(n):
            li = p[i]
            # For this i, compute deltas for all j > i
            # a[i,:] * (b[li,:] - b[p[j],:]) for all j
            # This is a_i @ (b[li] - b[p[j]]) for each j
            
            # Vectorize: for all j, compute a[i,:]·b[li,:] - a[i,:]·b[p[j],:]
            a_i_b_li = np.dot(a[i, :], b[li, :])
            a_i_b_all = a[i, :] @ b[p[i], :]  # not quite, need b[p[j],:] for each j
            
            # Better: a[i,:] @ b.T gives a[i,:]·b[k,:] for all k
            a_i_b_rows = a[i, :] @ b.T  # shape (n,), a_i_b_rows[k] = a[i,:]·b[k,:]
            a_i_part = a_i_b_li - a_i_b_rows[p]  # a[i,:]·(b[li,:] - b[p[j],:]) for all j
            
            # Similarly for a[j,:]·(b[p[j],:] - b[li,:])
            # = a[j,:]·b[p[j],:] - a[j,:]·b[li,:]
            # a[j,:]·b[p[j],:] is the diagonal element
            a_diag_b = np.array([np.dot(a[j, :], b[p[j], :]) for j in range(n)])
            a_all_b_li = b[li, :] @ a.T  # b[li,:]·a[j,:] for all j = a[:,li]·b[li,:]... 
            # Actually a[j,:]·b[li,:] = (a.T @ b[li,:])[j]
            a_j_b_li = a.T @ b[li, :]  # shape (n,), a_j_b_li[j] = a[j,:]·b[li,:]
            
            a_j_part = a_diag_b - a_j_b_li  # a[j,:]·(b[p[j],:] - b[li,:]) for all j
            
            # a[:,i]·(b[:,li]-b[:,lj]) for all j
            # a[:,i]·b[:,li] - a[:,i]·b[:,lj]
            a_col_i_b_li = np.dot(a[:, i], b[:, li])
            a_col_i_b_all = a[:, i] @ b  # a[:,i]·b[k,:] for all k... no
            # a[:,i]·b[:,k] = (a[:,i] @ b)[k]
            a_col_i_b_rows = a[:, i] @ b  # shape (n,), [k] = a[:,i]·b[:,k]
            a_ki_part = a_col_i_b_li - a_col_i_b_rows[p]  # a[:,i]·(b[:,li]-b[:,p[j]]) for all j
            
            # a[:,j]·(b[:,lj]-b[:,li]) for all j
            # = a[:,j]·b[:,lj] - a[:,j]·b[:,li]
            # a[:,j]·b[:,p[j]] is diagonal-like
            a_col_diag_b = np.array([np.dot(a[:, j], b[:, p[j]]) for j in range(n)])
            a_col_all_b_li = b[:, li] @ a  # b[:,li]·a[:,j] for all j
            a_kj_part = a_col_diag_b - a_col_all_b_li
            
            # Cross term: a[i,j]*(b[li,p[j]] - b[p[j],li]) + a[j,i]*(b[p[j],li] - b[li,p[j]])
            # = (a[i,j] - a[j,i]) * (b[li,p[j]] - b[p[j],li])
            # For all j: (a[i,:] - a[:,i].T) * (b[li,p] - b[p,li])
            diff_a = a[i, :] - a[:, i]
            diff_b_cross = b[li, p] - b[p, li]
            cross_part = diff_a * diff_b_cross
            
            # Total delta for all j
            deltas = a_i_part + a_j_part + a_ki_part + a_kj_part + cross_part
            
            # Find best j > i with negative delta
            mask = np.arange(n) > i
            if not np.any(mask):
                continue
            deltas_masked = deltas.copy()
            deltas_masked[~mask] = 0  # Only consider j > i
            best_j_local = np.argmin(deltas_masked)
            best_delta_local = deltas_masked[best_j_local]
            
            if best_delta_local < best_delta:
                best_delta = best_delta_local
                best_i = i
                best_j = best_j_local
        
        if best_i >= 0 and best_delta < 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            improved = True
    
    return p, cost


def _perturb(p, n_perturbs):
    """Randomly swap n_perturbs pairs of elements."""
    n = len(p)
    p = p.copy()
    for _ in range(n_perturbs):
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]]."""
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = len(a)
    np.random.seed(42)
    
    # Step 1: Spectral initialization
    p = _spectral_init(a, b)
    cost = _compute_cost(a, b, p)
    
    best_p = p.copy()
    best_cost = cost
    
    # Step 2: Local search from spectral init
    p, cost = _local_search(a, b, p, time_limit=60, start_time=start_time)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    # Step 3: Iterated Local Search
    # Try multiple perturbations and local searches
    n_perturbs_list = [2, 3, 4, 5, 7, 10]
    pert_idx = 0
    il_iter = 0
    
    while time.time() < deadline - 10:  # Leave 10s margin
        il_iter += 1
        
        # Choose perturbation size
        np_ = n_perturbs_list[pert_idx % len(n_perturbs_list)]
        pert_idx += 1
        
        # Perturb from best solution
        p_pert = _perturb(best_p, np_)
        
        # Local search from perturbed solution
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_pert, time_limit=remaining_time - 2, start_time=time.time())
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Occasionally re-seed with spectral init for diversity
        if il_iter % 5 == 0 and time.time() < deadline - 30:
            p_spec = _spectral_init(a, b)
            p_spec, cost_spec = _local_search(a, b, p_spec, time_limit=20, start_time=time.time())
            if cost_spec < best_cost:
                best_cost = cost_spec
                best_p = p_spec.copy()
    
    # Final: try a few more local searches from best if time remains
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() >= deadline - 5:
                break
            p_try = _perturb(best_p, 1)
            p_try, cost_try = _local_search(a, b, p_try, time_limit=deadline - time.time() - 3, start_time=time.time())
            if cost_try < best_cost:
                best_cost = cost_try
                best_p = p_try.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
