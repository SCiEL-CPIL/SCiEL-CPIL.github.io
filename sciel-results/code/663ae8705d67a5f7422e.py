"""QAP solver for tai256c: spectral init + 2-opt local search with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # Use vectorized computation for speed
    # p is a permutation array
    # b_perm[i, j] = b[p[i], p[j]]
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _spectral_initial(a, b):
    """
    Generate an initial solution using spectral partitioning/matching.
    1. Compute spectral embedding of facilities based on flow matrix A.
    2. Compute spectral embedding of locations based on distance matrix B (converted to similarity).
    3. Match facility embeddings to location embeddings using Linear Sum Assignment.
    """
    n = len(a)
    
    # --- Facility Embedding ---
    # Use the Laplacian of the flow graph A
    # L = D - A
    deg_a = a.sum(axis=1)
    L_a = np.diag(deg_a) - a
    
    # Compute eigenvectors
    # We only need the first few non-trivial eigenvectors
    # np.linalg.eigh returns eigenvalues in ascending order
    w_a, v_a = np.linalg.eigh(L_a)
    
    # Use the first 2 non-trivial eigenvectors (indices 1 and 2)
    # If n is small, adjust
    num_dims = min(2, n - 1)
    if num_dims > 0:
        fac_coords = v_a[:, 1:1+num_dims]
    else:
        fac_coords = np.zeros((n, 1))

    # --- Location Embedding ---
    # Convert distance matrix B to a similarity matrix W
    # W_ij = exp(-B_ij / sigma)
    # Estimate sigma as the median of non-zero distances
    b_flat = b.flatten()
    non_zero_b = b_flat[b_flat > 0]
    if len(non_zero_b) > 0:
        sigma = np.median(non_zero_b)
    else:
        sigma = 1.0
    
    # Add small epsilon to avoid division by zero or overflow
    W_b = np.exp(-b / (sigma + 1e-6))
    # Set diagonal to 0 for the graph Laplacian
    np.fill_diagonal(W_b, 0)
    
    deg_b = W_b.sum(axis=1)
    L_b = np.diag(deg_b) - W_b
    
    w_b, v_b = np.linalg.eigh(L_b)
    
    if num_dims > 0:
        loc_coords = v_b[:, 1:1+num_dims]
    else:
        loc_coords = np.zeros((n, 1))

    # --- Matching ---
    # Minimize sum of squared Euclidean distances between matched points
    # dist(i, j) = ||fac_coords[i] - loc_coords[j]||^2
    #            = ||fac_coords[i]||^2 + ||loc_coords[j]||^2 - 2 * fac_coords[i] . loc_coords[j]
    
    fac_sq = np.sum(fac_coords**2, axis=1)
    loc_sq = np.sum(loc_coords**2, axis=1)
    
    # Cross term: (n, d) @ (d, n) -> (n, n)
    cross = fac_coords @ loc_coords.T
    
    # Cost matrix for LAP
    cost_mat = fac_sq[:, np.newaxis] + loc_sq[np.newaxis, :] - 2 * cross
    
    # Solve Linear Sum Assignment Problem
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    # p[i] is the location assigned to facility i
    # row_ind[k] is facility index, col_ind[k] is location index
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def _local_search_2opt(a, b, p, max_time=None, start_time=None):
    """
    2-opt local search for QAP.
    Tries swapping pairs of assignments and accepts if cost decreases.
    Uses incremental cost updates for efficiency.
    """
    n = len(a)
    if max_time is None:
        max_time = 140.0
    if start_time is None:
        start_time = time.time()
    
    current_p = p.copy()
    current_cost = _qap_cost(a, b, current_p)
    
    # Precompute row and column views of a and b for faster access
    # a_rows[i] is a[i, :]
    # b_rows[l] is b[l, :]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50  # Stop if no improvement for this many full passes
    
    while improved and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        
        if time.time() - start_time > max_time:
            break
        
        p_arr = current_p
        
        for i in range(n):
            if time.time() - start_time > max_time:
                return current_p, current_cost
            
            for j in range(i + 1, n):
                # Try swapping positions i and j
                pi = p_arr[i]
                pj = p_arr[j]
                
                # Compute cost delta
                # Delta = sum_k a[i,k](b[pj,p[k]] - b[pi,p[k]]) + 
                #         sum_k a[j,k](b[pi,p[k]] - b[pj,p[k]]) +
                #         sum_k a[k,i](b[p[k],pj] - b[p[k],pi]) +
                #         sum_k a[k,j](b[p[k],pi] - b[p[k],pj])
                
                # Vectorized computation
                # b[pj, p_k] is b_rows[pj][p_k]
                # b[pi, p_k] is b_rows[pi][p_k]
                
                # Note: p_k is the whole array p_arr
                # We can use fancy indexing
                
                # Optimization: Precompute b_rows and b_cols?
                # b is already a numpy array.
                
                delta_i_row = np.dot(a[i, :], b[pj, p_arr]) - np.dot(a[i, :], b[pi, p_arr])
                delta_j_row = np.dot(a[j, :], b[pi, p_arr]) - np.dot(a[j, :], b[pj, p_arr])
                delta_i_col = np.dot(a[:, i], b[p_arr, pj]) - np.dot(a[:, i], b[p_arr, pi])
                delta_j_col = np.dot(a[:, j], b[p_arr, pi]) - np.dot(a[:, j], b[p_arr, pj])
                
                total_delta = delta_i_row + delta_j_row + delta_i_col + delta_j_col
                
                if total_delta < 0:
                    # Accept the swap
                    p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
                    current_cost += total_delta
                    improved = True
                    no_improve_count = 0
                    break  # Restart the j loop with updated permutation
        
        if not improved:
            no_improve_count += 1
    
    return p_arr, current_cost


def solve(a, b):
    """Solve the QAP instance using spectral init + 2-opt local search."""
    start_time = time.time()
    n = len(a)
    
    # Generate initial solution
    p = _spectral_initial(a, b)
    
    # Run local search with time budget
    max_time = 140.0  # Leave 10s safety margin
    p, cost = _local_search_2opt(a, b, p, max_time=max_time, start_time=start_time)
    
    # Final check: ensure we're within time limit
    if time.time() - start_time > max_time - 5:
        return p.tolist()
    
    return p.tolist()
# EVOLVE-BLOCK-END
