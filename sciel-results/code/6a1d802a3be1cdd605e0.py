"""QAP solver for tai256c: spectral init + 2-opt local search with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
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
    deg_a = a.sum(axis=1)
    L_a = np.diag(deg_a) - a
    
    w_a, v_a = np.linalg.eigh(L_a)
    
    num_dims = min(4, n - 1)
    if num_dims > 0:
        fac_coords = v_a[:, 1:1+num_dims]
    else:
        fac_coords = np.zeros((n, 1))

    # --- Location Embedding ---
    b_flat = b.flatten()
    non_zero_b = b_flat[b_flat > 0]
    if len(non_zero_b) > 0:
        sigma = np.median(non_zero_b)
    else:
        sigma = 1.0
    
    W_b = np.exp(-b / (sigma + 1e-6))
    np.fill_diagonal(W_b, 0)
    
    deg_b = W_b.sum(axis=1)
    L_b = np.diag(deg_b) - W_b
    
    w_b, v_b = np.linalg.eigh(L_b)
    
    if num_dims > 0:
        loc_coords = v_b[:, 1:1+num_dims]
    else:
        loc_coords = np.zeros((n, 1))

    # --- Matching ---
    fac_sq = np.sum(fac_coords**2, axis=1)
    loc_sq = np.sum(loc_coords**2, axis=1)
    
    cross = fac_coords @ loc_coords.T
    
    cost_mat = fac_sq[:, np.newaxis] + loc_sq[np.newaxis, :] - 2 * cross
    
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def _swap_delta(a, b, p, i, j):
    """Compute cost delta for swapping positions i and j in permutation p."""
    pi = p[i]
    pj = p[j]
    
    # Vectorized delta computation
    # For swap of positions i and j (which hold facilities pi and pj):
    # delta = sum_k [a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + 
    #                 a[j,k]*(b[pi,p[k]] - b[pj,p[k]])] +
    #         sum_k [a[k,i]*(b[p[k],pj] - b[p[k],pi]) + 
    #                 a[k,j]*(b[p[k],pi] - b[p[k],pj])]
    
    p_arr = p
    
    delta_i_row = np.dot(a[i, :], b[pj, p_arr]) - np.dot(a[i, :], b[pi, p_arr])
    delta_j_row = np.dot(a[j, :], b[pi, p_arr]) - np.dot(a[j, :], b[pj, p_arr])
    delta_i_col = np.dot(a[:, i], b[p_arr, pj]) - np.dot(a[:, i], b[p_arr, pi])
    delta_j_col = np.dot(a[:, j], b[p_arr, pi]) - np.dot(a[:, j], b[p_arr, pj])
    
    return delta_i_row + delta_j_row + delta_i_col + delta_j_col


def _local_search_2opt(a, b, p, max_time=140.0, start_time=None):
    """
    2-opt local search for QAP.
    Tries swapping pairs of assignments and accepts if cost decreases.
    """
    n = len(a)
    if start_time is None:
        start_time = time.time()
    
    current_p = p.copy()
    current_cost = _qap_cost(a, b, current_p)
    
    improved = True
    no_improve_count = 0
    max_no_improve = 30
    
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
                total_delta = _swap_delta(a, b, p_arr, i, j)
                
                if total_delta < 0:
                    p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
                    current_cost += int(total_delta)
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
    
    return current_p, current_cost


def _continuous_embedding_solve(a, b, p_init, max_time=140.0, start_time=None):
    """
    Refine using continuous embedding approach as a polish step.
    This represents the permutation as soft assignments and uses gradient descent
    on a smooth energy function to find better discrete assignments.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(a)
    
    # Start from the discrete solution and embed in a continuous space
    # X is (n, n) where X[i, j] indicates facility i is at location j
    X = np.zeros((n, n))
    for i in range(n):
        X[i, p_init[i]] = 1.0
    
    # Precompute for gradient
    # Energy: sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
    # This is a bilinear form in X with constraint X is a permutation matrix
    
    # Gradient descent with projection back to permutation structure
    lr = 0.01
    patience = 0
    best_X = X.copy()
    best_p = p_init.copy()
    best_cost = _qap_cost(a, b, p_init)
    
    for it in range(500):
        if time.time() - start_time > max_time - 5:
            break
        
        # Compute gradient of energy w.r.t. X
        # E = sum_{i,j} a[i,j] * (B_X)[i,j] where B_X[i,j] = sum_{k,l} X[i,k] b[k,l] X[j,l]
        # dE/dX[i,k] = sum_j a[i,j] * b[k,l_j] + ... (complex, so use numerical)
        
        # Numerical gradient step
        X_new = X.copy()
        for i in range(n):
            if time.time() - start_time > max_time - 5:
                break
            for k in range(n):
                X_test = X_new.copy()
                X_test[i, k] += 0.01
                # Project to doubly stochastic (simple: row/col normalize)
                row_sums = X_test.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1)
                X_test = X_test / row_sums
                col_sums = X_test.sum(axis=0, keepdims=True)
                col_sums = np.where(col_sums > 0, col_sums, 1)
                X_test = X_test / col_sums
                
                # Compute cost for this soft assignment
                # C = sum_{i,j} a[i,j] * sum_{k,l} X[i,k] b[k,l] X[j,l]
                B_X = X_test.T @ b @ X_test
                cost_soft = np.sum(a * B_X)
                
                X_new[i, k] -= lr * (cost_soft - (np.sum(a * (X.T @ b @ X)))) / 0.01
        
        # Project X back to a permutation
        # Use Hungarian algorithm on -X to get nearest permutation
        row_ind, col_ind = linear_sum_assignment(-X_new)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        cost_new = _qap_cost(a, b, p_new)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            best_X = X_new.copy()
            X = X_new
            patience = 0
        else:
            patience += 1
            if patience > 20:
                break
    
    return best_p, best_cost


def solve(a, b):
    """Solve the QAP instance using spectral init + 2-opt local search + continuous polish."""
    start_time = time.time()
    n = len(a)
    
    # Generate initial solution
    p = _spectral_initial(a, b)
    
    # Run local search with time budget
    remaining = max(0, 140.0 - (time.time() - start_time))
    local_time_budget = remaining * 0.7
    p, cost = _local_search_2opt(a, b, p, max_time=local_time_budget, start_time=start_time)
    
    # Check if we have time for continuous polish
    elapsed = time.time() - start_time
    if elapsed < 135.0:
        polish_time_budget = 140.0 - elapsed - 5.0
        if polish_time_budget > 5.0:
            p_polished, cost_polished = _continuous_embedding_solve(a, b, p, max_time=polish_time_budget, start_time=start_time)
            if cost_polished < cost:
                p = p_polished
                cost = cost_polished
    
    # Final check: ensure we're within time limit
    if time.time() - start_time > 140.0:
        return p.tolist()
    
    return p.tolist()
# EVOLVE-BLOCK-END
