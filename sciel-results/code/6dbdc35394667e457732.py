import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1 (the matrix multiplication tensor)."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_schoolbook_decomposition(n, m, p):
    """Construct a rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_outer_products(U, V, W, n, m, p):
    """Precompute the flattened outer product for each term."""
    rank = U.shape[1]
    outers = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        outers[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return outers

def _try_reduce_rank(U, V, W, n, m, p, max_reductions, time_budget):
    """
    Iteratively try to reduce the rank by finding linear dependencies 
    in the tensor terms and eliminating them.
    """
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    outers = _compute_outer_products(U_cur, V_cur, W_cur, n, m, p)
    current_rank = U_cur.shape[1]
    
    for _ in range(max_reductions):
        if time.time() - start_time > time_budget:
            break
            
        # Find null space of the matrix of outer products
        # outers has shape (n*m*p, current_rank)
        # We want to find a vector alpha such that outers @ alpha = 0
        # This is the null space of outers.T
        try:
            # Use SVD to find null space
            U_svd, s, Vt = np.linalg.svd(outers, full_matrices=False)
            # The last row of Vt corresponds to the smallest singular value
            # If the smallest singular value is close to 0, we have a dependency
            if s[-1] < 1e-8:
                # The null space vector is the last row of Vt (transposed)
                alpha = Vt[-1, :]
                
                # Find the index of the largest coefficient in alpha to eliminate
                # We want to eliminate one term, so we pick the one with max |alpha|
                idx_to_eliminate = np.argmax(np.abs(alpha))
                
                # Construct the new coefficients for the remaining terms
                # alpha[idx_to_eliminate] * T_idx + sum_{r != idx} alpha[r] * T_r = 0
                # => T_idx = - (1/alpha[idx]) * sum_{r != idx} alpha[r] * T_r
                
                # We need to update U, V, W such that the new decomposition is valid.
                # The new term for idx is replaced by a linear combination of others.
                # Actually, we just drop the term idx and add its contribution to the others.
                
                # Let's verify the dependency first
                recon = outers @ alpha
                if np.allclose(recon, 0, atol=1e-8):
                    # The dependency is valid.
                    # We eliminate term idx_to_eliminate.
                    # The new U, V, W will have rank current_rank - 1.
                    # We need to adjust the remaining terms.
                    
                    # The equation is: sum_r alpha[r] * T_r = 0
                    # T_idx = - (1/alpha[idx]) * sum_{r != idx} alpha[r] * T_r
                    # This means we can remove T_idx and add -alpha[r]/alpha[idx] * T_r to the others?
                    # No, the decomposition is sum T_r = T_target.
                    # If sum alpha[r] T_r = 0, then sum (1 + alpha[r]) T_r = T_target + sum alpha[r] T_r = T_target.
                    # This doesn't help eliminate a term directly unless we can set a coefficient to 0.
                    
                    # Alternative approach:
                    # If T_idx = sum_{r != idx} c_r T_r, then we can remove T_idx and add c_r T_r to the others?
                    # No, that changes the sum.
                    
                    # Correct approach:
                    # We have a linear dependence: sum_r alpha[r] T_r = 0.
                    # This means the vectors are linearly dependent.
                    # We can remove one vector (say T_idx) and express it as a combination of the others.
                    # But the decomposition is a SUM of rank-1 tensors.
                    # If T_idx = sum_{r != idx} c_r T_r, then sum_{r != idx} (T_r + c_r T_r) is not correct.
                    
                    # Let's rethink.
                    # The tensor is T = sum_{r=1}^R T_r.
                    # If T_idx = sum_{r != idx} c_r T_r, then T = sum_{r != idx} T_r + sum_{r != idx} c_r T_r = sum_{r != idx} (1 + c_r) T_r.
                    # This reduces the rank by 1!
                    
                    # So we need to find c_r such that T_idx = sum_{r != idx} c_r T_r.
                    # This is equivalent to finding alpha such that alpha[idx] T_idx + sum_{r != idx} alpha[r] T_r = 0.
                    # Then T_idx = - (1/alpha[idx]) sum_{r != idx} alpha[r] T_r.
                    # So c_r = - alpha[r] / alpha[idx].
                    
                    # Update the coefficients of the remaining terms.
                    # U_new[:, r] = U[:, r] + c_r * U[:, idx]
                    # V_new[:, r] = V[:, r] + c_r * V[:, idx]
                    # W_new[:, r] = W[:, r] + c_r * W[:, idx]
                    
                    indices = [i for i in range(current_rank) if i != idx_to_eliminate]
                    U_new = np.zeros((n * m, current_rank - 1))
                    V_new = np.zeros((m * p, current_rank - 1))
                    W_new = np.zeros((p * n, current_rank - 1))
                    
                    for new_r, orig_r in enumerate(indices):
                        c_r = -alpha[orig_r] / alpha[idx_to_eliminate]
                        U_new[:, new_r] = U_cur[:, orig_r] + c_r * U_cur[:, idx_to_eliminate]
                        V_new[:, new_r] = V_cur[:, orig_r] + c_r * V_cur[:, idx_to_eliminate]
                        W_new[:, new_r] = W_cur[:, orig_r] + c_r * W_cur[:, idx_to_eliminate]
                    
                    # Verify the new decomposition
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                        U_cur, V_cur, W_cur = U_new, V_new, W_new
                        outers = _compute_outer_products(U_cur, V_cur, W_cur, n, m, p)
                        current_rank = U_cur.shape[1]
                        continue
        except Exception:
            pass
        
        # If no reduction was made, break
        break

    return U_cur, V_cur, W_cur

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the schoolbook decomposition (rank 48)
    U48, V48, W48 = _construct_schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U48.copy(), V48.copy(), W48.copy()
    best_rank = 48

    # Try to reduce rank
    U_try, V_try, W_try = _try_reduce_rank(
        best_U, best_V, best_W, n, m, p,
        max_reductions=10,
        time_budget=100
    )
    if U_try.shape[1] < best_rank:
        best_U, best_V, best_W = U_try, V_try, W_try
        best_rank = U_try.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
