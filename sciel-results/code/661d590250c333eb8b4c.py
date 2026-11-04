import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 factors
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, col_off):
        for t in range(7):
            col = col_off + t
            for lr in range(2):
                for lc in range(2):
                    val = s_u[t][lr*2+lc]
                    if val != 0:
                        U[(a_r + lr) * m + (a_c + lc), col] += val
                    val = s_v[t][lr*2+lc]
                    if val != 0:
                        V[(b_r + lr) * p + (b_c + lc), col] += val
                    val = s_w[t][lr*2+lc]
                    if val != 0:
                        W[(c_c + lc) * n + (c_r + lr), col] += val
                        
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2, 0, 2, 7)
    add_strassen_block(0, 2, 2, 0, 0, 0, 14)
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    col_off = 28
    for j in range(4):
        for k in range(4):
            col = col_off + j * 4 + k
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
    
    # Verify correctness of initial decomposition
    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        # Tensor check: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1 if valid
        # Actually the decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
        # U has shape (n*m, r), indexed as (i*m + j, r)
        # V has shape (m*p, r), indexed as (j*p + k, r)
        # W has shape (p*n, r), indexed as (k*n + i, r)
        # The product should satisfy: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k) contribution
        # More precisely: the tensor T[i,j,k] = delta(i,j,k) (for matmul A*B where C[i,k]=sum_j A[i,j]B[j,k])
        # Wait, for matrix multiplication A(n×m) * B(m×p) = C(n×p):
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the j dimension)
        # Actually no. The tensor for matmul is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Because C[i,k] = sum_j A[i,j] * B[j,k], so the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
        
        # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is expensive for large r, so we'll do it efficiently using einsum or matrix ops
        
        # Reshape and use tensor contraction
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r U.reshape(n,m,r)[i,j,r] * V.reshape(m,p,r)[j,k,r] * W.reshape(p,n,r)[k,i,r]
        
        U_r = U_mat.reshape(n, m, r)
        V_r = V_mat.reshape(m, p, r)
        W_r = W_mat.reshape(p, n, r)
        
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # T should be all ones
        return np.allclose(T, 1.0, atol=1e-10)
    
    if not verify(U, V, W):
        # Fallback: return as-is (shouldn't happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Local search: try to find linear dependencies to reduce rank
    # Strategy: check if any column of the combined tensor is a linear combination of others
    # The "combined tensor" for each rank term r is the vector:
    # vec_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] (flattened)
    # We want to find if any vec_r is in the span of the others.
    
    # More efficient: check rank of the matrix whose columns are the rank-r terms
    # Each term r contributes a vector of dimension n*m*p = 48
    # The full tensor is all-ones (48 entries)
    # We need sum_r u_r ⊗ v_r ⊗ w_r = T (the all-ones tensor)
    
    # To check if we can reduce rank by 1:
    # Check if any term is a linear combination of the others.
    # Build matrix M where column r is the flattened tensor term u_r ⊗ v_r ⊗ w_r
    # Then check if rank(M) < rank (number of columns)
    
    def get_flattened_terms(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        nmp = n * m * p
        terms = np.zeros((nmp, r))
        for r_idx in range(r):
            u = U_mat[:, r_idx]  # (n*m,)
            v = V_mat[:, r_idx]  # (m*p,)
            w = W_mat[:, r_idx]  # (p*n,)
            # u[i*m+j], v[j*p+k], w[k*n+i]
            # Term[i*m + j, j*p + k, k*n + i] = u[i*m+j] * v[j*p+k] * w[k*n+i]
            # Flatten index: (i*m+j) * (m*p) + (j*p+k) ... no, 3D flatten
            # Actually we need to build the 3D tensor and flatten
            u3 = u.reshape(n, m)
            v3 = v.reshape(m, p)
            w3 = w.reshape(p, n)
            # T[i,j,k] = u3[i,j] * v3[j,k] * w3[k,i]
            term_3d = np.einsum('ij,jk,ki->ijk', u3, v3, w3)
            terms[:, r_idx] = term_3d.flatten()
        return terms
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_reductions=5):
        current_U = U_mat.copy()
        current_V = V_mat.copy()
        current_W = W_mat.copy()
        current_rank = current_U.shape[1]
        
        for reduction in range(max_reductions):
            if time.time() > deadline - 10:
                break
            
            terms = get_flattened_terms(current_U, current_V, current_W)
            nmp = n * m * p
            
            # Check if any column is a linear combination of the others
            # For each column r, check if it's in the span of the others
            # Use SVD: rank of terms matrix
            rank_of_terms = np.linalg.matrix_rank(terms, tol=1e-10)
            
            if rank_of_terms >= current_rank:
                # No dependency found, try a different approach
                # Try checking pairs of terms that might combine
                break
            
            # Find which column is dependent
            # Compute null space of the transpose or find dependent column
            # Use: for each column, compute residual after projecting onto others
            for r_idx in range(current_rank):
                if time.time() > deadline - 10:
                    break
                
                # Check if column r_idx is in span of others
                others = np.delete(terms, r_idx, axis=1)
                # Solve: others @ x ≈ terms[:, r_idx]
                try:
                    x, residuals, rank_others, sv = np.linalg.lstsq(others, terms[:, r_idx], rcond=1e-10)
                    residual = np.linalg.norm(others @ x - terms[:, r_idx])
                    if residual < 1e-8:
                        # Column r_idx is dependent! Remove it.
                        # But we need to maintain the decomposition.
                        # If term r_idx = sum_{r' != r_idx} x[r'-adj] * term r'
                        # This means the tensor contribution of r_idx equals sum of others
                        # So we can simply remove column r_idx from U, V, W
                        current_U = np.delete(current_U, r_idx, axis=1)
                        current_V = np.delete(current_V, r_idx, axis=1)
                        current_W = np.delete(current_W, r_idx, axis=1)
                        current_rank -= 1
                        break
                except:
                    continue
            
            if current_rank == current_U.shape[1]:
                break  # No reduction made
        
        return current_U, current_V, current_W, current_rank
    
    # Attempt rank reduction
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    try:
        U_red, V_red, W_red, new_rank = try_reduce_rank(U, V, W, max_reductions=10)
        if new_rank < best_rank and verify(U_red, V_red, W_red):
            best_rank = new_rank
            best_U, best_V, best_W = U_red, V_red, W_red
    except:
        pass
    
    # Additional: try random linear combinations of terms to find dependencies
    # This is a heuristic: if we can find a set of coefficients that makes one term vanish
    if time.time() < deadline - 20:
        try:
            rng = np.random.default_rng(42)
            for attempt in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Try to find a linear relation among a subset of terms
                # Pick a random target term and see if it can be expressed via others
                target = rng.integers(0, best_rank)
                others_idx = [i for i in range(best_rank) if i != target]
                
                if len(others_idx) < 2:
                    continue
                
                terms = get_flattened_terms(best_U, best_V, best_W)
                target_vec = terms[:, target]
                others_mat = terms[:, others_idx]
                
                try:
                    x, residuals, rk, sv = np.linalg.lstsq(others_mat, target_vec, rcond=1e-10)
                    residual = np.linalg.norm(others_mat @ x - target_vec)
                    if residual < 1e-8:
                        # Found dependency! Remove target
                        keep = [i for i in range(best_rank) if i != target]
                        best_U = best_U[:, keep]
                        best_V = best_V[:, keep]
                        best_W = best_W[:, keep]
                        best_rank -= 1
                        if verify(best_U, best_V, best_W):
                            break
                        else:
                            # Revert
                            best_U = np.hstack([best_U, np.zeros((n*m, 1))])
                            best_V = np.hstack([best_V, np.zeros((m*p, 1))])
                            best_W = np.hstack([best_W, np.zeros((p*n, 1))])
                            best_rank += 1
                except:
                    continue
        except:
            pass
    
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
