import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 if it corresponds to a valid term in C = AB
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So term A[i,j] B[j,k] contributes to C[i,k]
    # Index in U: i*m + j
    # Index in V: j*p + k
    # Index in W: k*n + i
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Try to find a low rank decomposition
    # Start from a low rank and increase
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook as fallback
    U_school = np.zeros((size_u, best_rank))
    V_school = np.zeros((size_v, best_rank))
    W_school = np.zeros((size_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx, col] = 1
                V_school[v_idx, col] = 1
                W_school[w_idx, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Try ranks from 30 to best_rank - 1
    for rank in range(30, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts for each rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randn(size_u, rank)
            V = np.random.randn(size_v, rank)
            W = np.random.randn(size_w, rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                # T_uvw = sum_c U_uc V_vc W_wc
                # For fixed V, W, U is linear in each entry?
                # Residual R = T - sum_c U_c V_c W_c
                # We want to minimize ||R||^2
                # This is equivalent to solving a linear system for each row of U?
                # Or simply gradient descent.
                # Given the small size, we can use a simple least squares approach.
                # Flatten T and use matrix multiplication?
                # T = U @ (V^T * W)? No.
                
                # Let's use a simple gradient step or LS.
                # Compute current tensor
                # T_est = sum_c np.outer(np.outer(U[:,c], V[:,c]), W[:,c])
                # This is slow.
                
                # Better:
                # Fix V, W. Compute M_u = sum_c V[:,c] W[:,c] (outer product summed over c)
                # Then T_est[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
                # For fixed u, T_est[u, :, :] = sum_c U[u,c] (V[:,c] W[:,c]^T)?
                # Let M_vw[c] = V[:,c] * W[:,c] (elementwise)
                # Then T_est[u, v, w] = sum_c U[u,c] M_vw[c, v, w]
                # This is a linear system for U[:, u] given T[u, :, :].
                
                # Reshape M_vw to (rank, size_v * size_w)
                M_vw = np.einsum('vc,wc->vcw', V, W)
                M_vw_flat = M_vw.reshape(rank, -1)
                
                # Solve U[:, u] such that U[:, u] @ M_vw_flat.T approx T[u, :, :].flat
                # Least squares: U[:, u] = (M_vw_flat @ M_vw_flat.T)^-1 M_vw_flat @ T[u].flat
                # Using np.linalg.lstsq
                T_u_flat = T[u, :, :].reshape(-1)
                try:
                    U[:, u], residuals, rank_ls, sv = np.linalg.lstsq(M_vw_flat, T_u_flat, rcond=None)
                except:
                    break
                    
                # Update V
                M_uw = np.einsum('uc,wc->ucw', U, W)
                M_uw_flat = M_uw.reshape(rank, -1)
                for v in range(size_v):
                    T_v_flat = T[:, v, :].reshape(-1)
                    try:
                        V[:, v], residuals, rank_ls, sv = np.linalg.lstsq(M_uw_flat, T_v_flat, rcond=None)
                    except:
                        break
                        
                # Update W
                M_uv = np.einsum('uc,vc->ucv', U, V)
                M_uv_flat = M_uv.reshape(rank, -1)
                for w in range(size_w):
                    T_w_flat = T[:, :, w].reshape(-1)
                    try:
                        W[:, w], residuals, rank_ls, sv = np.linalg.lstsq(M_uv_flat, T_w_flat, rcond=None)
                    except:
                        break
                        
                # Check error
                if it % 10 == 0:
                    T_est = np.einsum('uc,vc,wc->uvw', U, V, W)
                    err = np.linalg.norm(T - T_est)
                    if err < 1e-9:
                        found = True
                        break
                        
            if found:
                best_result = (U, V, W, rank)
                break
                
        if found:
            break
            
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to lists of lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = U_school.tolist()
        V_list = V_school.tolist()
        W_list = W_school.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
