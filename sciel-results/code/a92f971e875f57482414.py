import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start time
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Dimensions: (n*m, m*p, p*n)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    
    # Function to check if decomposition is valid
    def check_decomposition(U, V, W, tol=1e-6):
        # U: (dim_u, R), V: (dim_v, R), W: (dim_w, R)
        # Reconstruct tensor
        # T_recon[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # Efficient way:
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        error = np.linalg.norm(T - T_recon)
        return error < tol, T_recon

    # Function to perform ALS for a given rank
    def try_rank_als(R, max_iters=500, tol=1e-8):
        # Initialize
        U = np.random.randn(dim_u, R)
        V = np.random.randn(dim_v, R)
        W = np.random.randn(dim_w, R)
        
        best_error = np.inf
        best_U, best_V, best_W = U, V, W
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Fix V, W, solve for U
            # T[i, :, :] = sum_r U[i, r] * (V[:, r] * W[:, r])^T
            # Let M_r = V[:, r] * W[:, r] (outer product, shape dim_v x dim_w)
            # T[i, :, :] = sum_r U[i, r] * M_r
            # This is a linear system for each i:
            # T[i, :, :].flatten() = U[i, :] @ M_r.flatten() (summed over r)
            # Let A_mat be dim_v*dim_w x R, where A_mat[:, r] = (V[:, r] * W[:, r]).flatten()
            # T_flat is dim_u x dim_v*dim_w
            # We want U @ A_mat.T = T_flat
            # U = T_flat @ A_mat @ (A_mat.T @ A_mat)^-1
            
            # Construct A_mat
            # V[:, r] * W[:, r] is element-wise product, then flatten
            # Shape (R, dim_v, dim_w) -> (R, dim_v*dim_w)
            VW = V[:, :, None] * W[:, None, :] # (R, dim_v, dim_w)
            A_mat = VW.reshape(R, -1).T # (dim_v*dim_w, R)
            
            T_flat = T.reshape(dim_u, -1) # (dim_u, dim_v*dim_w)
            
            # Solve U @ A_mat.T = T_flat
            # U = T_flat @ A_mat @ inv(A_mat.T @ A_mat)
            # Use lstsq for stability
            try:
                U_new, residuals, rank_u, s = np.linalg.lstsq(A_mat.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
                
            # Fix U, W, solve for V
            # T[:, j, :] = sum_r V[j, r] * (U[:, r] * W[:, r])^T
            # Let B_r = U[:, r] * W[:, r]
            # B_mat is (dim_u*dim_w, R)
            UW = U[:, :, None] * W[:, None, :] # (R, dim_u, dim_w)
            B_mat = UW.reshape(R, -1).T # (dim_u*dim_w, R)
            
            T_flat_v = T.reshape(dim_v, -1) # (dim_v, dim_u*dim_w) -- Wait, T is (dim_u, dim_v, dim_w)
            # We need T[:, j, :] for each j.
            # T is (dim_u, dim_v, dim_w).
            # Move axis 1 to 0: T_trans = T.transpose(1, 0, 2) -> (dim_v, dim_u, dim_w)
            T_trans = T.transpose(1, 0, 2)
            T_flat_v = T_trans.reshape(dim_v, -1)
            
            try:
                V_new, residuals, rank_v, s = np.linalg.lstsq(B_mat.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
                
            # Fix U, V, solve for W
            # T[:, :, k] = sum_r W[k, r] * (U[:, r] * V[:, r])^T
            # Let C_r = U[:, r] * V[:, r]
            # C_mat is (dim_u*dim_v, R)
            UV = U[:, :, None] * V[:, None, :] # (R, dim_u, dim_v)
            C_mat = UV.reshape(R, -1).T # (dim_u*dim_v, R)
            
            T_trans_w = T.transpose(2, 0, 1) # (dim_w, dim_u, dim_v)
            T_flat_w = T_trans_w.reshape(dim_w, -1)
            
            try:
                W_new, residuals, rank_w, s = np.linalg.lstsq(C_mat.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
                
            # Check error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            error = np.linalg.norm(T - T_recon)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < tol:
                return best_U, best_V, best_W, error
                
        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 16
    lower_bound = max(n*m, m*p, p*n)
    
    best_result = None
    best_rank = rank_schoolbook
    
    for R in range(lower_bound, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for each rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            res = try_rank_als(R, max_iters=100, tol=1e-7)
            if res:
                U, V, W, err = res
                # Verify strictly
                valid, _ = check_decomposition(U, V, W, tol=1e-5)
                if valid:
                    best_result = (U, V, W, R)
                    found = True
                    break
        if found:
            break
            
    if best_result:
        U, V, W, R = best_result
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U = np.zeros((dim_u, rank_schoolbook))
        V = np.zeros((dim_v, rank_schoolbook))
        W = np.zeros((dim_w, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
