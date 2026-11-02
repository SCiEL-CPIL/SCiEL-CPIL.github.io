import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Schoolbook Decomposition (Rank 63)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Precompute the target tensor in flattened form
    # T_flat[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid (i,j,k)
    T_flat = np.zeros((n * m, m * p, p * n))
    T_target = np.ones((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[a, b, c] = 1

    def check_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the tensor."""
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        error = np.linalg.norm(T_recon - T_target)
        return error < 1e-6, error

    def round_and_verify(U, V, W):
        """Round to nearest half-integer and verify."""
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        valid, error = check_decomposition(U_int, V_int, W_int)
        if valid:
            return U_int, V_int, W_int
        return None

    def als_decompose(U, V, W, T_flat, max_iters=200, tol=1e-8):
        """
        ALS optimization for CP decomposition.
        T = sum_r U_r ⊗ V_r ⊗ W_r
        where T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        """
        n_m = U.shape[0]
        m_p = V.shape[0]
        p_n = W.shape[0]
        R = U.shape[1]
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            # Update U: for each a, solve for U[a, :]
            # T_flat[a, :, :] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is a linear system: U[a, :] @ (V[:,r] ⊗ W[:,r])^T = T_flat[a, :, :]
            # Let M_r = V[:,r] * W[:,r] (element-wise), then T_flat[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # Actually: T_flat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T_flat[a,:,:].ravel() = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # So we solve: X @ U[a, :]^T = T_flat[a,:,:].ravel()
            # where X[b,c,r] = V[b,r] * W[c,r]
            
            X_U = V[:, :, None] * W[None, :, :]  # shape (m_p, p_n, R)
            X_U = X_U.transpose(0, 2, 1).reshape(m_p * p_n, R)  # shape (m_p*p_n, R)
            # X_U[b*p_n+c, r] = V[b,r] * W[c,r]
            
            # Solve: X_U @ U[a,:]^T = T_flat[a,:,:].ravel()
            # Use least squares
            U_new = np.zeros((n_m, R))
            for a in range(n_m):
                rhs = T_flat[a, :, :].ravel()
                # Solve X_U @ x = rhs
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(X_U, rhs, rcond=None)
                    U_new[a, :] = sol
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Update V: for each b, solve for V[b, :]
            # T_flat[:, b, :] = sum_r V[b,r] * outer(U[:,r], W[:,r])
            # X_V[a,c,r] = U[a,r] * W[c,r]
            X_V = U[:, :, None] * W[None, :, :]  # shape (n_m, p_n, R)
            X_V = X_V.transpose(0, 2, 1).reshape(n_m * p_n, R)  # shape (n_m*p_n, R)
            
            V_new = np.zeros((m_p, R))
            for b in range(m_p):
                rhs = T_flat[:, b, :].ravel()
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(X_V, rhs, rcond=None)
                    V_new[b, :] = sol
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Update W: for each c, solve for W[c, :]
            # T_flat[:, :, c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
            # X_W[a,b,r] = U[a,r] * V[b,r]
            X_W = U[:, :, None] * V[None, :, :]  # shape (n_m, m_p, R)
            X_W = X_W.transpose(0, 2, 1).reshape(n_m * m_p, R)  # shape (n_m*m_p, R)
            
            W_new = np.zeros((p_n, R))
            for c in range(p_n):
                rhs = T_flat[:, :, c].ravel()
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(X_W, rhs, rcond=None)
                    W_new[c, :] = sol
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
            # Check convergence
            U_3d = U.reshape(n, m, -1)
            V_3d = V.reshape(m, p, -1)
            W_3d = W.reshape(p, n, -1)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
            err = np.linalg.norm(T_recon - T_target)
            if err < tol:
                break
        
        return U, V, W

    best_result = (U_school, V_school, W_school)
    best_rank = rank_school

    # Search for lower rank decompositions
    # Try ranks from 62 down to 40
    for R in range(62, 39, -1):
        if time.time() - start_time > time_limit - 15:
            break
        
        found = False
        # Try multiple random initializations
        for seed in range(20):
            if time.time() - start_time > time_limit - 15:
                break
            
            np.random.seed(seed + R * 1000)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            U_opt, V_opt, W_opt = als_decompose(U, V, W, T_flat, max_iters=150, tol=1e-7)
            
            # Check continuous error
            U_3d = U_opt.reshape(n, m, -1)
            V_3d = V_opt.reshape(m, p, -1)
            W_3d = W_opt.reshape(p, n, -1)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
            err_cont = np.linalg.norm(T_recon - T_target)
            
            # If continuous error is small, try rounding
            if err_cont < 1e-2:
                result = round_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    U_int, V_int, W_int = result
                    best_result = (U_int, V_int, W_int)
                    best_rank = R
                    found = True
                    break
        
        if found:
            continue

    U_out, V_out, W_out = best_result
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

# EVOLVE-BLOCK-END
