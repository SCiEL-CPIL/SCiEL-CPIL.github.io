import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    np.random.seed(42)
    
    def compute_error(U, V, W):
        """Compute sum of squared errors efficiently using einsum"""
        # Reshape to 3D: (n, m, R), (m, p, R), (p, n, R)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        err = np.sum((S - 1.0) ** 2)
        return err

    def try_snap(U, V, W):
        """Try to snap to half-integers and verify exactness"""
        # Try snapping to integers first, then half-integers
        # Actually, just rounding to 0.5 is the requirement "integer/half-integer"
        # np.round(x * 2) / 2 snaps to 0.5 grid.
        
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        snap_err = compute_error(U_snap, V_snap, W_snap)
        
        if snap_err < 1e-8:
            return U_snap.copy(), V_snap.copy(), W_snap.copy()
        return None

    best_rank = R_fb
    best_factors = (U_fb, V_fb, W_fb)
    
    # Try ranks from 62 down to 54
    ranks_to_try = [62, 61, 60, 59, 58, 57, 56, 55, 54]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        best_err_for_R = float('inf')
        best_U_V_W_for_R = None
        
        # More restarts for higher ranks
        n_restarts = 50 if R >= 60 else 30
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            # Use sparse/structured initialization closer to schoolbook
            # Start with random Gaussian, but scale down
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # ALS iterations
            max_iters = 100
            prev_err = float('inf')
            current_err = compute_error(U, V, W)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each (i,j), minimize sum_k (sum_r U_ij_r V_jk_r W_ki_r - 1)^2
                # This is a linear least squares problem for U_ij
                # Let A be the matrix where A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                # We want sum_r U_ij_r A[r, k] to approximate 1 for all k
                # So we solve A^T A U_ij = A^T 1
                
                for i in range(n):
                    for j in range(m):
                        # Construct matrix M of shape (R, p)
                        # M[r, k] = V[j*p+k, r] * W[k*n+i, r]
                        # Vectorized: 
                        V_j = V[j*p:(j+1)*p, :].T # shape (p, R)
                        W_i = W[i::n, :].T # shape (p, R) -- wait, W index is k*n+i
                        # W[k*n+i, r] for fixed i, varying k.
                        # W is (p*n, R). Indices k*n+i.
                        # For fixed i, k ranges 0..p-1.
                        # Indices: i, n+i, 2n+i, ...
                        W_i_vec = W[i::n, :].T # shape (p, R)
                        
                        M = V_j * W_i_vec # shape (p, R)
                        
                        try:
                            # Solve (M.T @ M) u = M.T @ 1
                            MMt = M.T @ M
                            MMt += 1e-8 * np.eye(R)
                            rhs = M.T @ np.ones(p)
                            U[i * m + j, :] = np.linalg.solve(MMt, rhs)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # M[r, i] = U[i*m+j, r] * W[k*n+i, r]
                        # U fixed j, varying i.
                        U_j_vec = U[j::m, :].T # shape (n, R) -- wait, U index is i*m+j
                        # For fixed j, i ranges 0..n-1.
                        # Indices: j, m+j, 2m+j...
                        U_j_vec = U[j::m, :].T # shape (n, R)
                        
                        W_k_vec = W[k*n:(k+1)*n, :].T # shape (n, R)
                        
                        M = U_j_vec * W_k_vec # shape (n, R)
                        
                        try:
                            MMt = M.T @ M
                            MMt += 1e-8 * np.eye(R)
                            rhs = M.T @ np.ones(n)
                            V[j * p + k, :] = np.linalg.solve(MMt, rhs)
                        except:
                            pass
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        # M[r, j] = U[i*m+j, r] * V[j*p+k, r]
                        U_i_vec = U[i*m:(i+1)*m, :].T # shape (m, R)
                        V_k_vec = V[k::p, :].T # shape (m, R) -- wait, V index is j*p+k
                        # For fixed k, j ranges 0..m-1.
                        # Indices: k, p+k, 2p+k...
                        V_k_vec = V[k::p, :].T # shape (m, R)
                        
                        M = U_i_vec * V_k_vec # shape (m, R)
                        
                        try:
                            MMt = M.T @ M
                            MMt += 1e-8 * np.eye(R)
                            rhs = M.T @ np.ones(m)
                            W[k * n + i, :] = np.linalg.solve(MMt, rhs)
                        except:
                            pass
                
                current_err = compute_error(U, V, W)
                
                if current_err < 1e-12:
                    break
                
                if current_err > prev_err * 1.01 and it > 10:
                    break
                prev_err = current_err
                
            # Polish with L-BFGS-B if we are close but not exact
            if current_err < 1.0: # Only polish if we made some progress
                def objective(x):
                    U_flat = x[:n*m*R].reshape(n*m, R)
                    V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
                    return compute_error(U_flat, V_flat, W_flat)
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                try:
                    res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-15})
                    if res.success or res.fun < current_err:
                        U = res.x[:n*m*R].reshape(n*m, R)
                        V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                        W = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                        current_err = res.fun
                except:
                    pass
            
            # Check for snap-able solution
            result = try_snap(U, V, W)
            if result is not None:
                U_snap, V_snap, W_snap = result
                snap_err = compute_error(U_snap, V_snap, W_snap)
                if snap_err < best_err_for_R:
                    best_err_for_R = snap_err
                    best_U_V_W_for_R = (U_snap, V_snap, W_snap)
                
                if best_err_for_R < 1e-8:
                    best_rank = R
                    best_factors = best_U_V_W_for_R
                    break

    U, V, W = best_factors
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
