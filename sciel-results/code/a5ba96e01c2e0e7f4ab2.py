"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook decomposition: rank = n*m*p = 63
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W, rank
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact using the trace convention W[k*n+i, r]"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        rank = U.shape[1]
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T_recon = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        T_target = np.ones((n, m, p), dtype=float)
        return np.max(np.abs(T_recon - T_target)) < tol
    
    def try_rank(r, n, m, p, max_iters=150, tol=1e-9, seed=42, deadline=None):
        """Try to find a rank-r decomposition using alternating least squares"""
        np.random.seed(seed)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize columns to prevent blowup
        for c in range(r):
            u_norm = np.linalg.norm(U[:, c])
            v_norm = np.linalg.norm(V[:, c])
            w_norm = np.linalg.norm(W[:, c])
            if u_norm > 1e-12 and v_norm > 1e-12 and w_norm > 1e-12:
                scale = (u_norm * v_norm * w_norm) ** (1.0 / 3.0)
                U[:, c] /= u_norm * scale
                V[:, c] /= v_norm * scale
                W[:, c] /= w_norm * scale
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute current tensor
            T_curr = np.zeros((n, m, p), dtype=float)
            for c in range(r):
                u_col = U[:, c].reshape(n, m)
                v_col = V[:, c].reshape(m, p)
                w_col = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_curr[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n, m, p), dtype=float)
            err = np.max(np.abs(T_curr - T_target))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
            
            # Alternating least squares: fix two factors, solve for the third
            # For U: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For fixed (i,j), we need: T[i,j,:] = U[i*m+j, :] @ (V[j, :, :] * W[:, i, :])
            # Actually, let's do gradient-based updates which are simpler for 3-way tensors
            
            # Gradient descent on the objective: sum (T_curr - 1)^2
            for c in range(r):
                u_col = U[:, c].reshape(n, m)
                v_col = V[:, c].reshape(m, p)
                w_col = W[:, c].reshape(p, n)
                
                grad_U = np.zeros((n, m), dtype=float)
                grad_V = np.zeros((m, p), dtype=float)
                grad_W = np.zeros((p, n), dtype=float)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual = T_curr[i, j, k] - 1.0
                            grad_U[i, j] += residual * v_col[j, k] * w_col[k, i]
                            grad_V[j, k] += residual * u_col[i, j] * w_col[k, i]
                            grad_W[k, i] += residual * u_col[i, j] * v_col[j, k]
                
                # Adaptive learning rate
                lr = 0.01
                U[:, c] -= lr * grad_U.flatten()
                V[:, c] -= lr * grad_V.flatten()
                W[:, c] -= lr * grad_W.flatten()
                
                # Renormalize
                u_norm = np.linalg.norm(U[:, c])
                v_norm = np.linalg.norm(V[:, c])
                w_norm = np.linalg.norm(W[:, c])
                if u_norm > 1e-12 and v_norm > 1e-12 and w_norm > 1e-12:
                    scale = (u_norm * v_norm * w_norm) ** (1.0 / 3.0)
                    U[:, c] /= u_norm * scale
                    V[:, c] /= v_norm * scale
                    W[:, c] /= w_norm * scale
            
            if deadline is not None and time.time() > deadline - 30:
                break
        
        if best_err < tol:
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            return U_list, V_list, W_list, r
        
        return None
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school, rank_school = schoolbook(n, m, p)
    best_result = (U_school, V_school, W_school, rank_school)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(21, 21, 9) = 21
    # Try ranks from 21 upward, with multiple seeds for each rank
    for r in range(21, rank_school):
        if time.time() > deadline - 60:
            break
        
        # Try multiple random seeds for each rank
        for seed in range(r, r + 5):
            if time.time() > deadline - 30:
                break
            
            result = try_rank(r, n, m, p, max_iters=100, tol=1e-9, seed=seed, deadline=deadline - 30)
            if result is not None:
                U_try, V_try, W_try, r_try = result
                if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-9):
                    best_result = (U_try, V_try, W_try, r_try)
                    break  # Found a valid lower rank, return it
        
        if best_result[3] < rank_school:
            break  # Found a valid lower rank
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
