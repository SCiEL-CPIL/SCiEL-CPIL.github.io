"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook decomposition
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
    
    # Build the multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # We want to find U, V, W such that:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (using trace convention W[k*n+i])
    # Actually the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: M[i,j,k] = delta_{i,j,k} meaning the coefficient of A[i,j]*B[j,k] in C[i,k]
    # Decomposition: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    def build_tensor(n, m, p):
        """Build the multiplication tensor as a 3D array T[i,j,k]"""
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute reconstructed tensor
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        rank = U.shape[1]
        T_recon = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            # Outer product of U[:,c], V[:,c], W[:,c]
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        T_target = np.ones((n, m, p), dtype=float)
        return np.max(np.abs(T_recon - T_target)) < tol
    
    def try_rank(r, n, m, p, max_iters=200, tol=1e-9, seed=42):
        """Try to find a rank-r decomposition using iterative optimization"""
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
            
            # Gradient descent on the objective: sum (T_curr - 1)^2
            # Update each factor
            for c in range(r):
                u_col = U[:, c].reshape(n, m)
                v_col = V[:, c].reshape(m, p)
                w_col = W[:, c].reshape(p, n)
                
                # Gradient w.r.t. U[i*m+j, c]
                # d/dU[i*m+j,c] sum_{i',j',k'} (sum_c' U[i',j',c'] V[j',k',c'] W[k',i',c'] - 1)^2
                # = 2 * (T_curr[i,j,k] - 1) * V[j,k,c] * W[k,i,c]  (for each k)
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
                
                # Learning rate
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
            
            if time.time() > deadline - 30:
                break
        
        if best_err < tol:
            # Convert to lists with integer/half-integer entries if possible
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
    # Try ranks from 21 upward
    for r in range(21, rank_school):
        if time.time() > deadline - 60:
            break
        
        result = try_rank(r, n, m, p, max_iters=100, tol=1e-9, seed=r)
        if result is not None:
            U_try, V_try, W_try, r_try = result
            if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-9):
                best_result = (U_try, V_try, W_try, r_try)
                break  # Found a valid lower rank, return it
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
