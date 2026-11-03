import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # Tensor T (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    
    # Create the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def try_rank(R, max_iters=1000, tol=1e-12, num_restarts=5):
        best_residual = np.inf
        best_factors = None
        
        for restart in range(num_restarts):
            # Random init
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Normalize to prevent scale issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iters):
                # Update U
                # T_U is (n*m) x (m*p * p*n)
                # T_U[i, (j,k)] = T[i, j, k]
                # T_U = U @ (W @ V.T).T ?
                # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Fix V, W.
                # For each i, T[i, :, :] = U[i, :] @ (V.T @ W.T)? No.
                # T[i, j, k] = sum_r U[i, r] (V[j, r] W[k, r])
                # Let M[i, (j,k)] = T[i, j, k]
                # M = U @ Z.T where Z[r, (j,k)] = V[j, r] W[k, r]
                # So U = M @ pinv(Z.T)
                
                Z = np.zeros((R, m*p * p*n))
                for r in range(R):
                    for j in range(m*p):
                        for k in range(p*n):
                            Z[r, j * p*n + k] = V[j, r] * W[k, r]
                
                M_U = np.zeros((n*m, m*p * p*n))
                for i in range(n*m):
                    M_U[i, :] = T[i, :, :].flatten()
                
                # Solve U M = M_U
                # U = M_U @ Z^+
                U_new, _, _, _ = np.linalg.lstsq(Z.T, M_U.T, rcond=None)
                U = U_new.T
                
                # Update V
                Z_V = np.zeros((R, n*m * p*n))
                for r in range(R):
                    for i in range(n*m):
                        for k in range(p*n):
                            Z_V[r, i * p*n + k] = U[i, r] * W[k, r]
                
                M_V = np.zeros((m*p, n*m * p*n))
                for j in range(m*p):
                    M_V[j, :] = T[:, j, :].flatten()
                    
                V_new, _, _, _ = np.linalg.lstsq(Z_V.T, M_V.T, rcond=None)
                V = V_new.T
                
                # Update W
                Z_W = np.zeros((R, n*m * m*p))
                for r in range(R):
                    for i in range(n*m):
                        for j in range(m*p):
                            Z_W[r, i * m*p + j] = U[i, r] * V[j, r]
                
                M_W = np.zeros((p*n, n*m * m*p))
                for k in range(p*n):
                    M_W[k, :] = T[:, :, k].flatten()
                    
                W_new, _, _, _ = np.linalg.lstsq(Z_W.T, M_W.T, rcond=None)
                W = W_new.T
                
                # Check residual
                T_approx = np.zeros_like(T)
                for r in range(R):
                    T_approx += np.einsum('ir,jr,kr->ijk', U[:, r], V[:, r], W[:, r])
                residual = np.linalg.norm(T - T_approx)
                if residual < tol:
                    return U, V, W
                if residual < best_residual:
                    best_residual = residual
                    best_factors = (U.copy(), V.copy(), W.copy())
                    
        if best_factors:
            U, V, W = best_factors
            # Try to snap to integers/half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify
            T_approx = np.zeros_like(T)
            for r in range(R):
                T_approx += np.einsum('ir,jr,kr->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
            
            if np.allclose(T_approx, T, atol=1e-9):
                return U_snap, V_snap, W_snap
                
        return None

    # Try ranks
    for R in [35, 34, 33, 32, 31, 30]:
        start_time = time.time()
        factors = try_rank(R, max_iters=200, num_restarts=10)
        if factors:
            U, V, W = factors
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        if time.time() - start_time > 100:
            break
            
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
