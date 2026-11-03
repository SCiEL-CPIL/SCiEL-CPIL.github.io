import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Search for lower rank
    # We try ranks from 20 to 35
    for R in range(20, best_rank):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS
            for it in range(50):
                # Update U
                # T = sum U_r V_r^T W_r^T
                # For fixed V, W, U is solved by:
                # T_{ijk} = sum_r U_{ir} (V_{jr} W_{kr})
                # Let M_r = V_r W_r^T (m*p x p*n)
                # T_i = sum_r U_{ir} M_r
                # This is a linear system for U_i.
                
                # Reshape T to (n*m, m*p*p*n)
                T_flat = T.reshape(n * m, m * p * p * n)
                
                # Build A matrix for U: (R, m*p*p*n)
                A_U = np.zeros((R, m * p * p * n))
                for r in range(R):
                    M_r = np.outer(V[:, r], W[:, r]) # (m*p, p*n)
                    A_U[r] = M_r.flatten()
                    
                # Solve U = (A_U^T A_U)^-1 A_U^T T_flat^T
                # Use least squares
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_U.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                T_flat_V = T.reshape(m * p, n * m * p * n)
                A_V = np.zeros((R, n * m * p * n))
                for r in range(R):
                    M_r = np.outer(U[:, r], W[:, r]) # (n*m, p*n)
                    A_V[r] = M_r.flatten()
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_V.T, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                T_flat_W = T.reshape(p * n, n * m * m * p)
                A_W = np.zeros((R, n * m * m * p))
                for r in range(R):
                    M_r = np.outer(U[:, r], V[:, r]) # (n*m, m*p)
                    A_W[r] = M_r.flatten()
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_W.T, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
            # Round to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            # Verify
            T_approx = np.zeros((n * m, m * p, p * n), dtype=int)
            for r in range(R):
                T_approx += np.outer(np.outer(U_int[:, r], V_int[:, r]), W_int[:, r])
                
            if np.array_equal(T_approx, T.astype(int)):
                best_U = U_int.tolist()
                best_V = V_int.tolist()
                best_W = W_int.tolist()
                best_rank = R
                return {"U": best_U, "V": best_V, "W": best_W}
                
    return {"U": best_U, "V": best_V, "W": best_W}
