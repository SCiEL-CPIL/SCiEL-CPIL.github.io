import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank
    # Lower bound is n*p = 14
    for r in range(n * p, rank_sb):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Reshape for easy indexing
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            W_3d = W.reshape(p, n, r)
            
            def residuals(x):
                U_flat = x[:n*m*r]
                V_flat = x[n*m*r : n*m*r + m*p*r]
                W_flat = x[n*m*r + m*p*r :]
                
                U_r = U_flat.reshape(n, m, r)
                V_r = V_flat.reshape(m, p, r)
                W_r = W_flat.reshape(p, n, r)
                
                # Compute tensor product
                # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # We want T[i,j,k] = 1 for all i,j,k
                
                # Efficient computation:
                # For each c, U_c (n,m) outer V_c (m,p) outer W_c (p,n)
                # This is expensive.
                # Alternative:
                # C = sum_c (U_c * W_c^T) ? No.
                # Let's use einsum.
                # T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                # This creates a (n,m,p) tensor.
                # n=2, m=4, p=7 -> 56 elements.
                # r is small.
                # This is fast.
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                # Target is 1 everywhere
                return (T - 1).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if result.cost < 1e-8:
                    # Check for integer/half-integer rounding
                    U_sol = result.x[:n*m*r].reshape(n, m, r)
                    V_sol = result.x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
                    W_sol = result.x[n*m*r + m*p*r :].reshape(p, n, r)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify rounded solution
                    T_round = np.einsum('ijc,jkc,kic->ijk', U_round, V_round, W_round)
                    if np.allclose(T_round, 1, atol=1e-9):
                        # Valid solution found
                        best_rank = r
                        best_U = U_round.flatten().reshape(n*m, r)
                        best_V = V_round.flatten().reshape(m*p, r)
                        best_W = W_round.flatten().reshape(p*n, r)
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    # Convert to list of lists if required, or keep as numpy arrays?
    # The interface says "factor matrices of shapes ... with integer/half-integer entries"
    # The seed program returns lists of lists.
    # "Keep the function names and signatures ... exactly as in the current program"
    # The current program returns {"U": U, "V": V, "W": W} where U,V,W are lists of lists.
    
    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
