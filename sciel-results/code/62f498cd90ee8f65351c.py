"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    best_rank = n * m * p  # 84
    best_decomp = None

    # Try to find lower-rank decomposition via random initialization + optimization
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # Try several random initializations with various ranks
    for rank in range(40, best_rank + 1):
        if time.time() > deadline - 20:
            break
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            # Random initialization
            np.random.seed(rank * 1000 + trial)
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1

            def objective(params):
                if time.time() > deadline - 10:
                    return 1e10
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # and compare to 1
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                            err += (val - 1.0) ** 2
                return err

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use Nelder-Mead for local optimization
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 200, 'fatol': 1e-8, 'xatol': 1e-8})
            except Exception:
                continue
            
            if result.fun < 1e-6:
                U = result.x[:n*m*rank].reshape(n*m, rank).tolist()
                V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank).tolist()
                W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank).tolist()
                best_rank = rank
                best_decomp = (U, V, W)
                break  # Found lower rank, stop trying this rank
        
        if best_decomp is not None and best_rank < rank:
            break  # Found lower rank

    if best_decomp is not None:
        return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
    
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
