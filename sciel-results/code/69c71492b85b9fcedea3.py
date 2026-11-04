"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index triples
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def residual(params, rank):
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        
        res = np.empty(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            res[idx] = np.dot(U_flat[u_idx, :], V_flat[v_idx, :] * W_flat[w_idx, :]) - 1.0
        return res
    
    # Try ranks from low to high
    for rank in range(12, 25):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - 10 - time.time()
        max_attempts = max(1, int(remaining / 3))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 10000 + attempt * 7 + 42)
            scale = 1.0 / (rank ** 0.5)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=300)
                if result.cost < 1e-12:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                    
                    max_err = 0.0
                    for (i, j, k) in indices:
                        val = np.dot(U_sol[i * m + j, :], V_sol[j * p + k, :] * W_sol[k * n + i, :])
                        max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-8:
                        def to_list(arr):
                            lst = []
                            for row in arr:
                                row_lst = []
                                for val in row:
                                    rounded = round(val * 2) / 2
                                    if abs(val - rounded) < 1e-6:
                                        row_lst.append(rounded)
                                    else:
                                        row_lst.append(float(val))
                                lst.append(row_lst)
                            return lst
                        
                        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
            except Exception:
                continue
    
    # Fallback: naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
