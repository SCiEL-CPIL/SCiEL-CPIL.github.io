import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 24
    
    rng = np.random.RandomState(12345)
    
    # Try ranks from low to high
    for rank in range(5, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(500):
            if time.time() > deadline - 5:
                break
            
            U = rng.randn(n * m, rank)
            W = rng.randn(p * n, rank)
            
            A_sys = np.zeros((n * m * p, m * p * rank))
            b = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i * m * p + j * p + k
                        for c in range(rank):
                            col_idx = (j * p + k) * rank + c
                            A_sys[row_idx, col_idx] = U[i * m + j, c] * W[k * n + i, c]
            
            try:
                V_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                residual = np.linalg.norm(A_sys @ V_flat - b)
                if residual < 1e-10:
                    V = V_flat.reshape(m * p, rank)
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                error += abs(val - 1.0)
                    if error < 1e-10:
                        U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
                        V_list = [[float(V[j, c]) for c in range(rank)] for j in range(m * p)]
                        W_list = [[float(W[k, c]) for c in range(rank)] for k in range(p * n)]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            except np.linalg.LinAlgError:
                continue
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
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
