import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    for rank in range(89, 40, -1):
        if time.time() > deadline:
            break
        
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 7919 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for it in range(300):
                if time.time() > deadline:
                    break
                
                # Alternating least squares
                for i in range(n):
                    for j in range(m):
                        M = np.stack([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                
                for j in range(m):
                    for k in range(p):
                        M = np.stack([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                
                for k in range(p):
                    for i in range(n):
                        M = np.stack([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                
                if it % 50 == 0 or it == 299:
                    max_err = max(
                        abs(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1)
                        for i in range(n) for j in range(m) for k in range(p)
                    )
                    if max_err < 1e-10:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return schoolbook()
