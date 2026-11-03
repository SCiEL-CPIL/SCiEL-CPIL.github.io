import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n  # 3
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # U[i*m + j, r] = 1 if i == r else 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1.0
            
    # V[j*p + k, r] = 1 for all j, k, r
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                V[j * p + k][r] = 1.0
                
    # W[k*n + i, r] = 1 if i == r else 0
    for k in range(p):
        for i in range(n):
            W[k * n + i][i] = 1.0
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
