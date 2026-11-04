import time
import random
import itertools

DIMENSIONS = [2, 3, 9]

def _check_exact(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    N = n * m  # 6
    M = m * p  # 27
    P = p * n  # 18
    schoolbook_rank = n * m * p  # 54
    
    U = [[0] * schoolbook_rank for _ in range(N)]
    V = [[0] * schoolbook_rank for _ in range(M)]
    W = [[0] * schoolbook_rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
