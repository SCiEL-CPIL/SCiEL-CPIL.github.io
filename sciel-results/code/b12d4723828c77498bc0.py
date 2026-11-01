import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 2, 2, 9
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True

    # Rank 32 decomposition:
    # 4 Strassen blocks (2x2x2) for cols 0-7 (rank 7 each) -> 28
    # 1 Schoolbook block (2x2x1) for col 8 (rank 4) -> 4
    # Total 32.
    
    # Try to improve with 3 blocks of 2x2x3 (rank 10 each) -> 30.
    # I will implement a rank 10 decomposition for 2x2x3 if possible.
    # Otherwise, fallback to 32.
    
    # For now, return the known rank 32 decomposition.
    # Strassen U, V, W for 2x2x2
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    strassen_W = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ]
    
    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        for c in range(7):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = strassen_U[i * m + j][c]
            for j in range(m):
                V[j * p + k0][col] = strassen_V[j * 2 + 0][c]
                V[j * p + k1][col] = strassen_V[j * 2 + 1][c]
            for i in range(n):
                W[k0 * n + i][col] = strassen_W[0 * 2 + i][c]
                W[k1 * n + i][col] = strassen_W[1 * 2 + i][c]
            col += 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    if verify(result["U"], result["V"], result["W"]):
        return result
    
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
