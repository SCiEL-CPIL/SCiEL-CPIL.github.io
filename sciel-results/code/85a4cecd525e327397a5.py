"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k2][c] * W[k2*n+i3][c]
                                if abs(s) > tol:
                                    return False
        return True
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Strategy: Try to find a rank-10 decomposition using the block structure.
    # For 2x5 @ 5x5, we can view this as two independent 1x5 @ 5x5 problems.
    # Each 1x5 @ 5x5 has rank 5 (vector-matrix multiplication).
    # So total rank should be 10 if we can combine them properly.
    
    # Rank 10 construction:
    # c = i0*5 + j0 where i0 in {0,1}, j0 in 0..4
    # U[i*m+j, c] = delta(i, i0) * delta(j, j0)
    # V[j*p+k, c] = delta(j, j0)
    # W[k*n+i, c] = delta(i, i0)
    
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i0 in range(n):
        for j0 in range(m):
            c = i0 * m + j0
            # U[i*m+j, c] = delta(i, i0) * delta(j, j0)
            for i in range(n):
                for j in range(m):
                    if i == i0 and j == j0:
                        U[i * m + j][c] = 1
            
            # V[j*p+k, c] = delta(j, j0)
            for j in range(m):
                for k in range(p):
                    if j == j0:
                        V[j * p + k][c] = 1
            
            # W[k*n+i, c] = delta(i, i0)
            for k in range(p):
                for i in range(n):
                    if i == i0:
                        W[k * n + i][c] = 1
    
    # Verify the rank-10 decomposition
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If rank 10 doesn't work, try rank 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            for i in range(n):
                U[i * m + j0][c] = 1
                W[k0 * n + i][c] = 1
            V[j0 * p + k0][c] = 1
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
