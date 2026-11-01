import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1) > 1e-9:
                    return False, (i, j, k, val)
    # Check that non-matching positions are 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if (a,b,c) corresponds to a valid (i,j,k)
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                valid = (j1 == j2) and (i1 == i2) and (k1 == k2)
                val = np.sum(U[a] * V[b] * W[c])
                if valid:
                    if abs(val - 1) > 1e-9:
                        return False, (a, b, c, val)
                else:
                    if abs(val) > 1e-9:
                        return False, (a, b, c, val)
    return True, None

def construct():
    n, m, p = DIMENSIONS
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
