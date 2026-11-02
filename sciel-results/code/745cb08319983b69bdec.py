"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Actually the tensor is: T[a,b,c] where a = i*m+j (input A), b = j'*p+k (input B), c = k'*n+i' (output)
    # T[a,b,c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} summed... 
    # The matrix multiplication tensor: T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # So T[a,b,c] is nonzero when a=(i,j), b=(j,k), c=(k,i) with matching indices.
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k; else 0.
    # Check: sum_c U[a,c]*V[b,c]*W[c_index,c] where c_index = k*n+i
    # Let's build the target tensor
    # For each (i, j, k): a = i*m+j, b = j*p+k, c = k*n+i, T[a,b,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(rank))
                if abs(val - 1.0) > tol:
                    return False
                # Check zeros: for other (a,b,c) combinations that should be 0
    # Check that all other entries are 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if this should be nonzero
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                should_be = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                val = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(rank))
                if abs(val - should_be) > tol:
                    return False
    return True

def _construct_rank24(n, m, p):
    """Construct a rank-24 decomposition for 2×3 × 3×8 by splitting into 8 independent 2×3×3×1."""
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for k in range(p):
        for j in range(m):
            U[0 * m + j][col] = 1  # A[0,j]
            U[1 * m + j][col] = 1  # A[1,j]
            V[j * p + k][col] = 1  # B[j,k]
            W[k * n + 0][col] = 1  # output C[0,k]
            W[k * n + 1][col] = 1  # output C[1,k]
            col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try rank 24 construction
    result = _construct_rank24(n, m, p)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
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
# EVOLVE-BLOCK-END
