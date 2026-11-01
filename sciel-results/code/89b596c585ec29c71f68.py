import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target: 2x3x9
    # Strategy: Decompose into two independent 1x3x9 multiplications
    # Each 1x3x9 multiplication has rank exactly 9 (one rank-1 term per output column)
    # Total rank = 2 * 9 = 18
    
    rank = 2 * p  # 18
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For row i of the 2x3 matrix, and column k of the 3x9 matrix:
    # Output[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # We use rank-1 terms indexed by (i, k) for i in 0..1, k in 0..8
    # Term (i, k) contributes to U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    # where col = i*p + k
    
    for i in range(n):
        for k in range(p):
            col = i * p + k
            # U: entry at row (i, j) for each j
            for j in range(m):
                U[i * m + j, col] = 1.0
            # V: entry at row (j, k) for each j
            for j in range(m):
                V[j * p + k, col] = 1.0
            # W: entry at row (k, i)
            W[k * n + i, col] = 1.0
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check nonzero entries
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                # Should be 1.0
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Nonzero entry ({i},{j},{k}) gives {val}")
    
    # Check zero entries
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in nonzero_set:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        raise ValueError(f"Zero entry ({a},{b},{c}) gives {val}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
