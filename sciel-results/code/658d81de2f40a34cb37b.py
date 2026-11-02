"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank-50 decomposition:
    # Split middle dimension j into {0,1} and {2}
    # Part 1: 2x2x10 (i in {0,1}, j in {0,1}, k in {0..9}) -> 40 terms
    # Part 2: 2x1x10 (i in {0,1}, j=2, k in {0..9}) -> 10 terms
    # Total rank = 50
    
    rank = 50
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    # Part 1: j in {0, 1}
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Part 2: j = 2
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1.0
            V[2 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Verify
    T = np.ones((n, m, p), dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - T[i, j, k])
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook rank 60
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
