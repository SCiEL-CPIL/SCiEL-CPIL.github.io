import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-32 tensor decomposition for 2x2 * 2x9 matrix multiplication.
    Uses Strassen's algorithm for 2x2 blocks and schoolbook for the remainder.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    # U: (n*m, rank) = (4, 32)
    # V: (m*p, rank) = (18, 32)
    # W: (p*n, rank) = (18, 32)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(k1, k2, start_col):
        """Add 7 Strassen products for columns k1, k2 starting at column start_col."""
        # U indices: (i, j) -> i*m + j
        u00, u01, u10, u11 = 0, 1, 2, 3
        
        # V indices: (j, k) -> j*p + k
        v00 = 0 * p + k1
        v01 = 0 * p + k2
        v10 = 1 * p + k1
        v11 = 1 * p + k2
        
        # W indices: (k, i) -> k*n + i
        w00 = k1 * n + 0
        w01 = k1 * n + 1
        w02 = k2 * n + 0
        w03 = k2 * n + 1
        
        # M1 = (a00 + a11) * (b00 + b11)
        c = start_col + 0
        U[u00][c] += 1; U[u11][c] += 1
        V[v00][c] += 1; V[v11][c] += 1
        W[w00][c] += 1; W[w03][c] += 1
        
        # M2 = (a10 + a11) * b00
        c = start_col + 1
        U[u10][c] += 1; U[u11][c] += 1
        V[v00][c] += 1
        W[w01][c] += 1; W[w03][c] -= 1
        
        # M3 = a00 * (b01 - b11)
        c = start_col + 2
        U[u00][c] += 1
        V[v01][c] += 1; V[v11][c] -= 1
        W[w02][c] += 1; W[w03][c] += 1
        
        # M4 = a11 * (b10 - b00)
        c = start_col + 3
        U[u11][c] += 1
        V[v10][c] += 1; V[v00][c] -= 1
        W[w00][c] += 1; W[w01][c] += 1
        
        # M5 = (a00 + a01) * b11
        c = start_col + 4
        U[u00][c] += 1; U[u01][c] += 1
        V[v11][c] += 1
        W[w00][c] -= 1; W[w02][c] += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        c = start_col + 5
        U[u10][c] += 1; U[u00][c] -= 1
        V[v00][c] += 1; V[v01][c] += 1
        W[w03][c] += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        c = start_col + 6
        U[u01][c] += 1; U[u11][c] -= 1
        V[v10][c] += 1; V[v11][c] += 1
        W[w00][c] += 1
        
        return start_col + 7

    def add_schoolbook_block(k, start_col):
        """Add 4 schoolbook products for a single column k starting at column start_col."""
        u00, u01, u10, u11 = 0, 1, 2, 3
        v0 = 0 * p + k
        v1 = 1 * p + k
        w0 = k * n + 0
        w1 = k * n + 1
        
        # N1 = a00 * b0k
        c = start_col + 0
        U[u00][c] += 1; V[v0][c] += 1; W[w0][c] += 1
        
        # N2 = a01 * b1k
        c = start_col + 1
        U[u01][c] += 1; V[v1][c] += 1; W[w0][c] += 1
        
        # N3 = a10 * b0k
        c = start_col + 2
        U[u10][c] += 1; V[v0][c] += 1; W[w1][c] += 1
        
        # N4 = a11 * b1k
        c = start_col + 3
        U[u11][c] += 1; V[v1][c] += 1; W[w1][c] += 1
        
        return start_col + 4

    col = 0
    # Use Strassen for pairs (0,1), (2,3), (4,5), (6,7)
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = add_strassen_block(k1, k2, col)
    # Use Schoolbook for column 8
    col = add_schoolbook_block(8, col)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
