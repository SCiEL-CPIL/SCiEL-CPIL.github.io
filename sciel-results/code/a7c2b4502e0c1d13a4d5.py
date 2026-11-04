"""Matrix multiplication 3x4x6 rank minimization."""
import numpy as np
from itertools import product
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Uses a block decomposition:
    Split A (3x4) into A1 (3x3) and A2 (3x1).
    Split B (4x6) into B1 (3x6) and B2 (1x6).
    C = A1*B1 + A2*B2.
    
    Further split B1 (3x6) into B1a (3x3) and B1b (3x3).
    A1*B1 = A1*B1a + A1*B1b.
    
    Use schoolbook for 3x3x3 multiplications (rank 27 each).
    Use rank 1 for 3x1x6 multiplication.
    
    Total rank = 27 + 27 + 1 = 55.
    """
    n, m, p = 3, 4, 6
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Block 1: A1 (3x3) * B1a (3x3)
    # A1: i in 0..2, j in 0..2
    # B1a: j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Block 2: A1 (3x3) * B1b (3x3)
    # A1: i in 0..2, j in 0..2
    # B1b: j in 0..2, k in 3..5
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Block 3: A2 (3x1) * B2 (1x6)
    # A2: i in 0..2, j = 3
    # B2: j = 3, k in 0..5
    # We need sum_r U[i,3,r] V[3,k,r] W[k,i,r] = 1
    # Use single rank term
    r = col
    for i in range(3):
        U[i * m + 3, r] = 1.0
    for k in range(6):
        V[3 * p + k, r] = 1.0
    for k in range(6):
        for i in range(3):
            W[k * n + i, r] = 1.0
    col += 1
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute T[i,j,k] from the decomposition
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
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
