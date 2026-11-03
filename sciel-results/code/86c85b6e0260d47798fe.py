"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # Try to construct a low-rank decomposition
    # For 3x3 * 3x8, let's try to use algebraic structure
    
    # Approach: Use the fact that we can decompose using fewer terms
    # by exploiting linearity and the specific dimensions
    
    # Let's try a systematic construction
    # The multiplication tensor has entries T[i,j,k] where:
    # i in [0,9) indexes (row, col) of A: i = row*m + col
    # j in [0,24) indexes (col, out_col) of B: j = col*p + out_col  
    # k in [0,24) indexes (out_col, row) of C: k = out_col*n + row
    
    # T[i,j,k] = 1 if col_A == col_B and row_C == row_A and out_col_C == out_col_B
    
    # Let's try rank 21 using GF(3) structure
    # Map indices to field elements
    
    # Actually, let me try a direct construction with small rank
    # Start with schoolbook and try to reduce
    
    # For now, let's try rank 27 (3*9) as a starting point,
    # then try to reduce
    
    # Better: let's use the known Strassen-like decomposition for 3x3x3
    # which has rank 7, and extend it
    
    # For 3x3 * 3x8, we can view it as 8 independent 3x3 multiplications
    # but sharing the same A, so we can batch them
    
    # Let me try a construction with rank 24
    
    rank = 24
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Try a structured decomposition
    # For each output column group, use a small set of products
    
    # Actually, let me try the simplest possible: just return schoolbook
    # but with a lower rank if possible
    
    # Let me try rank 18
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # This is getting complex. Let me just return a valid schoolbook
    # decomposition to ensure correctness, with the minimum rank I can verify.
    
    # Schoolbook rank = n*m*p = 72
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
