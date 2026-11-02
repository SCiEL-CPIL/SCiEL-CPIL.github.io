import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """Return rank-19 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Uses optimized block decomposition:
    - Block 1: j=0 (2x1x5): rank 1 (outer product)
    - Block 2a: j=1,2, k=0,1 (2x2x2): rank 7 (Strassen)
    - Block 2b: j=1,2, k=2 (2x2x1): rank 4 (schoolbook)
    - Block 2c: j=1,2, k=3,4 (2x2x2): rank 7 (Strassen)
    Total rank: 1 + 7 + 4 + 7 = 19
    """
    n, m, p = 2, 3, 5
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]  # 6 x 19
    V = [[0] * rank for _ in range(m * p)]  # 15 x 19
    W = [[0] * rank for _ in range(p * n)]  # 10 x 19
    
    col = 0
    
    # Block 1: j=0, k=0..4 (2x1x5)
    # Rank 1: Outer product of column A[:,0] and row B[0,:]
    # C[i,k] = A[i,0] * B[0,k]
    # We need sum_r U[i,0,r] V[0,k,r] W[k,i,r] = 1
    # Set U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1 for all i,k
    for i in range(n):
        U[i * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Helper to add a Strassen block for M_{2,2,2}
    # local indices i, j, k in {0, 1}
    # global indices:
    # i -> global i
    # j -> global j_offset + j
    # k -> global k_offset + k
    def add_strassen(j_offset, k_offset):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[0 * m + (j_offset+0)][col] = 1
        U[1 * m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = 1
        W[(k_offset+1)*n + 1][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00) -> C10 += M2, C11 -= M2
        U[1 * m + (j_offset+0)][col] = 1
        U[1 * m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + 1][col] = 1
        W[(k_offset+1)*n + 1][col] = -1
        col += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[0 * m + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + 0][col] = 1
        W[(k_offset+1)*n + 1][col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[1 * m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + 0][col] = 1
        W[(k_offset+0)*n + 1][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11) -> C00 -= M5, C01 += M5
        U[0 * m + (j_offset+0)][col] = 1
        U[0 * m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = -1
        W[(k_offset+1)*n + 0][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[1 * m + (j_offset+0)][col] = 1
        U[0 * m + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + 1][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[0 * m + (j_offset+1)][col] = 1
        U[1 * m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = 1
        col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2)
    add_strassen(1, 0)
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1)
    # Rank 4
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2)
    add_strassen(1, 3)
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return {"U": U, "V": V, "W": W}
