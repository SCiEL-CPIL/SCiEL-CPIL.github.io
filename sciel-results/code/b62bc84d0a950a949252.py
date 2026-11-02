import numpy as np

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix.
    
    Strategy:
    Block 1: j=2. M_{2,1,5}, rank 10.
    Block 2: j in {0,1}. M_{2,2,5} split into M_{2,2,2} (rank 7) + M_{2,2,2} (rank 7) + M_{2,2,1} (rank 4) = 18.
    Total: 28.
    """
    n, m, p = 2, 3, 5
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper to add a Strassen block for M_{2,2,2}
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00)
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        col += 1

    # Block 1: j=2 (A[:, 2] * B[2, :])
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2a: k in {0, 1}. M_{2,2,2}.
    add_strassen(0, 0, 0)
    
    # Block 2b: k in {2, 3}. M_{2,2,2}.
    add_strassen(0, 0, 2)
    
    # Block 2c: k in {4}. M_{2,2,1}.
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return {"U": U, "V": V, "W": W}
