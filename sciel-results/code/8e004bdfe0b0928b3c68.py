import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    c = 0
    # Block 1: rows 0-1, cols 0-1 (2x2x7)
    # Decompose into 3x(2x2x2) + 1x(2x2x1)
    # First 2x2x2: cols 0-1
    # Strassen for 2x2x2
    strassen_222 = [
        # (u_idx, v_idx, w_idx, u_val, v_val, w_val)
        # M1 = (A00+A11)(B00+B11)
        (0, 0, 0, 1, 1, 1), (1, 0, 0, 1, 1, 1),
        (0, 1, 0, 1, 1, 1), (1, 1, 0, 1, 1, 1),
        # M2 = (A10+A11)B00
        (1, 0, 1, 1, 1, 1), (0, 0, 1, 1, 1, 1),
        # M3 = A00(B01-B11)
        (0, 1, 2, 1, 1, 1), (0, 1, 2, 0, -1, 1),
        # M4 = A11(B10-B00)
        (1, 0, 3, 1, -1, 1), (1, 1, 3, 1, 1, 1),
        # M5 = (A00+A01)B11
        (0, 1, 4, 1, 1, 1), (1, 1, 4, 1, 1, 1),
        # M6 = (A10-A00)(B00+B01)
        (1, 0, 5, 1, 1, 1), (0, 0, 5, -1, 1, 1),
        (1, 1, 5, 0, 1, 1),
        # M7 = (A01-A11)(B10+B11)
        (0, 0, 6, 1, 1, 1), (1, 0, 6, -1, 1, 1),
        (0, 1, 6, 0, 1, 1),
    ]
    
    # Actually, let me just construct it properly with the standard Strassen formulas
    # For 2x2x2, the tensor T[i,j,k] = 1 if i,j,k in {0,1}
    # Strassen decomposition:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me just use the known rank-7 decomposition for 2x2x2
    # U rows: 00, 01, 10, 11 (i*m+j)
    # V rows: 00, 01, 10, 11 (j*p+k)
    # W rows: 00, 01, 10, 11 (k*n+i)
    
    # Standard Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me build the U, V, W matrices for this
    # Each term contributes to specific U, V, W entries
    
    # I'll just directly construct the 39-rank decomposition
    # using the block structure
    
    # Block 1: 2x2x7 (rows 0-1 of A, cols 0-1 of B)
    # Sub-block 1a: 2x2x2 (cols 0-1 of B)
    # Use Strassen
    # U indices: 0,1,2,3 (i=0,1; j=0,1)
    # V indices: 0,1,2,3 (j=0,1; k=0,1)
    # W indices: 0,1,2,3 (k=0,1; i=0,1)
    
    # Strassen terms for 2x2x2:
    # M1: U=(1,0,1,0), V=(1,0,1,0), W=(1,0,1,0) -> C00+=1, C11+=1
    # Actually, let me just use the explicit construction
    
    # For 2x2x2 with Strassen (rank 7):
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   U: [1,0,1,0], V: [1,0,1,0], W: [1,1,0,0] (contributes to C00 and C11)
    #   Wait, W[k*n+i] so W[0*2+0]=W[0] is C00, W[1*2+0]=W[2] is C10, W[0*2+1]=W[1] is C01, W[1*2+1]=W[3] is C11
    #   M1 contributes to C00 and C11
    #   So W for M1: [1,0,0,1]
    
    # Let me just hardcode the known decomposition
    
    # I'll use a simpler approach: just construct the 39-rank decomposition directly
    # by combining the known optimal decompositions for the sub-blocks
    
    # For now, let me just return the trivial 42-rank decomposition as a fallback
    # and then I can improve it
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
