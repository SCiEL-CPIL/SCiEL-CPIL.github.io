import time
import random
import numpy as np

def construct():
    """
    Rank-minimal tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses 3 Strassen blocks (rank 7 each) + schoolbook for last column (rank 4) = rank 25.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen decomposition for 2x2 * 2x2
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices: 0:B00, 1:B01, 2:B10, 3:B11
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    # Strassen intermediate products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U coefficients for each Strassen term (row = A index, col = term)
    strassen_U = np.array([
        [1, 0, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 0, 1, 0, -1],  # A01
        [0, 1, 0, 0, 0, -1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ], dtype=int)
    
    # V coefficients for each Strassen term (row = B index, col = term)
    strassen_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [1, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, -1, 1, 0, 0, 1],  # B10
        [1, 0, 0, 0, 1, 0, 1]    # B11
    ], dtype=int)
    
    # W coefficients: W[k*n+i] = C[i,k] where i=row in C, k=col in C
    # Term 1 (M1): C00 += 1, C10 += -1, C01 += 1, C11 += 1
    # Term 2 (M2): C10 += 1
    # Term 3 (M3): C01 += 1
    # Term 4 (M4): C11 += 1
    # Term 5 (M5): C01 += 1
    # Term 6 (M6): C10 += 1
    # Term 7 (M7): C11 += 1
    
    # W has shape (p*n, rank) = (14, rank)
    # Global index = k*n + i = 2*k + i
    strassen_W_local = np.zeros((4, 7), dtype=int)  # [c_index, term_index]
    strassen_W_local[0, 0] = 1   # C00 from M1
    strassen_W_local[2, 0] = -1  # C10 from M1
    strassen_W_local[1, 0] = 1   # C01 from M1
    strassen_W_local[3, 0] = 1   # C11 from M1
    strassen_W_local[2, 1] = 1   # C10 from M2
    strassen_W_local[1, 2] = 1   # C01 from M3
    strassen_W_local[3, 3] = 1   # C11 from M4
    strassen_W_local[1, 4] = 1   # C01 from M5
    strassen_W_local[2, 5] = 1   # C10 from M6
    strassen_W_local[3, 6] = 1   # C11 from M7
    
    # Process 3 Strassen blocks for columns 0-5 of B/C
    for b in range(3):
        base_col = 2 * b  # starting column index for this block
        for r in range(7):
            col = b * 7 + r
            
            # Update U
            for i in range(4):
                U[i, col] += strassen_U[i, r]
            
            # Update V
            for j in range(4):
                coeff = strassen_V[j, r]
                if coeff != 0:
                    v_idx = j * p + (base_col + j % 2)  # This is wrong, let me fix
                    # Actually: B indices in block: 0:B00, 1:B01, 2:B10, 3:B11
                    # Global V index = j_global * p + k_global
                    # j_global is row in B (0 or 1), k_global is column in B (0-6)
                    # For block b, columns are 2b and 2b+1
                    # Local index 0 -> (0, 2b), 1 -> (0, 2b+1), 2 -> (1, 2b), 3 -> (1, 2b+1)
                    pass
            
            # Let me redo this more carefully
            # Local B index to global V index mapping for block b:
            # loc 0: j=0, k=2b   -> V[0*p + 2b]
            # loc 1: j=0, k=2b+1 -> V[0*p + 2b+1]
            # loc 2: j=1, k=2b   -> V[1*p + 2b]
            # loc 3: j=1, k=2b+1 -> V[1*p + 2b+1]
            
            # strassen_V has shape (4, 7) where row is local B index
            for loc_j in range(4):
                coeff = strassen_V[loc_j, r]
                if coeff != 0:
                    j_global = loc_j // 2
                    k_offset = loc_j % 2
                    v_idx = j_global * p + (2 * b + k_offset)
                    V[v_idx, col] += coeff
            
            # Update W
            # Local C index to global W index mapping for block b:
            # loc 0: i=0, k=2b   -> W[(2b)*n + 0]
            # loc 1: i=0, k=2b+1 -> W[(2b+1)*n + 0]
            # loc 2: i=1, k=2b   -> W[(2b)*n + 1]
            # loc 3: i=1, k=2b+1 -> W[(2b+1)*n + 1]
            for loc_c in range(4):
                coeff = strassen_W_local[loc_c, r]
                if coeff != 0:
                    i_global = loc_c % 2
                    k_offset = loc_c // 2
                    w_idx = (2 * b + k_offset) * n + i_global
                    W[w_idx, col] += coeff
    
    # Schoolbook for last column (col 6 of B, C)
    # A is 2x2, B is 2x1, C is 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 4 terms
    for r in range(4):
        col = 21 + r  # 3*7 = 21, so columns 21,22,23,24
        
        if r == 0:
            # A00 * B0 -> C0
            U[0, col] = 1
            V[0 * p + 6, col] = 1
            W[6 * n + 0, col] = 1
        elif r == 1:
            # A01 * B1 -> C0
            U[1, col] = 1
            V[1 * p + 6, col] = 1
            W[6 * n + 0, col] = 1
        elif r == 2:
            # A10 * B0 -> C1
            U[2, col] = 1
            V[0 * p + 6, col] = 1
            W[6 * n + 1, col] = 1
        elif r == 3:
            # A11 * B1 -> C1
            U[3, col] = 1
            V[1 * p + 6, col] = 1
            W[6 * n + 1, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
