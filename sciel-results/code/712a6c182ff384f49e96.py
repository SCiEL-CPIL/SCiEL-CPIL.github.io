"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    # Target rank: 3 * 14 + 8 = 50
    rank = 50
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0] * rank for _ in range(p * n)]  # 14 rows
    
    # Strassen factors for 2x2x2
    # Indices: i (0,1), j (0,1), k (0,1)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # M1..M7
    strassen_U = [
        [1, 0, 0, 0, 0, 0, 0], # A00
        [0, 0, 1, 1, 0, 0, 0], # A10, A11
        [1, 1, 0, 0, 0, 0, 0], # A00, A10
        [0, 0, 0, 1, 0, 0, 0], # A11
        [0, 0, 0, 1, 1, 0, 0], # A11, A01 -> Wait, standard Strassen uses specific combos
        # Let's use the standard 7 products:
        # M1 = A00 * (B01 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = (A00 + A10) * B10
        # M4 = A11 * (B00 - B10)
        # M5 = (A01 + A11) * B11
        # M6 = (A00 - A10) * (B00 + B10)
        # M7 = (A01 - A11) * (B01 + B11)
        
        # Corrected U rows for i=0 (A00, A01), i=1 (A10, A11)
        # Row 0: A00
        # Row 1: A01
        # Row 2: A10
        # Row 3: A11
    ]
    
    # Let's define the coefficients explicitly for clarity and correctness
    # U indices: i*2 + j
    # V indices: j*2 + k
    # W indices: k*2 + i
    
    # M1: A00 * (B01 + B11)
    # M2: (A10 + A11) * B00
    # M3: (A00 + A10) * B10
    # M4: A11 * (B00 - B10)
    # M5: (A01 + A11) * B11
    # M6: (A00 - A10) * (B00 + B10)
    # M7: (A01 - A11) * (B01 + B11)
    
    # U columns (for A indices 0,1,2,3 corresponding to A00,A01,A10,A11)
    U_s = [
        [1, 0, 1, 0, 0, 1, 0], # A00 (rows 0,1? No, A00 is idx 0)
        [0, 1, 0, 0, 1, 0, 1], # A01 (idx 1)
        [0, 0, 1, 1, 0, -1, 0], # A10 (idx 2)
        [0, 0, 1, 1, 1, 0, -1]  # A11 (idx 3)
    ]
    
    # V columns (for B indices 0,1,2,3 corresponding to B00,B01,B10,B11)
    V_s = [
        [0, 0, 1, 1, 0, 1, 0], # B00 (idx 0)
        [1, 0, 0, 0, 0, 0, 1], # B01 (idx 1)
        [0, 0, 0, -1, 0, 1, 0], # B10 (idx 2)
        [1, 1, 0, 0, 1, 0, 0]  # B11 (idx 3)
    ]
    
    # W columns (for C indices 0,1,2,3 corresponding to C00,C01,C10,C11)
    # C00 = M4 + M5 + M6 - M7
    # C01 = M1 + M7
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M3
    W_s = [
        [0, 0, 0, 1, 1, 1, -1], # C00 (idx 0)
        [1, 0, 0, 0, 0, 0, 1],  # C01 (idx 1)
        [0, 1, 1, 0, 0, 0, 0],  # C10 (idx 2)
        [1, -1, 1, 1, 0, 0, 0]  # C11 (idx 3)
    ]
    
    col_offset = 0
    
    # Process 3 blocks of k pairs: (0,1), (2,3), (4,5)
    for block in range(3):
        k_start = block * 2
        k_end = k_start + 2
        
        # Two Strassen decompositions: one for j in {0,1}, one for j in {2,3}
        for j_block in range(2):
            j_start = j_block * 2
            
            for c in range(7):
                col = col_offset + c
                
                # Update U
                for i in range(2):
                    for j in range(2):
                        idx_A = i * 2 + j
                        idx_U_row = i * m + (j_start + j)
                        U[idx_U_row][col] = U_s[idx_A][c]
                        
                # Update V
                for j in range(2):
                    for k in range(2):
                        idx_B = j * 2 + k
                        idx_V_row = (j_start + j) * p + (k_start + k)
                        V[idx_V_row][col] = V_s[idx_B][c]
                        
                # Update W
                for k in range(2):
                    for i in range(2):
                        idx_C = k * 2 + i
                        idx_W_row = (k_start + k) * n + i
                        W[idx_W_row][col] = W_s[idx_C][c]
                        
            col_offset += 7

    # Process last k=6 with Schoolbook (Rank 8)
    for i in range(2):
        for j in range(4):
            col = col_offset
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
