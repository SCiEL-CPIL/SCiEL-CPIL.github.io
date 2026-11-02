import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    
    # Helper to add a 1x4x2 block (Rank 8)
    def add_1x4x2(row_a, col_b_start, col_c_start):
        nonlocal col_idx
        # A: row row_a, cols 0..3
        # B: rows 0..3, cols col_b_start..col_b_start+1
        # C: row row_a, cols col_c_start..col_c_start+1
        for k in range(4):
            for c in range(2):
                u_idx = row_a * m + k
                v_idx = k * p + (col_b_start + c)
                w_idx = (col_c_start + c) * n + row_a
                
                U[u_idx, col_idx] = 1
                V[v_idx, col_idx] = 1
                W[w_idx, col_idx] = 1
                col_idx += 1

    # Helper to add a 2x2x2 Strassen block (Rank 7)
    def add_strassen_2x2x2(row_a_start, col_a_start, row_b_start, col_b_start, row_c_start, col_c_start):
        nonlocal col_idx
        # A: rows row_a_start..row_a_start+1, cols col_a_start..col_a_start+1
        # B: rows row_b_start..row_b_start+1, cols col_b_start..col_b_start+1
        # C: rows row_c_start..row_c_start+1, cols col_c_start..col_c_start+1
        
        # Strassen 2x2x2 terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We define the 7 rank-1 tensors for Strassen
        # Each term is a combination of U, V, W entries
        
        # Local indices for 2x2 block:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # U terms (A)
        # V terms (B)
        # W terms (C)
        
        # We will fill 7 columns
        
        # Term 1: M1
        # A: A11+A22 -> U[0]+=1, U[3]+=1
        # B: B11+B22 -> V[0]+=1, V[3]+=1
        # C: C11+=M1, C12-=M1, C21+=M1, C22-=M1 ? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's just use the explicit 7 terms for U, V, W
        # Standard Strassen 2x2x2 decomposition:
        
        # 1. U: [1, 0, 0, 1], V: [1, 0, 0, 1], W: [1, 0, 0, -1] (M1)
        #    Wait, W corresponds to C.
        #    M1 contributes to C11 (+), C22 (-).
        #    So W: [1, 0, 0, -1] ?
        #    C11 is index 0, C12 is 1, C21 is 2, C22 is 3.
        #    M1: C11+=1, C22-=1. W: [1, 0, 0, -1].
        
        # 2. U: [0, 0, 1, 1], V: [1, 0, 0, 0], W: [0, 0, 1, 0] (M2)
        #    M2: C21+=1. W: [0, 0, 1, 0].
        
        # 3. U: [1, 0, 0, 0], V: [0, 1, 0, -1], W: [0, 1, 0, 1] (M3)
        #    M3: C12+=1, C22+=1. W: [0, 1, 0, 1].
        
        # 4. U: [0, 0, 0, 1], V: [0, 0, 1, -1], W: [1, 0, 1, 0] (M4)
        #    M4: C11+=1, C21+=1. W: [1, 0, 1, 0].
        
        # 5. U: [1, 1, 0, 0], V: [0, 0, 0, 1], W: [-1, 1, 0, 0] (M5)
        #    M5: C11-=1, C12+=1. W: [-1, 1, 0, 0].
        
        # 6. U: [0, 0, 1, -1], V: [1, 1, 0, 0], W: [0, 0, 0, 1] (M6)
        #    M6: C22+=1. W: [0, 0, 0, 1].
        
        # 7. U: [0, 1, 0, -1], V: [0, 0, 1, 1], W: [1, 0, 0, 0] (M7)
        #    M7: C11+=1. W: [1, 0, 0, 0].
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_vals, v_vals, w_vals in terms:
            for l in range(4):
                if u_vals[l] != 0:
                    u_idx = (row_a_start + l//2) * m + (col_a_start + l%2)
                    U[u_idx, col_idx] += u_vals[l]
                if v_vals[l] != 0:
                    v_idx = (row_b_start + l//2) * p + (col_b_start + l%2)
                    V[v_idx, col_idx] += v_vals[l]
                if w_vals[l] != 0:
                    w_idx = (col_c_start + l%2) * n + (row_c_start + l//2)
                    W[w_idx, col_idx] += w_vals[l]
            col_idx += 1

    # Helper to add a 2x4x2 block (Rank 14)
    def add_2x4x2(row_a_start, row_b_start, col_b_start, row_c_start, col_c_start):
        # A: rows row_a_start..row_a_start+1, cols 0..3 (global)
        # B: rows row_b_start..row_b_start+3, cols col_b_start..col_b_start+1
        # C: rows row_c_start..row_c_start+1, cols col_c_start..col_c_start+1
        
        # Split A into A_left (cols 0-1) and A_right (cols 2-3)
        # Split B into B_top (rows 0-1) and B_bottom (rows 2-3)
        
        # C = A_left B_top + A_right B_bottom
        
        # Term 1: A_left (2x2) * B_top (2x2)
        add_strassen_2x2x2(row_a_start, 0, row_b_start, col_b_start, row_c_start, col_c_start)
        
        # Term 2: A_right (2x2) * B_bottom (2x2)
        add_strassen_2x2x2(row_a_start, 2, row_b_start + 2, col_b_start, row_c_start, col_c_start)

    # Execute Blocks
    
    # 1. C_1L1: A row 0, B cols 0-1, C cols 0-1
    add_1x4x2(0, 0, 0)
    
    # 2. C_1L2: A row 0, B cols 2-3, C cols 2-3
    add_1x4x2(0, 2, 2)
    
    # 3. C_1R1: A row 0, B cols 4-5, C cols 4-5
    add_1x4x2(0, 4, 4)
    
    # 4. C_1R2: A row 0, B cols 6-7, C cols 6-7
    add_1x4x2(0, 6, 6)
    
    # 5. C_2L1: A rows 1-2, B rows 0-3 cols 0-1, C rows 1-2 cols 0-1
    add_2x4x2(1, 0, 0, 1, 0)
    
    # 6. C_2L2: A rows 1-2, B rows 0-3 cols 2-3, C rows 1-2 cols 2-3
    add_2x4x2(1, 0, 2, 1, 2)
    
    # 7. C_2R1: A rows 1-2, B rows 0-3 cols 4-5, C rows 1-2 cols 4-5
    add_2x4x2(1, 0, 4, 1, 4)
    
    # 8. C_2R2: A rows 1-2, B rows 0-3 cols 6-7, C rows 1-2 cols 6-7
    add_2x4x2(1, 0, 6, 1, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

