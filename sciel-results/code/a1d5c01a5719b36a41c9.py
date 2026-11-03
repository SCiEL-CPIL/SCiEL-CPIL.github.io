import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 5
    
    # Dimensions for factor matrices
    dim_U = n * m  # 16
    dim_V = m * p  # 20
    dim_W = p * n  # 20
    
    # Target rank: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1) = 65
    rank = 65
    
    U = np.zeros((dim_U, rank), dtype=object)
    V = np.zeros((dim_V, rank), dtype=object)
    W = np.zeros((dim_W, rank), dtype=object)
    
    # --- Part 1: Strassen 4x4x4 (Rank 49) ---
    
    # Base Strassen 2x2x2 factors (Rank 7)
    # U_b: (4, 7), V_b: (4, 7), W_b: (4, 7)
    # Indices in base: i,j in 0..1 for U/V/W blocks
    U_b = np.zeros((4, 7), dtype=object)
    V_b = np.zeros((4, 7), dtype=object)
    W_b = np.zeros((4, 7), dtype=object)
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U_b[0, 0] = 1; U_b[3, 0] = 1
    V_b[0, 0] = 1; V_b[3, 0] = 1
    W_b[0, 0] = 1; W_b[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U_b[2, 1] = 1; U_b[3, 1] = 1
    V_b[0, 1] = 1
    W_b[2, 1] = 1 # C10 is index 2 in 2x2 (k=1, i=0 -> 1*2+0=2)
    
    # Term 2: M3 = A00(B01-B11)
    U_b[0, 2] = 1
    V_b[1, 2] = 1; V_b[3, 2] = -1
    W_b[1, 2] = 1 # C01 is index 1 (k=0, i=1 -> 0*2+1=1)
    
    # Term 3: M4 = A11(B10-B00)
    U_b[3, 3] = 1
    V_b[2, 3] = 1; V_b[0, 3] = -1
    W_b[2, 3] = 1 # C10
    
    # Term 4: M5 = (A00+A01)B11
    U_b[0, 4] = 1; U_b[1, 4] = 1
    V_b[3, 4] = 1
    W_b[0, 4] = -1 # C00
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U_b[2, 5] = 1; U_b[0, 5] = -1
    V_b[0, 5] = 1; V_b[1, 5] = 1
    W_b[3, 5] = 1 # C11
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U_b[1, 6] = 1; U_b[3, 6] = -1
    V_b[2, 6] = 1; V_b[3, 6] = 1
    W_b[0, 6] = 1 # C00
    
    # Super-terms definitions
    # Each super-term k (0..6) has:
    # A_blocks: list of (block_idx, coeff) where block_idx 0=A11, 1=A12, 2=A21, 3=A22
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    
    super_terms = [
        # P1
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # P2
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1)]},
        # P3
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1)]},
        # P4
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(2, 1)]},
        # P5
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (3, -1)]}, # C00 -1, C22 -1? 
        # Check P5 formula: C11 -= P5, C22? 
        # C11 = P1 + P4 - P5 + P7. C22 = P1 - P2 + P3 + P6.
        # So P5 only affects C11 with -1.
        # My C_blocks for P5 should be [(0, -1)]?
        # Wait, C11 is block 0. C22 is block 3.
        # So P5 contributes -1 to C11.
        # Let's re-verify C_blocks for all terms.
    ]
    
    # Correct C_blocks based on:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    super_terms = [
        # P1
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # P2
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1)]},
        # P3
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # P4
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        # P5
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # P6
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # P7
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]
    
    # Map block indices to global offsets
    # A blocks: 0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)
    # B blocks: 0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2) -- Note B is 4x4 here
    # C blocks: 0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)
    
    def get_offset(block_idx):
        # Returns (row_offset, col_offset) for 2x2 blocks in 4x4
        r = (block_idx // 2) * 2
        c = (block_idx % 2) * 2
        return r, c
    
    for k, term in enumerate(super_terms):
        for t in range(7):
            col_idx = k * 7 + t
            
            # Fill U
            for block_idx, coeff in term['A']:
                r_off, c_off = get_offset(block_idx)
                for i in range(2):
                    for j in range(2):
                        val = U_b[i*2+j, t]
                        if val != 0:
                            U[(r_off+i)*4 + (c_off+j), col_idx] += val * coeff
            
            # Fill V
            for block_idx, coeff in term['B']:
                r_off, c_off = get_offset(block_idx)
                for i in range(2): # row of B block
                    for j in range(2): # col of B block
                        val = V_b[i*2+j, t]
                        if val != 0:
                            V[(r_off+i)*4 + (c_off+j), col_idx] += val * coeff
            
            # Fill W
            for block_idx, coeff in term['C']:
                r_off, c_off = get_offset(block_idx)
                for i in range(2): # k (col of C)
                    for j in range(2): # i (row of C)
                        val = W_b[i*2+j, t] # W_b index is k*2+i? 
                        # In base 2x2, W index is k*2+i.
                        # Here i is row of C block, j is col of C block?
                        # W_b[k, i] corresponds to C[i, k].
                        # In base loop: W_b[i*2+j, t] where i=k, j=i_row.
                        # So val is for C[j, i].
                        if val != 0:
                            # Global C index: row = r_off + j, col = c_off + i
                            # W index = col_C * n + row_C = (c_off+i)*4 + (r_off+j)
                            W[(c_off+i)*4 + (r_off+j), col_idx] += val * coeff
    
    # --- Part 2: Schoolbook 4x4x1 (Rank 16) ---
    # B col 4 (index 4)
    # C col 4 (index 4)
    col_offset = 49
    for i in range(4):
        for j in range(4):
            idx = col_offset + i*4 + j
            U[i*4 + j, idx] = 1
            V[j*5 + 4, idx] = 1
            W[4*4 + i, idx] = 1
            
    # Convert to lists
    U_list = [[int(val) if isinstance(val, (int, np.integer)) else val for val in row] for row in U]
    V_list = [[int(val) if isinstance(val, (int, np.integer)) else val for val in row] for row in V]
    W_list = [[int(val) if isinstance(val, (int, np.integer)) else val for val in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
