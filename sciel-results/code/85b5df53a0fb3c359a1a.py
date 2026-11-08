import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Target rank: 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Local indices for A: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # Local indices for B: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # Local indices for C: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    
    # Strassen terms definition:
    # term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {local_idx: val}
    # V_coeffs: dict {local_idx: val}
    # W_coeffs: dict {local_idx: val}
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}), # C10 = M2 + M4. So M2 contributes to C10 (idx 2).
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1}), # C01 = M3 + M5. M3 contributes to C01 (idx 1).
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # C00 has -M5+M7+M1+M4? 
        # Let's re-verify Strassen output formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1 -> C00 (1), C11 (1)
        # M2 -> C10 (1), C11 (-1)
        # M3 -> C01 (1), C11 (1)
        # M4 -> C00 (1), C10 (1)
        # M5 -> C00 (-1), C01 (1)
        # M6 -> C11 (1)
        # M7 -> C00 (1)
        
        # Let's redefine W_coeffs accordingly.
    ]
    
    # Corrected Strassen Terms (U, V, W)
    # Local A idx: 00:0, 01:1, 10:2, 11:3
    # Local B idx: 00:0, 01:1, 10:2, 11:3
    # Local C idx: 00:0, 01:1, 10:2, 11:3
    
    terms = [
        # M1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # C10(1), C11(-1)
        # M3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # C01(1), C11(1)
        # M4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # C00(1), C10(1)
        # M5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # C00(-1), C01(1)
        # M6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C11(1)
        # M7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}), # C00(1)
    ]
    
    # Sub-problems:
    # 1. A_L (cols 0,1) with B rows 0,1. C cols 0,1
    # 2. A_L (cols 0,1) with B rows 0,1. C cols 2,3
    # 3. A_L (cols 0,1) with B rows 0,1. C cols 4,5
    # 4. A_L (cols 0,1) with B rows 0,1. C cols 6,7
    # 5. A_R (cols 2,3) with B rows 2,3. C cols 0,1
    # 6. A_R (cols 2,3) with B rows 2,3. C cols 2,3
    # 7. A_R (cols 2,3) with B rows 2,3. C cols 4,5
    # 8. A_R (cols 2,3) with B rows 2,3. C cols 6,7
    
    # Mapping local to global:
    # A local 0,1,2,3 -> Global A indices:
    # For A_L: Global A indices are 0,1,4,5 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    #   Local 0 (00) -> Global 0
    #   Local 1 (01) -> Global 1
    #   Local 2 (10) -> Global 4
    #   Local 3 (11) -> Global 5
    # For A_R: Global A indices are 2,3,6,7
    #   Local 0 (00) -> Global 2
    #   Local 1 (01) -> Global 3
    #   Local 2 (10) -> Global 6
    #   Local 3 (11) -> Global 7
    
    # B local 0,1,2,3 -> Global B indices:
    # B index = j * 8 + k
    # For B rows 0,1:
    #   Local 0 (00) -> j=0, k=offset+0 -> Global 0*8 + (2*block+0)
    #   Local 1 (01) -> j=0, k=offset+1 -> Global 0*8 + (2*block+1)
    #   Local 2 (10) -> j=1, k=offset+0 -> Global 1*8 + (2*block+0)
    #   Local 3 (11) -> j=1, k=offset+1 -> Global 1*8 + (2*block+1)
    # For B rows 2,3:
    #   Local 0 (00) -> j=2, k=offset+0 -> Global 2*8 + (2*block+0)
    #   Local 1 (01) -> j=2, k=offset+1 -> Global 2*8 + (2*block+1)
    #   Local 2 (10) -> j=3, k=offset+0 -> Global 3*8 + (2*block+0)
    #   Local 3 (11) -> j=3, k=offset+1 -> Global 3*8 + (2*block+1)
    
    # C local 0,1,2,3 -> Global C indices:
    # C index = k * 2 + i
    # For C cols 2*block, 2*block+1:
    #   Local 0 (00) -> i=0, k=2*block -> Global (2*block)*2 + 0
    #   Local 1 (01) -> i=0, k=2*block+1 -> Global (2*block+1)*2 + 0
    #   Local 2 (10) -> i=1, k=2*block -> Global (2*block)*2 + 1
    #   Local 3 (11) -> i=1, k=2*block+1 -> Global (2*block+1)*2 + 1
    
    sub_problems = []
    for block in range(4):
        # A_L part
        a_map = [0, 1, 4, 5]
        b_row_base = 0
        c_col_base = block * 2
        sub_problems.append((a_map, b_row_base, c_col_base))
        
        # A_R part
        a_map = [2, 3, 6, 7]
        b_row_base = 2
        c_col_base = block * 2
        sub_problems.append((a_map, b_row_base, c_col_base))
    
    for a_map, b_row_base, c_col_base in sub_problems:
        # a_map: [local0, local1, local2, local3] -> global A indices
        # b_row_base: 0 or 2
        # c_col_base: 0, 2, 4, 6
        
        # Construct B map
        b_map = [
            b_row_base * 8 + c_col_base,
            b_row_base * 8 + c_col_base + 1,
            (b_row_base + 1) * 8 + c_col_base,
            (b_row_base + 1) * 8 + c_col_base + 1
        ]
        
        # Construct C map
        c_map = [
            c_col_base * 2 + 0,
            (c_col_base + 1) * 2 + 0,
            c_col_base * 2 + 1,
            (c_col_base + 1) * 2 + 1
        ]
        
        for u_loc, v_loc, w_loc in terms:
            for l_idx, val in u_loc.items():
                g_idx = a_map[l_idx]
                U[g_idx][col] += val
            for l_idx, val in v_loc.items():
                g_idx = b_map[l_idx]
                V[g_idx][col] += val
            for l_idx, val in w_loc.items():
                g_idx = c_map[l_idx]
                W[g_idx][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
