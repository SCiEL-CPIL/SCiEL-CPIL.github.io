import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # M_k definitions: (U_coeffs, V_coeffs, W_coeffs)
    # W_coeffs map to C_00, C_01, C_10, C_11
    
    strassen_defs = [
        # M1
        (
            {0: 1, 2: 1},  # A00, A11 (indices in 2x2 block: 00=0, 01=1, 10=2, 11=3? No, let's use flat index in 2x2: 0,1,2,3)
            {0: 1, 3: 1},  # B00, B11
            {0: 1, 3: 1}    # C00, C11
        ),
        # M2
        (
            {2: 1, 3: 1},  # A10, A11
            {0: 1},        # B00
            {2: 1, 3: -1}  # C10, C11
        ),
        # M3
        (
            {0: 1},        # A00
            {1: 1, 3: -1}, # B01, B11
            {1: 1, 3: 1}   # C01, C11
        ),
        # M4
        (
            {3: 1},        # A11
            {2: 1, 0: -1}, # B10, B00
            {0: 1, 2: 1}   # C00, C10
        ),
        # M5
        (
            {0: 1, 1: 1},  # A00, A01
            {3: 1},        # B11
            {0: -1, 1: 1}  # C00, C01
        ),
        # M6
        (
            {2: 1, 0: -1}, # A10, A00
            {0: 1, 1: 1},  # B00, B01
            {3: 1}         # C11
        ),
        # M7
        (
            {1: 1, 3: -1}, # A01, A11
            {2: 1, 3: 1},  # B10, B11
            {0: 1}         # C00
        ),
    ]
    
    # 6 Strassen instances
    # s=0,1,2: L * B_top[s]
    # s=3,4,5: R * B_bot[s-3]
    
    for s in range(6):
        base_col = s * 7
        
        # Determine A block offset and B block offset
        if s < 3:
            # L: A cols 0,1
            a_col_offset = 0
            # B_top: B rows 0,1, cols 2s, 2s+1
            b_row_offset = 0
            b_col_offset = 2 * s
        else:
            # R: A cols 2,3
            a_col_offset = 2
            # B_bot: B rows 2,3, cols 2(s-3), 2(s-3)+1
            b_row_offset = 2
            b_col_offset = 2 * (s - 3)
            
        # Map local 2x2 indices to global U, V, W indices
        # Local A index l (0..3) -> global U index
        # A is 2x4. Local 2x2 is at cols a_col_offset, a_col_offset+1.
        # Local 0: (0, a_col_offset) -> U[0*4 + a_col_offset]
        # Local 1: (0, a_col_offset+1) -> U[0*4 + a_col_offset+1]
        # Local 2: (1, a_col_offset) -> U[1*4 + a_col_offset]
        # Local 3: (1, a_col_offset+1) -> U[1*4 + a_col_offset+1]
        
        # Local B index l (0..3) -> global V index
        # B is 4x6. Local 2x2 is at rows b_row_offset, b_row_offset+1 and cols b_col_offset, b_col_offset+1.
        # Local 0: (b_row_offset, b_col_offset) -> V[b_row_offset*6 + b_col_offset]
        # Local 1: (b_row_offset, b_col_offset+1) -> V[b_row_offset*6 + b_col_offset+1]
        # Local 2: (b_row_offset+1, b_col_offset) -> V[(b_row_offset+1)*6 + b_col_offset]
        # Local 3: (b_row_offset+1, b_col_offset+1) -> V[(b_row_offset+1)*6 + b_col_offset+1]
        
        # Local C index l (0..3) -> global W index
        # C is 2x6. Local 2x2 result is at cols 2s, 2s+1 (for s<3) or 2(s-3), 2(s-3)+1 (for s>=3).
        # Wait, the output column offset is the same as B column offset.
        c_col_offset = b_col_offset
        # Local 0: (0, c_col_offset) -> W[c_col_offset*2 + 0]
        # Local 1: (0, c_col_offset+1) -> W[(c_col_offset+1)*2 + 0]
        # Local 2: (1, c_col_offset) -> W[c_col_offset*2 + 1]
        # Local 3: (1, c_col_offset+1) -> W[(c_col_offset+1)*2 + 1]
        
        for k in range(7):
            col = base_col + k
            u_coeffs, v_coeffs, w_coeffs = strassen_defs[k]
            
            for local_a, val in u_coeffs.items():
                # Convert local 2x2 index to global U index
                row = local_a // 2
                col_a = local_a % 2
                global_u_idx = row * m + (a_col_offset + col_a)
                U[global_u_idx][col] = val
                
            for local_b, val in v_coeffs.items():
                # Convert local 2x2 index to global V index
                row_b = local_b // 2
                col_b = local_b % 2
                global_v_idx = (b_row_offset + row_b) * p + (b_col_offset + col_b)
                V[global_v_idx][col] = val
                
            for local_c, val in w_coeffs.items():
                # Convert local 2x2 index to global W index
                row_c = local_c // 2
                col_c = local_c % 2
                global_w_idx = (c_col_offset + col_c) * n + row_c
                W[global_w_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
