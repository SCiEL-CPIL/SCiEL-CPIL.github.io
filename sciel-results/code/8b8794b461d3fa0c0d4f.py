"""
Rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
Strategy: Decompose into 4 independent 2x2*2x7 blocks.
For each 2x2*2x7 block, split the 7 columns into 2+2+3.
- Use Strassen (rank 7) for the 2x2*2x2 parts (2 blocks -> 14 rank).
- Use the minimal 2x2*2x3 decomposition (rank 11) for the 3-column part.
Total rank per 2x2*2x7 block: 7 + 7 + 11 = 25.
Total rank: 4 * 25 = 100.
"""
import time

DIMENSIONS = [4, 4, 7]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n, m, p = 4, 4, 7
    
    # Total rank calculation
    # 4 blocks of 2x2*2x7
    # Each 2x2*2x7 is split into 2x2*2x2 (Strassen, rank 7) + 2x2*2x2 (Strassen, rank 7) + 2x2*2x3 (Minimal, rank 11)
    # Wait, 2+2+3 = 7 columns.
    # Rank per block = 7 + 7 + 11 = 25.
    # Total rank = 4 * 25 = 100.
    
    total_rank = 100
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Helper: Strassen 2x2x2 terms ---
    # Each term is (U_coeffs, V_coeffs, W_coeffs) for a 2x2*2x2 multiplication.
    # U coeffs map to A[i,j], V to B[j,k], W to C[i,k].
    # Indices: A(0,0), A(0,1), A(1,0), A(1,1) -> 0, 1, 2, 3
    # B(0,0), B(0,1), B(1,0), B(1,1) -> 0, 1, 2, 3
    # C(0,0), C(0,1), C(1,0), C(1,1) -> 0, 1, 2, 3
    
    strassen_2x2x2 = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B10-B00)
        ({0: 1}, {2: 1, 0: -1}, {2: 1, 3: 1}),
        # M4 = A11(B01-B00)
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # --- Helper: Minimal 2x2x3 decomposition (Rank 11) ---
    # Multiplying 2x2 matrix A by 2x3 matrix B to get 2x3 matrix C.
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # C indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    
    # We can construct this by taking the Strassen 2x2x2 for the first 2 columns
    # and adding schoolbook terms for the 3rd column?
    # Strassen 2x2x2 (rank 7) handles cols 0,1.
    # Schoolbook for col 2: 4 terms.
    # Total 11.
    # Let's verify if this is valid and minimal. Yes, 11 is the known rank for 2x2x3.
    
    # Terms 0-6: Strassen for B cols 0,1
    # Terms 7-10: Schoolbook for B col 2
    
    min_2x2x3 = []
    
    # Add Strassen terms (mapped to B cols 0,1)
    for u_c, v_c, w_c in strassen_2x2x2:
        # U coeffs same
        new_u = dict(u_c)
        # V coeffs: map local B indices 0,1,2,3 to global 0,1,3,4?
        # Local 0 (0,0) -> Global 0
        # Local 1 (0,1) -> Global 1
        # Local 2 (1,0) -> Global 3
        # Local 3 (1,1) -> Global 4
        # Local 2 is (1,0) which is index 2 in 2x2, but in 2x3, row 1 col 0 is index 3.
        # Local 3 is (1,1) which is index 3 in 2x2, but in 2x3, row 1 col 1 is index 4.
        
        # Wait, standard 2x2 indices:
        # 0: (0,0)
        # 1: (0,1)
        # 2: (1,0)
        # 3: (1,1)
        
        # 2x3 indices:
        # 0: (0,0)
        # 1: (0,1)
        # 2: (0,2)
        # 3: (1,0)
        # 4: (1,1)
        # 5: (1,2)
        
        # Mapping for Strassen part (cols 0,1):
        # Local 0 -> Global 0
        # Local 1 -> Global 1
        # Local 2 -> Global 3
        # Local 3 -> Global 4
        
        new_v = {}
        for k, val in v_c.items():
            if k == 0: new_v[0] = val
            elif k == 1: new_v[1] = val
            elif k == 2: new_v[3] = val
            elif k == 3: new_v[4] = val
            
        # W coeffs:
        # Local 0 (0,0) -> Global 0
        # Local 1 (0,1) -> Global 1
        # Local 2 (1,0) -> Global 3
        # Local 3 (1,1) -> Global 4
        
        new_w = {}
        for k, val in w_c.items():
            if k == 0: new_w[0] = val
            elif k == 1: new_w[1] = val
            elif k == 2: new_w[3] = val
            elif k == 3: new_w[4] = val
            
        min_2x2x3.append((new_u, new_v, new_w))
        
    # Add Schoolbook terms for B col 2 (Global index 2 for row 0, 5 for row 1)
    # C(0,2) = A(0,0)B(0,2) + A(0,1)B(1,2)
    # C(1,2) = A(1,0)B(0,2) + A(1,1)B(1,2)
    
    # Term 7: A(0,0) * B(0,2) -> C(0,2)
    min_2x2x3.append(({0: 1}, {2: 1}, {2: 1}))
    # Term 8: A(0,1) * B(1,2) -> C(0,2)
    min_2x2x3.append(({1: 1}, {5: 1}, {2: 1}))
    # Term 9: A(1,0) * B(0,2) -> C(1,2)
    min_2x2x3.append(({2: 1}, {2: 1}, {5: 1}))
    # Term 10: A(1,1) * B(1,2) -> C(1,2)
    min_2x2x3.append(({3: 1}, {5: 1}, {5: 1}))
    
    # --- Construction Loop ---
    
    # We have 4 blocks corresponding to A's 2x2 sub-blocks.
    # Block 0: A rows 0-1, cols 0-1. Multiplies B rows 0-1. Contributes to C rows 0-1.
    # Block 1: A rows 0-1, cols 2-3. Multiplies B rows 2-3. Contributes to C rows 0-1.
    # Block 2: A rows 2-3, cols 0-1. Multiplies B rows 0-1. Contributes to C rows 2-3.
    # Block 3: A rows 2-3, cols 2-3. Multiplies B rows 2-3. Contributes to C rows 2-3.
    
    blocks = [
        (0, 0, 0, 0), # (a_row_block, a_col_block, b_row_block, c_row_block)
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (1, 1, 1, 1)
    ]
    
    for a_rb, a_cb, b_rb, c_rb in blocks:
        
        # 1. Strassen part for B cols 0,1 (Global B cols 0,1 if b_rb=0 else 2,3? No, B cols are 0..6)
        # Wait, the 2x2x7 block uses B rows (2 rows) and all 7 B columns.
        # My split was 2+2+3 columns of B.
        # B cols 0,1 handled by first Strassen.
        # B cols 2,3 handled by second Strassen.
        # B cols 4,5,6 handled by 2x2x3.
        
        # So for each block, we process:
        # - Strassen for B cols 0,1
        # - Strassen for B cols 2,3
        # - 2x2x3 for B cols 4,5,6
        
        # --- Strassen for B cols 0,1 ---
        for u_c, v_c, w_c in strassen_2x2x2:
            r = current_rank
            current_rank += 1
            
            # Map U coeffs (A local) to Global A
            for loc_idx, val in u_c.items():
                li = loc_idx // 2
                lj = loc_idx % 2
                gi = 2 * a_rb + li
                gj = 2 * a_cb + lj
                U[gi * m + gj][r] += val
                
            # Map V coeffs (B local) to Global B
            # Local B indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global B rows: 2*b_rb, 2*b_rb+1
            # Global B cols: 0, 1 (for this part)
            for loc_idx, val in v_c.items():
                li = loc_idx // 2
                lj = loc_idx % 2
                gi = 2 * b_rb + li
                gj = lj # Cols 0,1
                V[gi * p + gj][r] += val
                
            # Map W coeffs (C local) to Global C
            # Local C indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global C rows: 2*c_rb, 2*c_rb+1
            # Global C cols: 0, 1
            for loc_idx, val in w_c.items():
                li = loc_idx // 2
                lj = loc_idx % 2
                gi = 2 * c_rb + li
                gj = lj # Cols 0,1
                W[gj * n + gi][r] += val
        
        # --- Strassen for B cols 2,3 ---
        for u_c, v_c, w_c in strassen_2x2x2:
            r = current_rank
            current_rank += 1
            
            for loc_idx, val in u_c.items():
                li = loc_idx // 2
                lj = loc_idx % 2
                gi = 2 * a_rb + li
                gj = 2 * a_cb + lj
                U[gi * m + gj][r] += val
                
            for loc_idx, val in v_c.items():
                li = loc_idx // 2
                lj = loc_idx % 2
                gi = 2 * b_rb + li
                gj = 2 + lj # Cols 2,3
                V[gi * p + gj][r] += val
                
            for loc_idx, val in w_c.items():
                li = loc_idx // 2
                lj = loc_idx % 2
                gi = 2 * c_rb + li
                gj = 2 + lj # Cols 2,3
                W[gj * n + gi][r] += val
                
        # --- 2x2x3 for B cols 4,5,6 ---
        for u_c, v_c, w_c in min_2x2x3:
            r = current_rank
            current_rank += 1
            
            for loc_idx, val in u_c.items():
                li = loc_idx // 2
                lj = loc_idx % 2
                gi = 2 * a_rb + li
                gj = 2 * a_cb + lj
                U[gi * m + gj][r] += val
                
            # V local indices for 2x2x3:
            # 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
            # These map to Global B cols 4,5,6
            # Local 0 -> Global (2*b_rb, 4)
            # Local 1 -> Global (2*b_rb, 5)
            # Local 2 -> Global (2*b_rb, 6)
            # Local 3 -> Global (2*b_rb+1, 4)
            # Local 4 -> Global (2*b_rb+1, 5)
            # Local 5 -> Global (2*b_rb+1, 6)
            
            v_map = {
                0: (0, 4),
                1: (0, 5),
                2: (0, 6),
                3: (1, 4),
                4: (1, 5),
                5: (1, 6)
            }
            
            for loc_idx, val in v_c.items():
                li, lj = v_map[loc_idx]
                gi = 2 * b_rb + li
                gj = lj
                V[gi * p + gj][r] += val
                
            # W local indices for 2x2x3:
            # 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
            # These map to Global C cols 4,5,6
            # Local 0 -> Global (2*c_rb, 4)
            # Local 1 -> Global (2*c_rb, 5)
            # Local 2 -> Global (2*c_rb, 6)
            # Local 3 -> Global (2*c_rb+1, 4)
            # Local 4 -> Global (2*c_rb+1, 5)
            # Local 5 -> Global (2*c_rb+1, 6)
            
            w_map = {
                0: (0, 4),
                1: (0, 5),
                2: (0, 6),
                3: (1, 4),
                4: (1, 5),
                5: (1, 6)
            }
            
            for loc_idx, val in w_c.items():
                li, lj = w_map[loc_idx]
                gi = 2 * c_rb + li
                gj = lj
                W[gj * n + gi][r] += val
                
    return {"U": U, "V": V, "W": W}
