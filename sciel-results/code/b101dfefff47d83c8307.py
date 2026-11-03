import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 8
    
    # Define the rank-7 decomposition for 2x2 * 2x2 -> 2x2 (Strassen)
    # U_local[i, k] = coeff for A[i, j_i] in product k
    # V_local[j, k] = coeff for B[j_i, j] in product k
    # W_local[i, k] = coeff for C[i, j] from product k
    # Indices 0,1 for rows/cols of 2x2 matrices.
    
    # Strassen factors (7 terms)
    # 0: M1 = (A00+A11)(B00+B11)
    # 1: M2 = (A10+A11)B00
    # 2: M3 = A00(B10-B00)
    # 3: M4 = A11(B01-B11)
    # 4: M5 = (A00+A01)B11
    # 5: M6 = (A10-A00)(B00+B01)
    # 6: M7 = (A01-A11)(B10+B11)
    
    # U factors (4x7), row idx: 0:00, 1:01, 2:10, 3:11
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, -1], # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, 1]   # A11
    ], dtype=int)
    
    # V factors (4x7), row idx: 0:00, 1:01, 2:10, 3:11
    V_s = np.array([
        [1, 1, 1, 0, 0, 1, 0], # B00
        [0, 0, -1, 1, 0, 0, 0],# B01
        [0, 0, 1, 0, 1, 1, 1], # B10
        [1, 0, 0, -1, 1, 0, 1] # B11
    ], dtype=int)
    
    # W factors (4x7), row idx: 0:00, 1:01, 2:10, 3:11
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00 = M1 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],  # C01 = M3 + M5
        [0, 1, 0, 0, 0, 1, 0],  # C10 = M2 + M6
        [1, -1, 0, 1, 0, 0, 1]  # C11 = M1 - M2 + M4 + M7
    ], dtype=int)
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # The 2x4 * 4x8 multiplication splits into 4 independent 2x4 * 4x2 subproblems.
    # Let B be split into 4 blocks B_0..B_3, each 4x2.
    # Let A be split into A_left (cols 0,1) and A_right (cols 2,3).
    # C_i (2x2) = A_left * B_i_top (2x2) + A_right * B_i_bottom (2x2).
    # Each term is a 2x2*2x2 multiplication, decomposed into 7 rank-1 tensors.
    # Total 14 rank-1 tensors per block i.
    
    # Index mappings for A (2x4) -> 4 local indices for U_s (2x2 blocks)
    # A indices: 0:A00, 1:A01, 2:A02, 3:A03, 4:A10, 5:A11, 6:A12, 7:A13
    # A_left (A00,A01,A10,A11) maps to U_s rows [0,1,2,3]
    # A_right (A02,A03,A12,A13) maps to U_s rows [0,1,2,3]
    
    for i in range(4):
        # Base rank offset for this block
        base = i * 14
        
        # --- Term 1: A_left * B_i_top ---
        # B_i is columns 2i, 2i+1 of B.
        # B_i_top (rows 0,1) -> B[0, 2i], B[0, 2i+1], B[1, 2i], B[1, 2i+1]
        # B_i_bottom (rows 2,3) -> B[2, 2i], B[2, 2i+1], B[3, 2i], B[3, 2i+1]
        
        # A_left indices in global U: 0,1,4,5 (rows 0,1 of A, cols 0,1)
        # Map U_s row 0->U_idx0, 1->U_idx1, 2->U_idx4, 3->U_idx5
        u_map_l = [0, 1, 4, 5]
        
        # B_i_top indices in global V:
        # B[0, 2i]=2i, B[0, 2i+1]=2i+1
        # B[1, 2i]=8+2i, B[1, 2i+1]=8+2i+1
        v_map_top = [2*i, 2*i+1, 8+2*i, 8+2*i+1]
        
        # C_i indices in global W:
        # C[0, 2i]=2i*2+0=4i, C[0, 2i+1]=4i+2
        # C[1, 2i]=4i+1, C[1, 2i+1]=4i+3
        # W_s row 0->W_idx4i, 1->W_idx4i+2, 2->W_idx4i+1, 3->W_idx4i+3
        w_map = [4*i, 4*i+2, 4*i+1, 4*i+3]
        
        # Embed first 7 terms
        for r in range(7):
            rank_idx = base + r
            # U
            for la in range(4):
                coeff = U_s[la, r]
                if coeff:
                    U[u_map_l[la], rank_idx] += coeff
            # V
            for lb in range(4):
                coeff = V_s[lb, r]
                if coeff:
                    V[v_map_top[lb], rank_idx] += coeff
            # W
            for lc in range(4):
                coeff = W_s[lc, r]
                if coeff:
                    W[w_map[lc], rank_idx] += coeff
        
        # --- Term 2: A_right * B_i_bottom ---
        # A_right indices in global U: 2,3,6,7 (rows 0,1 of A, cols 2,3)
        # Map U_s row 0->U_idx2, 1->U_idx3, 2->U_idx6, 3->U_idx7
        u_map_r = [2, 3, 6, 7]
        
        # B_i_bottom indices in global V:
        # B[2, 2i]=16+2i, B[2, 2i+1]=16+2i+1
        # B[3, 2i]=24+2i, B[3, 2i+1]=24+2i+1
        v_map_bot = [16+2*i, 16+2*i+1, 24+2*i, 24+2*i+1]
        
        # W is the same as Term 1 (same output C_i)
        
        # Embed next 7 terms
        for r in range(7):
            rank_idx = base + 7 + r
            # U
            for la in range(4):
                coeff = U_s[la, r]
                if coeff:
                    U[u_map_r[la], rank_idx] += coeff
            # V
            for lb in range(4):
                coeff = V_s[lb, r]
                if coeff:
                    V[v_map_bot[lb], rank_idx] += coeff
            # W
            for lc in range(4):
                coeff = W_s[lc, r]
                if coeff:
                    W[w_map[lc], rank_idx] += coeff
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
