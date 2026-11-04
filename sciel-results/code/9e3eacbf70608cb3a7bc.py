import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # Strategy:
    # Split n=4 into 2 blocks of 2.
    # Split p=6 into 3 blocks of 2.
    # Split m=5 into 2 blocks: 2 and 3.
    
    # For each (i_block, k_block) pair (6 pairs):
    #   C_ik = A_i B_k
    #   A_i is 2x5, B_k is 5x2.
    #   Split A_i into A_i0 (2x2) and A_i1 (2x3).
    #   Split B_k into B_k0 (2x2) and B_k1 (3x2).
    #   C_ik = A_i0 B_k0 + A_i1 B_k1.
    #   Term 1: 2x2x2 multiplication. Use Strassen (Rank 7).
    #   Term 2: 2x3x2 multiplication. Use Schoolbook (Rank 12).
    #   Total per pair: 19.
    #   Total Rank: 6 * 19 = 114.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        # u_idx is index in U (n*m), v_idx in V (m*p), w_idx in W (p*n)
        # We are building lists of terms (columns)
        U_list.append({u_idx: u_val})
        V_list.append({v_idx: v_val})
        W_list.append({w_idx: w_val})

    # Helper for Strassen 2x2x2
    # A: 2x2, B: 2x2, C: 2x2
    # Returns list of (u_indices, v_indices, w_indices, coeffs)
    # We need to map local indices to global indices.
    
    def strassen_222(A_base, B_base, C_base, U_list, V_list, W_list):
        # A_base is (i, j_offset) where i in 0..1, j_offset in 0..1
        # B_base is (j_offset, k) where k in 0..1
        # C_base is (i, k)
        
        # Global indices:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Local j is 0, 1. Global j = j_offset + local_j.
        # Local k is 0, 1. Global k = k_offset + local_k.
        # Local i is 0, 1. Global i = i_offset + local_i.
        
        # We need to pass offsets.
        pass

    # Let's just build the matrices directly for the 114 rank case.
    # It's easier to generate the full U, V, W arrays.
    
    rank = 0
    # 6 blocks * (7 + 12)
    rank = 6 * 19
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Offsets for blocks
    # i_block in 0, 1 -> i_offset = i_block * 2
    # k_block in 0, 1, 2 -> k_offset = k_block * 2
    # m_split: 0->2, 1->3. j_offset_0 = 0, j_offset_1 = 2.
    
    for i_block in range(2):
        i_off = i_block * 2
        for k_block in range(3):
            k_off = k_block * 2
            
            # Part 1: A_i0 (2x2) * B_k0 (2x2) -> C_ik part 1
            # j range 0..1 (offset 0)
            # Use Strassen
            # Local indices i_l in 0,1; j_l in 0,1; k_l in 0,1
            # Global i = i_off + i_l
            # Global j = 0 + j_l
            # Global k = k_off + k_l
            
            # Strassen terms for 2x2x2
            # M1 = A00 * (B00 + B11)
            # M2 = (A00 + A10) * B11
            # M3 = (A01 + A11) * B00
            # M4 = A11 * (B01 - B00)
            # M5 = (A00 + A11) * B10
            # M6 = (A01 - A00) * (B00 + B10)
            # M7 = (A10 - A11) * (B11 + B01)
            
            # We need to add 7 columns.
            
            # Helper to add Strassen term
            def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
                # u_coeffs: dict { (i_l, j_l): val }
                # v_coeffs: dict { (j_l, k_l): val }
                # w_coeffs: dict { (k_l, i_l): val }
                for (il, jl), val in u_coeffs.items():
                    U[i_off + il, m * (jl) + k_off * 0 + 0 + 0, ] # Wait, U index is i*m + j.
                    # U index: (i_off + il) * m + (0 + jl)
                    U[(i_off + il) * m + jl, col_idx] += val
                for (jl, kl), val in v_coeffs.items():
                    # V index: j*p + k
                    # j = 0 + jl, k = k_off + kl
                    V[jl * p + (k_off + kl), col_idx] += val
                for (kl, il), val in w_coeffs.items():
                    # W index: k*n + i
                    # k = k_off + kl, i = i_off + il
                    W[(k_off + kl) * n + (i_off + il), col_idx] += val

            # M1
            add_strassen_term({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): 1}, col); col += 1
            # M2
            add_strassen_term({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,1): 1, (1,1): -1, (0,0): 1, (1,0): -1}, col); col += 1
            # M3
            add_strassen_term({(0,1): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1, (1,0): 1, (0,1): 1, (1,1): 1}, col); col += 1
            # M4
            add_strassen_term({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,1): -1, (1,1): 1, (0,0): 1, (1,0): -1}, col); col += 1
            # M5
            add_strassen_term({(0,0): 1, (1,1): 1}, {(1,0): 1}, {(0,0): -1, (1,0): -1, (0,1): 1, (1,1): 1}, col); col += 1
            # M6
            add_strassen_term({(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): -1, (0,1): -1, (1,1): 1}, col); col += 1
            # M7
            add_strassen_term({(1,0): 1, (1,1): -1}, {(1,1): 1, (0,1): 1}, {(0,0): -1, (1,0): 1, (0,1): -1, (1,1): 1}, col); col += 1
            
            # Part 2: A_i1 (2x3) * B_k1 (3x2) -> C_ik part 2
            # j range 2..4 (offset 2)
            # Schoolbook for 2x3x2. Rank 12.
            # i_l in 0,1. j_l in 0,1,2 (local). k_l in 0,1.
            # Global j = 2 + j_l.
            
            for il in range(2):
                for jl in range(3):
                    for kl in range(2):
                        U[(i_off + il) * m + (2 + jl), col] = 1.0
                        V[(2 + jl) * p + (k_off + kl), col] = 1.0
                        W[(k_off + kl) * n + (i_off + il), col] = 1.0
                        col += 1
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
