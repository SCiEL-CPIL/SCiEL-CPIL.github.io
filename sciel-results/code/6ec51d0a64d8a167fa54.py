import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target Rank 85
    total_rank = 85
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # --- Part 1: M_4,4,4 (Rank 49) ---
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # U indices: i*4 + j (0..15)
    # V indices: j*4 + k (0..15)
    # W indices: k*4 + i (0..15)
    
    # Base Strassen 2x2 terms (u, v, w) of length 4
    base_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 0, 1, 0]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]), # M5
        ([1, 0, -1, 0], [1, 1, 0, 0], [1, 0, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1])  # M7
    ]
    
    # 4x4 Strassen steps
    # a_coeffs: block -> coeff
    # b_coeffs: block -> coeff
    # w_map: block -> coeff for C
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),     # M2
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),     # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),     # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),     # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 1}),     # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {3: 1})      # M7
    ]
    
    for a_coeffs, b_coeffs, w_map in steps:
        for u_base, v_base, w_base in base_terms:
            u_global = np.zeros(16)
            v_global = np.zeros(16)
            w_global = np.zeros(16)
            
            # Fill U
            for local_idx, val in enumerate(u_base):
                if val == 0: continue
                r = local_idx // 2
                c = local_idx % 2
                for b, c_ab in a_coeffs.items():
                    i0 = (b // 2) * 2
                    j0 = (b % 2) * 2
                    gi = i0 + r
                    gj = j0 + c
                    gidx = gi * 4 + gj
                    u_global[gidx] += val * c_ab
            
            # Fill V
            for local_idx, val in enumerate(v_base):
                if val == 0: continue
                r = local_idx // 2
                c = local_idx % 2
                for b, c_bb in b_coeffs.items():
                    i0 = (b // 2) * 2
                    j0 = (b % 2) * 2
                    gi = i0 + r
                    gj = j0 + c
                    gidx = gi * 4 + gj
                    v_global[gidx] += val * c_bb
            
            # Fill W
            for local_idx, val in enumerate(w_base):
                if val == 0: continue
                r = local_idx // 2
                c = local_idx % 2
                for b, c_wb in w_map.items():
                    i0 = (b // 2) * 2
                    j0 = (b % 2) * 2
                    # W index is k*n + i.
                    # In the 4x4x4 subproblem, n=4.
                    # Block b of C corresponds to k in [i0, i0+1] and i in [j0, j0+1]?
                    # No. C is 4x4.
                    # Block 0: k in 0,1; i in 0,1
                    # Block 1: k in 0,1; i in 2,3
                    # Block 2: k in 2,3; i in 0,1
                    # Block 3: k in 2,3; i in 2,3
                    # So i0 is k_start, j0 is i_start.
                    gk = i0 + r
                    gi = j0 + c
                    gidx = gk * 4 + gi
                    w_global[gidx] += val * c_wb
            
            U[0:16, col] = u_global
            V[0:16, col] = v_global
            W[0:16, col] = w_global
            col += 1
            
    # --- Part 2: M_4,4,1 (Rank 16) ---
    # A is 4x4 (cols 0-3), B is 4x1 (col 0 of the 4x1 block, which is col 4 of B? No.)
    # My decomposition:
    # A = [A_left (4x4) | A_right (4x1)]
    # B = [B_top (4x5); B_bottom (1x5)]
    # Product = A_left B_top + A_right B_bottom
    # A_left B_top:
    #   Split B_top into B_top_left (4x4) and B_top_right (4x1).
    #   A_left B_top_left -> M_4,4,4 (Done)
    #   A_left B_top_right -> M_4,4,1
    #   Here A is 4x4 (indices i, j in 0..3).
    #   B is 4x1 (indices j in 0..3, k=0).
    #   C is 4x1 (indices k=0, i in 0..3).
    #   Global indices:
    #   U: i*5 + j (i in 0..3, j in 0..3) -> 0..15
    #   V: j*5 + k (j in 0..3, k=4? No, B_top_right is cols 4 of B?
    #   Wait, B is 5x5.
    #   B_top is rows 0-3. B_top_right is col 4 of B_top.
    #   So j in 0..3, k=4.
    #   V index: j*5 + 4.
    #   W index: k*4 + i = 4*4 + i = 16 + i.
    
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
            
    # --- Part 3: M_4,1,5 (Rank 20) ---
    # A_right is 4x1 (col 4 of A).
    # B_bottom is 1x5 (row 4 of B).
    # A indices: i in 0..3, j=4.
    # B indices: j=4, k in 0..4.
    # C indices: k in 0..4, i in 0..3.
    # U index: i*5 + 4.
    # V index: 4*5 + k = 20 + k.
    # W index: k*4 + i.
    
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1
            
    # Convert to lists if required, or return numpy arrays
    # The interface expects {'U', 'V', 'W'}
    # The seed program returns lists of lists.
    # I will return numpy arrays as they are valid and often preferred, 
    # but to be safe with the "integer/half-integer entries" and standard library, 
    # I'll stick to numpy as it's allowed.
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res['U'].shape)
