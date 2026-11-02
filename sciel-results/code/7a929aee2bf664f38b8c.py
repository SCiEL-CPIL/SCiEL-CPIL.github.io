import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix using Strassen-like block decomposition.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen factors for 2x2 * 2x2
    # u, v are linear combos of inputs
    # w_coeffs are coefficients for the 4 outputs
    # Order of u/v inputs: (0,0), (0,1), (1,0), (1,1)
    # Order of w outputs: (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Define blocks
    # Block 1: A_L (cols 0-1), B_TL (rows 0-1, cols 0-1), C_L (cols 0-1)
    # Block 2: A_R (cols 2-3), B_BL (rows 2-3, cols 0-1), C_L (cols 0-1)
    # Block 3: A_L (cols 0-1), B_TR (rows 0-1, cols 2-3), C_R (cols 2-3)
    # Block 4: A_R (cols 2-3), B_BR (rows 2-3, cols 2-3), C_R (cols 2-3)
    
    blocks = [
        {
            'u_offset': [0, 1, 4, 5],   # A_L indices in U: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            'v_offset': [0, 1, 4, 5],   # B_TL indices in V: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            'w_offset': [0, 2, 1, 3],   # C_L indices in W: (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3
            'start_rank': 0
        },
        {
            'u_offset': [2, 3, 6, 7],   # A_R indices in U: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
            'v_offset': [8, 9, 12, 13], # B_BL indices in V: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
            'w_offset': [0, 2, 1, 3],   # C_L indices in W
            'start_rank': 7
        },
        {
            'u_offset': [0, 1, 4, 5],   # A_L indices in U
            'v_offset': [2, 3, 6, 7],   # B_TR indices in V: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
            'w_offset': [4, 6, 5, 7],   # C_R indices in W: (0,2)->4, (0,3)->6, (1,2)->5, (1,3)->7
            'start_rank': 14
        },
        {
            'u_offset': [2, 3, 6, 7],   # A_R indices in U
            'v_offset': [10, 11, 14, 15],# B_BR indices in V: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
            'w_offset': [4, 6, 5, 7],   # C_R indices in W
            'start_rank': 21
        }
    ]
    
    for block in blocks:
        u_off = block['u_offset']
        v_off = block['v_offset']
        w_off = block['w_offset']
        start_r = block['start_rank']
        
        for t_idx, (u_c, v_c, w_c) in enumerate(strassen_terms):
            r = start_r + t_idx
            
            for k in range(4):
                if u_c[k] != 0:
                    U[u_off[k]][r] += u_c[k]
                if v_c[k] != 0:
                    V[v_off[k]][r] += v_c[k]
                if w_c[k] != 0:
                    W[w_off[k]][r] += w_c[k]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
