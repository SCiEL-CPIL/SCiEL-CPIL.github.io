import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms
    # A: [[a00, a01], [a10, a11]]
    # B: [[b00, b01], [b10, b11]]
    
    # u_coeffs[s] is a list of 4 coefficients for a00, a01, a10, a11
    u_coeffs = [
        [1, 0, 0, 1],  # M1: a + d
        [0, 0, 1, 1],  # M2: c + d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a + b
        [-1, 0, 1, 0], # M6: c - a
        [0, 1, 0, -1]  # M7: b - d
    ]
    
    # v_coeffs[s] is a list of 4 coefficients for b00, b01, b10, b11
    v_coeffs = [
        [1, 0, 0, 1],  # M1: e + h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f - h
        [0, 0, 1, -1], # M4: g - e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e + f
        [0, 0, 1, 1]   # M7: g + h
    ]
    
    # w_coeffs[s] is a dict mapping local C index (00, 01, 10, 11) to coefficient
    # Local indices: 0->00, 1->01, 2->10, 3->11
    w_coeffs = [
        {0: 1, 3: 1},      # M1: C00 + C11
        {2: 1, 3: -1},     # M2: C10 - C11
        {1: 1, 3: 1},      # M3: C01 + C11
        {2: 1, 0: 1},      # M4: C10 + C00
        {0: -1, 1: 1},     # M5: -C00 + C01
        {3: 1, 2: -1},     # M6: C11 - C10
        {0: 1}             # M7: C00
    ]

    for t in range(8):
        a_block = t // 4
        b_block = t % 4
        
        for s in range(7):
            c = t * 7 + s
            
            # Update U
            # Local A indices: (li, lj) in {0,1}
            # Global A index: i * 4 + (a_block * 2 + lj)
            # U row index is i * m + j_global
            # m = 4
            # j_global = a_block * 2 + lj
            # i is 0 or 1
            for li in range(2):
                for lj in range(2):
                    coeff = u_coeffs[s][li * 2 + lj]
                    if coeff != 0:
                        j_global = a_block * 2 + lj
                        i_global = li
                        u_row = i_global * m + j_global
                        U[u_row][c] += coeff
            
            # Update V
            # Local B indices: (lj, lk) in {0,1}
            # Global B index: (a_block * 2 + lj) * 8 + (b_block * 2 + lk)
            # V row index is j_global * p + k_global
            # p = 8
            for lj in range(2):
                for lk in range(2):
                    coeff = v_coeffs[s][lj * 2 + lk]
                    if coeff != 0:
                        j_global = a_block * 2 + lj
                        k_global = b_block * 2 + lk
                        v_row = j_global * p + k_global
                        V[v_row][c] += coeff
            
            # Update W
            # Local C indices: (li, lk) in {0,1}
            # Global C index: (b_block * 2 + lk) * 2 + li
            # W row index is k_global * n + i_global
            # n = 2
            for li in range(2):
                for lk in range(2):
                    local_c_idx = li * 2 + lk
                    if local_c_idx in w_coeffs[s]:
                        coeff = w_coeffs[s][local_c_idx]
                        k_global = b_block * 2 + lk
                        i_global = li
                        w_row = k_global * n + i_global
                        W[w_row][c] += coeff

    return {"U": U, "V": V, "W": W}
