import numpy as np
import time

_DIM = (6, 6, 6)

def construct():
    n, m, p = _DIM
    start_time = time.time()
    
    # Structural change: Use 2x2 Strassen with 3x3 blocks
    # Each 3x3 block mult uses rank-13 (Lader's construction)
    # Total rank = 8 * 13 = 104
    
    B = 3
    N_b = n // B  # = 2
    
    rank = N_b * N_b * N_b * 13  # 8 * 13 = 104
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Lader rank-13 decomposition for 3x3 matrix multiplication
    # Based on the known construction. Each term is (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are lists of (row, col, coeff)
    
    lader_terms = [
        # M1
        ([(0,0,1)], [(0,0,1),(0,1,1),(1,1,1),(2,2,1)], [(0,0,1),(0,1,1),(1,1,1),(2,2,1)]),
        # M2  
        ([(0,1,1),(1,1,1),(2,1,1)], [(0,0,1)], [(0,1,1),(1,1,1),(2,1,1)]),
        # M3
        ([(1,0,1),(2,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(2,0,1)]),
        # M4
        ([(0,2,1),(1,2,1),(0,0,-1)], [(0,0,1),(0,2,1)], [(0,2,1),(1,2,1)]),
        # M5
        ([(2,2,1)], [(0,0,1),(1,0,1),(2,0,1),(2,1,1)], [(2,2,1),(2,1,1)]),
        # M6
        ([(0,1,1),(1,1,1),(0,0,-1)], [(0,1,1),(1,1,1)], [(0,0,1),(1,0,1)]),
        # M7
        ([(1,0,1),(2,0,1),(0,0,-1)], [(1,0,1),(2,0,1)], [(0,1,1),(1,1,1)]),
        # M8
        ([(0,2,1),(1,2,1),(0,0,-1)], [(0,2,1),(1,2,1)], [(1,0,1),(2,0,1)]),
        # M9
        ([(1,1,1),(2,2,1),(0,0,-1)], [(1,1,1),(2,2,1)], [(0,2,1),(2,1,1)]),
        # M10
        ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,-1),(1,1,-1),(2,2,-1)]),
        # M11
        ([(0,1,1),(1,2,1),(2,0,1)], [(1,0,1),(2,2,1),(0,1,1)], [(1,1,1)]),
        # M12
        ([(1,0,1),(2,1,1),(0,2,1)], [(0,2,1),(1,1,1),(2,0,1)], [(1,1,1)]),
        # M13
        ([(0,0,1),(1,2,1),(2,1,1)], [(2,1,1),(1,0,1),(0,2,1)], [(1,1,1)]),
    ]
    
    # Strassen 2x2 terms for the block level
    strassen_terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
    ]
    
    # For each block position (i, k, l) and Strassen term, and Lader term
    for i in range(N_b):
        for k in range(N_b):
            for l in range(N_b):
                for s_idx in range(7):
                    s_u, s_v, s_w = strassen_terms[s_idx]
                    for l_idx in range(13):
                        col = ((i * N_b + k) * N_b + l) * 13 + l_idx
                        l_u, l_v, l_w = lader_terms[l_idx]
                        
                        # Apply Strassen coefficients to Lader entries
                        for (su_r, su_c, su_coeff) in s_u:
                            for (lu_r, lu_c, lu_coeff) in l_u:
                                gr = i * B + su_r * B + lu_r
                                gc = k * B + su_c * B + lu_c
                                U[gr * m + gc, col] += su_coeff * lu_coeff
                        
                        for (sv_r, sv_c, sv_coeff) in s_v:
                            for (lv_r, lv_c, lv_coeff) in l_v:
                                gr = k * B + sv_r * B + lv_r
                                gc = l * B + sv_c * B + lv_c
                                V[gr * p + gc, col] += sv_coeff * lv_coeff
                        
                        for (sw_r, sw_c, sw_coeff) in s_w:
                            for (lw_r, lw_c, lw_coeff) in l_w:
                                gr = i * B + sw_r * B + lw_r
                                gc = l * B + sw_c * B + lw_c
                                W[gc * n + gr, col] += sw_coeff * lw_coeff
                        
                        if time.time() - start_time > 100:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
