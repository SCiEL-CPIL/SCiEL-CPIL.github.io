"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen decomposition for 2x2 * 2x2
    # Term t:
    # A_coeffs: dict (i_loc, j_loc) -> coeff
    # B_coeffs: dict (j_loc, k_loc) -> coeff
    # C_coeffs: dict (i_loc, k_loc) -> coeff
    # 
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B01-B11)       -> C01 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    
    strassen_terms = [
        {
            "A": {(0,0): 1, (1,1): 1},
            "B": {(0,0): 1, (1,1): 1},
            "C": {(0,0): 1, (1,1): 1}
        },
        {
            "A": {(1,0): 1, (1,1): 1},
            "B": {(0,0): 1},
            "C": {(1,0): 1}
        },
        {
            "A": {(0,0): 1},
            "B": {(0,1): 1, (1,1): -1},
            "C": {(0,1): 1}
        },
        {
            "A": {(1,1): 1},
            "B": {(1,0): 1, (0,0): -1},
            "C": {(0,0): 1, (1,0): 1}
        },
        {
            "A": {(0,0): 1, (0,1): 1},
            "B": {(1,1): 1},
            "C": {(0,0): -1, (0,1): 1}
        },
        {
            "A": {(1,0): 1, (0,0): -1},
            "B": {(0,0): 1, (0,1): 1},
            "C": {(1,1): 1}
        },
        {
            "A": {(0,1): 1, (1,1): -1},
            "B": {(1,0): 1, (1,1): 1},
            "C": {(0,0): 1, (1,1): -1}
        }
    ]
    
    col = 0
    
    # P1: A_L (cols 0,1) * B_T (rows 0,1)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0..7}
    # C indices: i in {0,1}, k in {0..7}
    
    for b_idx in range(4):
        # b_idx corresponds to columns 4*b_idx and 4*b_idx+1 of B
        # Local k_loc 0 -> global k = 4*b_idx
        # Local k_loc 1 -> global k = 4*b_idx + 1
        
        for term in strassen_terms:
            # Process A
            for (i_loc, j_loc), coeff_a in term["A"].items():
                # Global i = i_loc (0 or 1)
                # Global j = j_loc (0 or 1) for P1
                g_i = i_loc
                g_j = j_loc
                u_idx = g_i * m + g_j
                U[u_idx][col] += coeff_a
            
            # Process B
            for (j_loc, k_loc), coeff_b in term["B"].items():
                # Global j = j_loc (0 or 1) for P1 (matches B_T rows)
                # Global k = 4*b_idx + k_loc
                g_j = j_loc
                g_k = 4 * b_idx + k_loc
                v_idx = g_j * p + g_k
                V[v_idx][col] += coeff_b
            
            # Process C
            for (i_loc, k_loc), coeff_c in term["C"].items():
                # Global i = i_loc (0 or 1)
                # Global k = 4*b_idx + k_loc
                g_i = i_loc
                g_k = 4 * b_idx + k_loc
                w_idx = g_k * n + g_i
                W[w_idx][col] += coeff_c
            
            col += 1
            
    # P2: A_R (cols 2,3) * B_B (rows 2,3)
    # A indices: i in {0,1}, j in {2,3}
    # B indices: j in {2,3}, k in {0..7}
    # C indices: i in {0,1}, k in {0..7}
    
    for b_idx in range(4):
        for term in strassen_terms:
            # Process A
            for (i_loc, j_loc), coeff_a in term["A"].items():
                # Global i = i_loc (0 or 1)
                # Global j = 2 + j_loc (2 or 3) for P2
                g_i = i_loc
                g_j = 2 + j_loc
                u_idx = g_i * m + g_j
                U[u_idx][col] += coeff_a
            
            # Process B
            for (j_loc, k_loc), coeff_b in term["B"].items():
                # Global j = 2 + j_loc (2 or 3) for P2 (matches B_B rows)
                # Global k = 4*b_idx + k_loc
                g_j = 2 + j_loc
                g_k = 4 * b_idx + k_loc
                v_idx = g_j * p + g_k
                V[v_idx][col] += coeff_b
            
            # Process C
            for (i_loc, k_loc), coeff_c in term["C"].items():
                # Global i = i_loc (0 or 1)
                # Global k = 4*b_idx + k_loc
                g_i = i_loc
                g_k = 4 * b_idx + k_loc
                w_idx = g_k * n + g_i
                W[w_idx][col] += coeff_c
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
