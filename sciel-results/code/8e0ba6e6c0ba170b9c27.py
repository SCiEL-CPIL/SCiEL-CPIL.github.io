"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We use 6 Strassen products, each of rank 7. Total rank 42.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 6 block products
    # A is 2x4. Split into A1 (cols 0,1) and A2 (cols 2,3).
    # B is 4x6. Split into B1 (rows 0,1), B2 (rows 2,3).
    # Wait, B is 4x6.
    # A1 is 2x2, A2 is 2x2.
    # B is 4x6.
    # C1 = A1 B_top_left (2x2) + A2 B_bottom_left (2x2)
    # C2 = A1 B_top_mid (2x2) + A2 B_bottom_mid (2x2)
    # C3 = A1 B_top_right (2x2) + A2 B_bottom_right (2x2)
    
    # Indices for A blocks (flattened i*m + j)
    # A1: i in 0,1; j in 0,1
    # A2: i in 0,1; j in 2,3
    
    # Indices for B blocks (flattened j*p + k)
    # B_top_left: j in 0,1; k in 0,1
    # B_top_mid: j in 0,1; k in 2,3
    # B_top_right: j in 0,1; k in 4,5
    # B_bottom_left: j in 2,3; k in 0,1
    # B_bottom_mid: j in 2,3; k in 2,3
    # B_bottom_right: j in 2,3; k in 4,5
    
    # Indices for C blocks (flattened k*n + i)
    # C1: k in 0,1; i in 0,1
    # C2: k in 2,3; i in 0,1
    # C3: k in 4,5; i in 0,1

    # Helper to get indices for a 2x2 block
    def get_a_block(j_start):
        return [0*m + j_start + 0, 0*m + j_start + 1, 1*m + j_start + 0, 1*m + j_start + 1]
        # Order: 00, 01, 10, 11

    def get_b_block(j_start, k_start):
        return [j_start*p + k_start + 0, j_start*p + k_start + 1, 
                (j_start+1)*p + k_start + 0, (j_start+1)*p + k_start + 1]
        # Order: 00, 01, 10, 11

    def get_c_block(k_start):
        return [k_start*n + 0, k_start*n + 1, 
                (k_start+1)*n + 0, (k_start+1)*n + 1]
        # Order: 00, 01, 10, 11

    # Strassen coefficients
    # U (X), V (Y), W (Z)
    # M1: (X00+X11)(Y00+Y11) -> Z00+Z11-Z10+Z01? No.
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Let's define the linear forms for each M.
    # X indices: 0,1,2,3 (00, 01, 10, 11)
    # Y indices: 0,1,2,3
    # Z indices: 0,1,2,3
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1 contributes to Z00 and Z11
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2 contributes to Z10 and Z11
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3 contributes to Z01 and Z11
        # M4
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # M4 contributes to Z00 and Z10
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5 contributes to Z00 and Z01
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6 contributes to Z11
        # M7
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]), # M7 contributes to Z00
    ]
    
    # Verify M7 W coeff:
    # Z00 = M1 + M4 - M5 + M7
    # M7 is (X01-X10)(Y10+Y11).
    # My W coeff for M7 is [1,0,0,0] which is Z00. Correct.
    # M6 W coeff: Z11 = M1 - M2 + M3 + M6.
    # M6 is (X10-X00)(Y00+Y01).
    # My W coeff for M6 is [0,0,0,1] which is Z11. Correct.
    
    # The 6 products
    products = [
        # (a_block_j_start, b_block_j_start, b_block_k_start, c_block_k_start)
        (0, 0, 0, 0), # A1 B11 -> C1
        (2, 2, 0, 0), # A2 B21 -> C1
        (0, 0, 2, 2), # A1 B12 -> C2
        (2, 2, 2, 2), # A2 B22 -> C2
        (0, 0, 4, 4), # A1 B13 -> C3
        (2, 2, 4, 4), # A2 B23 -> C3
    ]
    
    col = 0
    for a_js, b_js, b_ks, c_ks in products:
        a_idx = get_a_block(a_js)
        b_idx = get_b_block(b_js, b_ks)
        c_idx = get_c_block(c_ks)
        
        for u_c, v_c, w_c in strassen_terms:
            for k in range(4):
                if u_c[k] != 0:
                    U[a_idx[k]][col] = u_c[k]
                if v_c[k] != 0:
                    V[b_idx[k]][col] = v_c[k]
                if w_c[k] != 0:
                    W[c_idx[k]][col] = w_c[k]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
