"""Program for 2x2 by 2x10 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # We decompose the 2x10 B matrix into 5 blocks of 2x2.
    # Each block is multiplied by the 2x2 A matrix using Strassen's algorithm (rank 7).
    # Total rank = 5 * 7 = 35.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen intermediates for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: A00:0, A01:1, A10:2, A11:3
    # V indices for block g (cols 2g, 2g+1):
    # B00: 0*10 + 2g = 2g
    # B01: 0*10 + 2g+1 = 2g+1
    # B10: 1*10 + 2g = 10+2g
    # B11: 1*10 + 2g+1 = 10+2g+1
    #
    # W indices for block g:
    # C00: (2g)*2 + 0 = 4g
    # C01: (2g+1)*2 + 0 = 4g+2
    # C10: (2g)*2 + 1 = 4g+1
    # C11: (2g+1)*2 + 1 = 4g+3
    
    # Define the coefficients for each Strassen step
    # Each entry is a list of (index, coeff)
    U_coeffs = [
        [(0, 1), (3, 1)],       # M1: A00 + A11
        [(2, 1), (3, 1)],       # M2: A10 + A11
        [(0, 1)],               # M3: A00
        [(3, 1)],               # M4: A11
        [(0, 1), (1, 1)],       # M5: A00 + A01
        [(2, 1), (0, -1)],      # M6: A10 - A00
        [(1, 1), (3, -1)]       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 1), (3, 1)],       # M1: B00 + B11 (relative to block)
        [(0, 1)],               # M2: B00
        [(2, 1), (3, -1)],      # M3: B10 - B11
        [(1, 1), (0, -1)],      # M4: B01 - B00
        [(3, 1)],               # M5: B11
        [(0, 1), (1, 1)],       # M6: B00 + B01
        [(2, 1), (3, 1)]        # M7: B10 + B11
    ]
    
    # W contributions: list of (w_offset, coeff)
    # w_offset is relative to 4g
    W_contribs = [
        [(0, 1), (3, 1)],       # M1 -> C00(+1), C11(+1)
        [(1, 1), (3, -1)],      # M2 -> C10(+1), C11(-1)
        [(2, 1), (3, 1)],       # M3 -> C01(+1), C11(+1)
        [(0, 1), (1, 1)],       # M4 -> C00(+1), C10(+1)
        [(0, -1), (2, 1)],      # M5 -> C00(-1), C01(+1)
        [(3, 1)],               # M6 -> C11(+1)
        [(0, 1)]                # M7 -> C00(+1)
    ]
    
    for g in range(5):
        base_r = g * 7
        base_v = 2 * g
        base_w = 4 * g
        
        for s in range(7):
            r = base_r + s
            
            # Fill U
            for idx, coeff in U_coeffs[s]:
                U[idx][r] = coeff
                
            # Fill V
            for idx, coeff in V_coeffs[s]:
                # Map relative block indices to global V indices
                # 0->B00, 1->B01, 2->B10, 3->B11
                if idx == 0:
                    v_idx = base_v
                elif idx == 1:
                    v_idx = base_v + 1
                elif idx == 2:
                    v_idx = 10 + base_v
                elif idx == 3:
                    v_idx = 10 + base_v + 1
                V[v_idx][r] = coeff
                
            # Fill W
            for w_off, coeff in W_contribs[s]:
                w_idx = base_w + w_off
                W[w_idx][r] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
