# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x4 * 4x4 multiplication into 4 blocks of 2x2 * 2x2.
    # A is 2x4, split into A_L (2x2) and A_R (2x2).
    # B is 4x4, split into B_LL, B_RL (4x2 each? No, 2x2 blocks).
    # B = [[B_LL, B_LR], [B_RL, B_RR]] where each block is 2x2.
    # C = [[C_L, C_R]] where C_L = A_L B_LL + A_R B_RL (2x2)
    # and C_R = A_L B_LR + A_R B_RR (2x2).
    
    # Total 4 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with 7 scalar multiplications (Strassen).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # C = A * B
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 terms for a local 2x2 multiplication.
    # Each term is (U_part, V_part, W_part)
    # U_part, V_part, W_part are dicts mapping local index (0..3) to coeff.
    
    def strassen_terms():
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        return [
            # M1
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            # M2
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            # M3
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            # M4
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            # M5
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            # M6
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            # M7
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        ]

    terms = strassen_terms()
    
    # Block mapping
    # Block 0: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_L (rows 0,1, cols 0,1)
    # Block 1: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_L (rows 0,1, cols 0,1)
    # Block 2: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_R (rows 0,1, cols 2,3)
    # Block 3: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_R (rows 0,1, cols 2,3)
    
    # Global indices:
    # A (2x4): i in {0,1}, j in {0,1,2,3}. Index = i*4 + j
    # B (4x4): k in {0,1,2,3}, r in {0,1,2,3}. Index = k*4 + r
    # C (2x4): i in {0,1}, r in {0,1,2,3}. Index = r*2 + i (Trace convention: W[k*n+i, r] in task desc? 
    # Task says: W[k*n+i,r]. 
    # Current code uses: return r * 2 + i.
    # Let's check the task description: "output factor follows the official trace convention W[k*n+i,r]".
    # Usually W is (p*n, rank). p=4, n=2. Size 8.
    # Index in W: k*n+i? No, W is the output matrix factor.
    # The tensor is sum_s U_{i,k,s} V_{k,r,s} W_{r,i,s}? 
    # Or W_{r,i,s} where index is r*n+i?
    # The current code uses `return r * 2 + i` for W.
    # Let's stick to the current code's index mapping for W as it scored 1.14.
    
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # A index
                i = li
                if b == 0 or b == 2:
                    j = lj
                else:
                    j = 2 + lj
                g_idx = i * 4 + j
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # B index
                k = li
                r = lj
                if b == 0: # B_LL
                    g_k = k
                    g_r = r
                elif b == 1: # B_RL
                    g_k = 2 + k
                    g_r = r
                elif b == 2: # B_LR
                    g_k = k
                    g_r = 2 + r
                else: # B_RR
                    g_k = 2 + k
                    g_r = 2 + r
                g_idx = g_k * 4 + g_r
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # C index
                i = li
                r = lj
                if b == 0 or b == 1:
                    g_r = r
                else:
                    g_r = 2 + r
                # Current code: return r * 2 + i
                g_idx = g_r * 2 + i
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
