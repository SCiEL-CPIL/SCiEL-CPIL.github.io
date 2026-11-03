import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 6x6 using 2x2 blocks of 3x3
    # Rank = 7 (Strassen mults) * 27 (schoolbook 3x3) = 189
    rank = 189
    
    # Initialize U, V, W as lists of lists (or numpy arrays)
    # Using lists for compatibility with potential integer checks, though numpy is fine
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # t: 1 to 7
    for t in range(7):
        # local indices a, b, c for 3x3 multiplication
        # a: row of X, b: col of Y, c: inner
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # Determine coefficients for X_t and Y_t based on Strassen formula
                    # X_t and Y_t are 2x2 block combinations
                    # We need alpha_t[a_b][k_b] and beta_t[k_b][b_b]
                    
                    # Strassen M1: (A00+A11)(B00+B11)
                    # Strassen M2: (A10+A11)B00
                    # Strassen M3: A00(B01-B11)
                    # Strassen M4: A11(B10-B00)
                    # Strassen M5: (A00+A01)B11
                    # Strassen M6: (A10-A00)(B00+B01)
                    # Strassen M7: (A01-A11)(B10+B11)
                    
                    # Define alpha (for A blocks) and beta (for B blocks)
                    # Indices a_b, k_b for A block row, col
                    # Indices k_b, b_b for B block row, col
                    
                    alphas = {}
                    betas = {}
                    
                    if t == 0: # M1
                        alphas[(0,0)] = 1; alphas[(1,1)] = 1
                        betas[(0,0)] = 1; betas[(1,1)] = 1
                    elif t == 1: # M2
                        alphas[(1,0)] = 1; alphas[(1,1)] = 1
                        betas[(0,0)] = 1
                    elif t == 2: # M3
                        alphas[(0,0)] = 1
                        betas[(0,1)] = 1; betas[(1,1)] = -1
                    elif t == 3: # M4
                        alphas[(1,1)] = 1
                        betas[(1,0)] = 1; betas[(0,0)] = -1
                    elif t == 4: # M5
                        alphas[(0,0)] = 1; alphas[(0,1)] = 1
                        betas[(1,1)] = 1
                    elif t == 5: # M6
                        alphas[(1,0)] = 1; alphas[(0,0)] = -1
                        betas[(0,0)] = 1; betas[(0,1)] = 1
                    elif t == 6: # M7
                        alphas[(0,1)] = 1; alphas[(1,1)] = -1
                        betas[(1,0)] = 1; betas[(1,1)] = 1
                    
                    # Coefficients for output C blocks
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # coeff[i_b][j_b]
                    coeffs = [[0]*2 for _ in range(2)]
                    if t == 0: # M1
                        coeffs[0][0] = 1; coeffs[1][1] = 1
                    elif t == 1: # M2
                        coeffs[1][0] = 1; coeffs[1][1] = -1
                    elif t == 2: # M3
                        coeffs[0][1] = 1; coeffs[1][1] = 1
                    elif t == 3: # M4
                        coeffs[0][0] = 1; coeffs[1][0] = 1
                    elif t == 4: # M5
                        coeffs[0][0] = -1; coeffs[0][1] = 1
                    elif t == 5: # M6
                        coeffs[1][1] = 1
                    elif t == 6: # M7
                        coeffs[0][0] = 1
                    
                    # Fill U, V, W
                    for a_b in range(2):
                        for k_b in range(2):
                            # U index: I_A * m + J_A
                            # I_A = 2*a_b + a
                            # J_A = 2*k_b + c
                            I_A = 2 * a_b + a
                            J_A = 2 * k_b + c
                            u_val = alphas.get((a_b, k_b), 0)
                            if u_val != 0:
                                U[I_A * m + J_A][col] = u_val
                            
                            # V index: J_B * p + K_B
                            # J_B = 2*k_b + c
                            # K_B = 2*b_b + b  (Note: b_b varies in outer loop? No, b_b is block index)
                            # Wait, V depends on k_b and b_b.
                            # We need to loop b_b as well for V and W?
                            # Yes, V and W have different block indices.
                            pass # Handled in nested loops below

                    # Correct Loop Structure for V and W
                    # V involves k_b and b_b
                    # W involves b_b and a_b
                    
                    # We already looped a_b and k_b for U.
                    # Let's restructure to iterate over all block combinations for each matrix.
                    
                    # Actually, for a fixed r, U has entries for all (a_b, k_b) in alphas.
                    # V has entries for all (k_b, b_b) in betas.
                    # W has entries for all (b_b, a_b) in coeffs.
                    
                    # U
                    for (ab, kb), val in alphas.items():
                        I_A = 2 * ab + a
                        J_A = 2 * kb + c
                        U[I_A * m + J_A][col] += val

                    # V
                    for (kb, bb), val in betas.items():
                        J_B = 2 * kb + c
                        K_B = 2 * bb + b
                        V[J_B * p + K_B][col] += val

                    # W
                    # W index: K_C * n + I_C
                    # K_C = 2 * bb + b
                    # I_C = 2 * ab + a
                    # Coeff is coeffs[ab][bb]
                    for ab in range(2):
                        for bb in range(2):
                            val = coeffs[ab][bb]
                            if val != 0:
                                K_C = 2 * bb + b
                                I_C = 2 * ab + a
                                W[K_C * n + I_C][col] += val

                    col += 1

    return {"U": U, "V": V, "W": W}
