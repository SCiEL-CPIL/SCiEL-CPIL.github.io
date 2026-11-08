"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 - M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 - M6
    
    # W index order: C00, C10, C01, C11
    # Indices: 0, 1, 2, 3
    
    col = 0
    for b in range(4):  # 4 blocks of k (size 2)
        for part in range(2):  # part 1: j in {0,1}, part 2: j in {2,3}
            for s in range(7):  # 7 Strassen terms
                j_offset = part * 2
                
                u_coeffs = [0, 0, 0, 0]
                v_coeffs = [0, 0, 0, 0]
                w_coeffs = [0, 0, 0, 0]
                
                if s == 0: # M1
                    u_coeffs = [1, 0, 0, 1]
                    v_coeffs = [1, 0, 0, 1]
                    w_coeffs = [1, 0, 0, 1]
                elif s == 1: # M2
                    u_coeffs = [0, 0, 1, 1]
                    v_coeffs = [0, 1, 0, 0]
                    w_coeffs = [0, 1, 0, -1]
                elif s == 2: # M3
                    u_coeffs = [1, 0, 0, 0]
                    v_coeffs = [0, 0, 1, -1]
                    w_coeffs = [0, 0, 1, 1]
                elif s == 3: # M4
                    u_coeffs = [0, 0, 0, 1]
                    v_coeffs = [1, 0, -1, 0]
                    w_coeffs = [-1, 1, 0, 0]
                elif s == 4: # M5
                    u_coeffs = [1, 1, 0, 0]
                    v_coeffs = [0, 0, 0, 1]
                    w_coeffs = [-1, 0, 1, 0]
                elif s == 5: # M6
                    u_coeffs = [-1, 0, 1, 0]
                    v_coeffs = [1, 0, 1, 0]
                    w_coeffs = [0, 0, 0, -1]
                elif s == 6: # M7
                    u_coeffs = [0, 1, 0, -1]
                    v_coeffs = [0, 1, 0, 1]
                    w_coeffs = [1, 0, 0, 0]

                for li in range(2):
                    for lj in range(2):
                        val_u = u_coeffs[li * 2 + lj]
                        if val_u != 0:
                            gi = li
                            gj = j_offset + lj
                            U[gi * m + gj][col] = val_u
                            
                for lj in range(2):
                    for lk in range(2):
                        val_v = v_coeffs[lj * 2 + lk]
                        if val_v != 0:
                            gj = j_offset + lj
                            gk = 2 * b + lk
                            V[gj * p + gk][col] = val_v
                            
                for lk in range(2):
                    for li in range(2):
                        val_w = w_coeffs[lk * 2 + li]
                        if val_w != 0:
                            gk = 2 * b + lk
                            gi = li
                            W[gk * n + gi][col] = val_w
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
