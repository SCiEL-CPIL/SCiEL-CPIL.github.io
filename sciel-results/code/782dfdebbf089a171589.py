import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # X * Y = Z
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y10 - Y11)
    # M4 = X11(Y01 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y10)
    # M7 = (X01 - X11)(Y11 + Y01)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Define the coefficients for U, V, W for each of the 7 factors
    # u[a, b] is coefficient for X_ab
    # v[c, d] is coefficient for Y_cd
    # w[e, f] is coefficient for Z_ef
    
    strassen_factors = [
        # (u, v, w)
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,1): 1, (0,1): 1}, {(0,0): 1, (1,1): 1})
    ]
    
    # Block mappings
    # A blocks: 0,0 and 0,1
    # A_00: i in {0,1}, j in {0,1} -> local a in {0,1}, b in {0,1}
    # A_01: i in {0,1}, j in {2,3} -> local a in {0,1}, b in {0,1}
    
    # B blocks: 0,0; 1,0; 0,1; 1,1; 0,2; 1,2
    # B_00: j in {0,1}, k in {0,1} -> local c in {0,1}, d in {0,1}
    # B_10: j in {2,3}, k in {0,1} -> local c in {0,1}, d in {0,1}
    # B_01: j in {0,1}, k in {2,3} -> local c in {0,1}, d in {0,1}
    # B_11: j in {2,3}, k in {2,3} -> local c in {0,1}, d in {0,1}
    # B_02: j in {0,1}, k in {4,5} -> local c in {0,1}, d in {0,1}
    # B_12: j in {2,3}, k in {4,5} -> local c in {0,1}, d in {0,1}
    
    # C blocks: 0,0; 0,1; 0,2
    # C_00: i in {0,1}, k in {0,1} -> local e in {0,1}, f in {0,1}
    # C_01: i in {0,1}, k in {2,3} -> local e in {0,1}, f in {0,1}
    # C_02: i in {0,1}, k in {4,5} -> local e in {0,1}, f in {0,1}
    
    factor_idx = 0
    for col in range(3):
        # Product 1: A_00 * B_0,col
        # A_00 global offset: i=0, j=0
        # B_0,col global offset: j=0, k=col*2
        for u, v, w in strassen_factors:
            for (a, b), ua in u.items():
                i_global = a
                j_global = b
                U[i_global * m + j_global][factor_idx] += ua
            for (c, d), vb in v.items():
                j_global = c
                k_global = d + col * 2
                V[j_global * p + k_global][factor_idx] += vb
            for (e, f), wc in w.items():
                i_global = e
                k_global = f + col * 2
                W[k_global * n + i_global][factor_idx] += wc
            factor_idx += 1
            
        # Product 2: A_01 * B_1,col
        # A_01 global offset: i=0, j=2
        # B_1,col global offset: j=2, k=col*2
        for u, v, w in strassen_factors:
            for (a, b), ua in u.items():
                i_global = a
                j_global = b + 2
                U[i_global * m + j_global][factor_idx] += ua
            for (c, d), vb in v.items():
                j_global = c + 2
                k_global = d + col * 2
                V[j_global * p + k_global][factor_idx] += vb
            for (e, f), wc in w.items():
                i_global = e
                k_global = f + col * 2
                W[k_global * n + i_global][factor_idx] += wc
            factor_idx += 1
            
    return {"U": U, "V": V, "W": W}
