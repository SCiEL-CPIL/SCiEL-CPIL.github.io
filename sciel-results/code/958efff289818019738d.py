import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # Indices: 0:00, 1:01, 2:10, 3:11
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y01-y11)
    # P4 = x11(y10-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    
    # Coefficients for U, V, W for each term
    # U: A factors, V: B factors, W: C factors
    strassen_terms = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Sub-products:
    # 1. A_L (cols 0,1) * B_TL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # 2. A_R (cols 2,3) * B_BL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # 3. A_L (cols 0,1) * B_TR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # 4. A_R (cols 2,3) * B_BR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    
    # Mappings:
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*4 + k
    # W index: k*n + i = k*2 + i
    
    sub_products = [
        # 1. A_L * B_TL -> C_L
        {
            'a_off_r': 0, 'a_off_c': 0, # A rows 0-1, cols 0-1
            'b_off_r': 0, 'b_off_c': 0, # B rows 0-1, cols 0-1
            'c_off_r': 0, 'c_off_c': 0, # C rows 0-1, cols 0-1 (W index k*2+i, so col offset 0)
        },
        # 2. A_R * B_BL -> C_L
        {
            'a_off_r': 0, 'a_off_c': 2, # A rows 0-1, cols 2-3
            'b_off_r': 2, 'b_off_c': 0, # B rows 2-3, cols 0-1
            'c_off_r': 0, 'c_off_c': 0, # C rows 0-1, cols 0-1
        },
        # 3. A_L * B_TR -> C_R
        {
            'a_off_r': 0, 'a_off_c': 0, # A rows 0-1, cols 0-1
            'b_off_r': 0, 'b_off_c': 2, # B rows 0-1, cols 2-3
            'c_off_r': 0, 'c_off_c': 2, # C rows 0-1, cols 2-3
        },
        # 4. A_R * B_BR -> C_R
        {
            'a_off_r': 0, 'a_off_c': 2, # A rows 0-1, cols 2-3
            'b_off_r': 2, 'b_off_c': 2, # B rows 2-3, cols 2-3
            'c_off_r': 0, 'c_off_c': 2, # C rows 0-1, cols 2-3
        },
    ]
    
    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = (sp['a_off_r'] + r) * 4 + (sp['a_off_c'] + c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = (sp['b_off_r'] + r) * 4 + (sp['b_off_c'] + c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                # W index is k*n + i. Here r is i (row of C), c is k (col of C)?
                # In C = A*B, C[i,k] = sum A[i,j] B[j,k].
                # Local C index: 0:00, 1:01, 2:10, 3:11.
                # loc_w // 2 is i (row), loc_w % 2 is k (col).
                global_w = (sp['c_off_r'] + r) * 2 + (sp['c_off_c'] + c)
                W[global_w][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
