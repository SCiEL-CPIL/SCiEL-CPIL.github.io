import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 60 decomposition
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y10 - y11)
    # M4 = x11(y01 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # U coeffs: (i0, j0) -> coeff
    # V coeffs: (j0, k0) -> coeff
    # W coeffs: (i0, k0) -> coeff
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
    ]
    
    # 8 block products
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    # Fill U
                    for (i0, j0), coeff in u_coeffs.items():
                        u_row = (i1 * 2 + i0) * m + (j1 * 2 + j0)
                        U[u_row][col] += coeff
                    # Fill V
                    for (j0, k0), coeff in v_coeffs.items():
                        v_row = (j1 * 2 + j0) * p + (k1 * 2 + k0)
                        V[v_row][col] += coeff
                    # Fill W
                    for (i0, k0), coeff in w_coeffs.items():
                        w_row = (k1 * 2 + k0) * n + (i1 * 2 + i0)
                        W[w_row][col] += coeff
                    col += 1
                    
    # 5th column (k=4)
    for i in range(n):
        for j in range(m):
            # C[i, 4] += A[i, j] B[j, 4]
            u_row = i * m + j
            v_row = j * p + 4
            w_row = 4 * n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
