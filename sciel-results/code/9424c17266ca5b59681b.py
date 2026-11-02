DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Each product: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: {(i,j): coeff} for A[i,j]
    # V_coeffs: {(j,k_rel): coeff} for B[j,k_rel] (relative to block)
    # W_coeffs: {(k_rel,i): coeff} for C[i,k_rel] (relative to block)
    
    products = [
        # M1 = (a00+a11)*(b0+b3)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (a10+a11)*b0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = a00*(b1-b3)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = a11*(b2-b0)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (a00+a01)*b3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (a10-a00)*(b0+b1)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (a01-a11)*(b2+b3)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for t in range(3):
        for (u_coeffs, v_coeffs, w_coeffs) in products:
            for (i, j), coeff in u_coeffs.items():
                U[i * m + j][col] = coeff
            for (j, k_rel), coeff in v_coeffs.items():
                k_abs = 2 * t + k_rel
                V[j * p + k_abs][col] = coeff
            for (k_rel, i), coeff in w_coeffs.items():
                k_abs = 2 * t + k_rel
                W[k_abs * n + i][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
