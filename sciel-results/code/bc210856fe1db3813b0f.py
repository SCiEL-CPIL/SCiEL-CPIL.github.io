def construct():
    n, m, p = 2, 2, 12
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for Strassen products
    # u: [A00, A01, A10, A11]
    # v: [B00, B01, B10, B11]
    # w: [C00, C10, C01, C11] -> Note: Index k*n+i
    # C00: k=0, i=0 -> 0
    # C10: k=0, i=1 -> 1
    # C01: k=1, i=0 -> 2
    # C11: k=1, i=1 -> 3
    
    strassen_coeffs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    r = 0
    for b in range(6):
        for s in range(7):
            u, v, w = strassen_coeffs[s]
            for i_loc in range(2):
                for j_loc in range(2):
                    val = u[i_loc * 2 + j_loc]
                    if val != 0:
                        U[i_loc * m + j_loc][r] = val
                        
            for j_loc in range(2):
                for k_loc in range(2):
                    val = v[j_loc * 2 + k_loc]
                    if val != 0:
                        # Global k = 2*b + k_loc
                        V[j_loc * p + (2 * b + k_loc)][r] = val
                        
            for k_loc in range(2):
                for i_loc in range(2):
                    val = w[k_loc * 2 + i_loc]
                    if val != 0:
                        # Global k = 2*b + k_loc
                        W[(2 * b + k_loc) * n + i_loc][r] = val
                        
            r += 1
            
    return {"U": U, "V": V, "W": W}
