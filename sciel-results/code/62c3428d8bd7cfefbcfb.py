import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 4 * 7 + 4 * 4  # 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms: each is (u_list, v_list, w_list)
    # u_list: [(i, j, coeff), ...]
    # v_list: [(j, k, coeff), ...]
    # w_list: [(k, i, coeff), ...]
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([((0,0,1),(1,1,1)), ((0,0,1),(1,1,1)), ((0,0,1),(1,1,1))]),
        # M2 = (A10+A11)(B00)
        ([((1,0,1),(1,1,1)), ((0,0,1)), ((1,0,1),(1,1,-1))]),
        # M3 = A00(B01-B11)
        ([((0,0,1)), ((0,1,1),(1,1,-1)), ((0,1,1),(1,1,1))]),
        # M4 = A11(B10-B00)
        ([((1,1,1)), ((1,0,1),(0,0,-1)), ((0,0,1),(1,0,1))]),
        # M5 = (A00+A01)(B11)
        ([((0,0,1),(0,1,1)), ((1,1,1)), ((0,0,-1),(0,1,1))]),
        # M6 = (A10-A00)(B00+B01)
        ([((1,0,1),(0,0,-1)), ((0,0,1),(0,1,1)), ((1,1,1))]),
        # M7 = (A01-A11)(B10+B11)
        ([((0,1,1),(1,1,-1)), ((1,0,1),(1,1,1)), ((0,0,1))]),
    ]
    
    for k0 in range(0, p, 2):
        i0 = 0
        j0 = 0
        
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            for (li, lj, coeff) in u_coeffs:
                gi = i0 + li
                gj = j0 + lj
                U[gi * m + gj][col] += coeff
            
            for (lj, lk, coeff) in v_coeffs:
                gj = j0 + lj
                gk = k0 + lk
                V[gj * p + gk][col] += coeff
            
            for (lk, li, coeff) in w_coeffs:
                gk = k0 + lk
                gi = i0 + li
                W[gk * n + gi][col] += coeff
            
            col += 1
    
    for k0 in range(0, p, 2):
        i0 = 0
        j0 = 2
        
        for li in range(2):
            for lk in range(2):
                gi = i0 + li
                gj = j0
                gk = k0 + lk
                U[gi * m + gj][col] = 1
                V[gj * p + gk][col] = 1
                W[gk * n + gi][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
