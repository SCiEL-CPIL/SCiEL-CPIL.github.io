import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 44
    # 7 for A11*B11 (Strassen)
    # 4 for A12*B21 (2x1 * 1x2) -> 2*1*2 = 4
    # 12 for A11*B12 (2x2 * 2x3) -> 2*2*3 = 12
    # 6 for A12*B22 (2x1 * 1x3) -> 2*1*3 = 6
    # 4 for A21*B11 (1x2 * 2x2) -> 1*2*2 = 4
    # 2 for A22*B21 (1x1 * 1x2) -> 1*1*2 = 2
    # 6 for A21*B12 (1x2 * 2x3) -> 1*2*3 = 6
    # 3 for A22*B22 (1x1 * 1x3) -> 1*1*3 = 3
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)] # 9x44
    V = [[0] * rank for _ in range(m * p)] # 15x44
    W = [[0] * rank for _ in range(p * n)] # 15x44
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        # u_vals is dict or list for specific indices
        # We need to map local indices to global indices
        
        # For now, we will just fill the matrices directly in the loops below
        pass

    # 1. A11 * B11 (Strassen, 7 terms)
    # Local indices: i in 0,1; j in 0,1; k in 0,1
    # Global U index: i*m + j = i*3 + j
    # Global V index: j*p + k = j*5 + k
    # Global W index: k*n + i = k*3 + i
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [u00, u01, u10, u11]
        # v: [v00, v01, v10, v11]
        # w: [w00, w01, w10, w11]
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)), # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)), # M2
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)), # M3
        ((0,0,0,1), (0,0,1,-1), (1,0,1,0)), # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)), # M5
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)), # M6
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0))  # M7
    ]
    
    for u_c, v_c, w_c in strassen_terms:
        # Fill U
        for li in range(2):
            for lj in range(2):
                idx = li * 3 + lj
                U[idx][col] = u_c[li*2 + lj]
        # Fill V
        for lj in range(2):
            for lk in range(2):
                idx = lj * 5 + lk
                V[idx][col] = v_c[lj*2 + lk]
        # Fill W
        for lk in range(2):
            for li in range(2):
                idx = lk * 3 + li
                W[idx][col] = w_c[lk*2 + li]
        col += 1

    # 2. A12 * B21 (2x1 * 1x2) -> 4 terms
    # A12: i in 0,1; j=2
    # B21: j=2; k in 0,1
    # C21: k in 0,1; i in 0,1
    # Term: A[i,2] * B[2,k] -> C[k,i]
    for i in range(2):
        for k in range(2):
            # U: i, 2
            U[i*3 + 2][col] = 1
            # V: 2, k
            V[2*5 + k][col] = 1
            # W: k, i
            W[k*3 + i][col] = 1
            col += 1

    # 3. A11 * B12 (2x2 * 2x3) -> 12 terms
    # A11: i in 0,1; j in 0,1
    # B12: j in 0,1; k in 0,1,2
    # C12: k in 0,1,2; i in 0,1
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*3 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1

    # 4. A12 * B22 (2x1 * 1x3) -> 6 terms
    # A12: i in 0,1; j=2
    # B22: j=2; k in 0,1,2
    # C22: k in 0,1,2; i in 0,1
    for i in range(2):
        for k in range(3):
            U[i*3 + 2][col] = 1
            V[2*5 + k][col] = 1
            W[k*3 + i][col] = 1
            col += 1

    # 5. A21 * B11 (1x2 * 2x2) -> 4 terms
    # A21: i=2; j in 0,1
    # B11: j in 0,1; k in 0,1
    # C21: k in 0,1; i=2
    for j in range(2):
        for k in range(2):
            U[2*3 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1

    # 6. A22 * B21 (1x1 * 1x2) -> 2 terms
    # A22: i=2; j=2
    # B21: j=2; k in 0,1
    # C21: k in 0,1; i=2
    for k in range(2):
        U[2*3 + 2][col] = 1
        V[2*5 + k][col] = 1
        W[k*3 + 2][col] = 1
        col += 1

    # 7. A21 * B12 (1x2 * 2x3) -> 6 terms
    # A21: i=2; j in 0,1
    # B12: j in 0,1; k in 0,1,2
    # C22: k in 0,1,2; i=2
    for j in range(2):
        for k in range(3):
            U[2*3 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1

    # 8. A22 * B22 (1x1 * 1x3) -> 3 terms
    # A22: i=2; j=2
    # B22: j=2; k in 0,1,2
    # C22: k in 0,1,2; i=2
    for k in range(3):
        U[2*3 + 2][col] = 1
        V[2*5 + k][col] = 1
        W[k*3 + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
