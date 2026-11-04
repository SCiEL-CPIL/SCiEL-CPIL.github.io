DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Determine A coeffs
                    # X_s[i,j]
                    # Determine B coeffs
                    # Y_s[j,k]
                    # Determine C coeffs
                    # M_s[i,k] contributes to C blocks
                    
                    # We will add to U, V, W at column 'col'
                    
                    # A coefficients
                    # s=0: A11+A22 -> (0,0,1), (1,1,1)
                    # s=1: A21+A22 -> (1,0,1), (1,1,1)
                    # s=2: A11 -> (0,0,1)
                    # s=3: A22 -> (1,1,1)
                    # s=4: A11+A12 -> (0,0,1), (0,1,1)
                    # s=5: A21-A11 -> (1,0,1), (0,0,-1)
                    # s=6: A12-A22 -> (0,1,1), (1,1,-1)
                    
                    a_coeffs = []
                    if s == 0: a_coeffs = [(0,0,1), (1,1,1)]
                    elif s == 1: a_coeffs = [(1,0,1), (1,1,1)]
                    elif s == 2: a_coeffs = [(0,0,1)]
                    elif s == 3: a_coeffs = [(1,1,1)]
                    elif s == 4: a_coeffs = [(0,0,1), (0,1,1)]
                    elif s == 5: a_coeffs = [(1,0,1), (0,0,-1)]
                    elif s == 6: a_coeffs = [(0,1,1), (1,1,-1)]
                    
                    # B coefficients
                    # s=0: B11+B22 -> (0,0,1), (1,1,1)
                    # s=1: B11 -> (0,0,1)
                    # s=2: B12-B22 -> (0,1,1), (1,1,-1)
                    # s=3: B21-B11 -> (1,0,1), (0,0,-1)
                    # s=4: B22 -> (1,1,1)
                    # s=5: B11+B12 -> (0,0,1), (0,1,1)
                    # s=6: B21+B22 -> (1,0,1), (1,1,1)
                    
                    b_coeffs = []
                    if s == 0: b_coeffs = [(0,0,1), (1,1,1)]
                    elif s == 1: b_coeffs = [(0,0,1)]
                    elif s == 2: b_coeffs = [(0,1,1), (1,1,-1)]
                    elif s == 3: b_coeffs = [(1,0,1), (0,0,-1)]
                    elif s == 4: b_coeffs = [(1,1,1)]
                    elif s == 5: b_coeffs = [(0,0,1), (0,1,1)]
                    elif s == 6: b_coeffs = [(1,0,1), (1,1,1)]
                    
                    # C coefficients
                    # M_s[i,k]
                    # s=0 (M1): C11(+1), C22(+1)
                    # s=1 (M2): C21(+1), C22(-1)
                    # s=2 (M3): C12(+1), C22(+1)
                    # s=3 (M4): C11(+1), C21(+1)
                    # s=4 (M5): C11(-1), C12(+1)
                    # s=5 (M6): C22(+1)
                    # s=6 (M7): C11(+1)
                    
                    c_coeffs = []
                    if s == 0: c_coeffs = [(0,0,1), (1,1,1)]
                    elif s == 1: c_coeffs = [(1,0,1), (1,1,-1)]
                    elif s == 2: c_coeffs = [(0,1,1), (1,1,1)]
                    elif s == 3: c_coeffs = [(0,0,1), (1,0,1)]
                    elif s == 4: c_coeffs = [(0,0,-1), (0,1,1)]
                    elif s == 5: c_coeffs = [(1,1,1)]
                    elif s == 6: c_coeffs = [(0,0,1)]
                    
                    # Fill U
                    for I, J, coeff in a_coeffs:
                        u_idx = (I * 3 + i) * 6 + (J * 3 + j)
                        U[u_idx][col] += coeff
                        
                    # Fill V
                    for J, K, coeff in b_coeffs:
                        v_idx = (J * 3 + j) * 6 + (K * 3 + k)
                        V[v_idx][col] += coeff
                        
                    # Fill W
                    for I, K, coeff in c_coeffs:
                        w_idx = (I * 3 + i) * 6 + (K * 3 + k)
                        W[w_idx][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
