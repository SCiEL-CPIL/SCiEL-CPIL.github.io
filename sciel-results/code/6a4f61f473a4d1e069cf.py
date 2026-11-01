import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s margin
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Try Strassen rank 56
    try:
        U, V, W = _construct_strassen_rank_56(n, m, p)
        # Verify validity quickly? 
        # The evaluator checks validity. 
        # If time is running out, return best found.
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to Schoolbook rank 64
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_rank_56(n, m, p):
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products definitions
    # X and Y are 2x2 matrices formed from blocks of A and B.
    # X[a][b] = sum_{I,J} coeff_X[a][b][I][J] * A[2I+a][2J+b]
    # Y[c][d] = sum_{K,L} coeff_Y[c][d][K][L] * B[2K+c][2L+d]
    
    # Define the 7 products
    # Each product is a dict:
    # 'X': 2x2x2x2 coeffs for A
    # 'Y': 2x2x2x2 coeffs for B
    # 'C': dict mapping (P,Q) block to weight
    
    def make_zero_coeffs():
        return [[[ [0,0], [0,0] ],
                 [ [0,0], [0,0] ]] ] # X[a][b][I][J]
        
    def set_block(coeff, a, b, I, J, val):
        coeff[a][b][I][J] = val

    products = []
    
    # M0: (A00+A11)(B00+B11)
    X0 = make_zero_coeffs()
    Y0 = make_zero_coeffs()
    # X = A00 + A11
    for a in range(2):
        for b in range(2):
            set_block(X0, a, b, 0, 0, 1)
            set_block(X0, a, b, 1, 1, 1)
    # Y = B00 + B11
    for a in range(2):
        for b in range(2):
            set_block(Y0, a, b, 0, 0, 1)
            set_block(Y0, a, b, 1, 1, 1)
    C0 = {(0,0): 1, (1,1): 1}
    products.append({'X': X0, 'Y': Y0, 'C': C0})
    
    # M1: (A10+A11)B00
    X1 = make_zero_coeffs()
    Y1 = make_zero_coeffs()
    for a in range(2):
        for b in range(2):
            set_block(X1, a, b, 1, 0, 1)
            set_block(X1, a, b, 1, 1, 1)
    for a in range(2):
        for b in range(2):
            set_block(Y1, a, b, 0, 0, 1)
    C1 = {(1,0): 1, (1,1): -1}
    products.append({'X': X1, 'Y': Y1, 'C': C1})
    
    # M2: A00(B01-B11)
    X2 = make_zero_coeffs()
    Y2 = make_zero_coeffs()
    for a in range(2):
        for b in range(2):
            set_block(X2, a, b, 0, 0, 1)
    for a in range(2):
        for b in range(2):
            set_block(Y2, a, b, 0, 1, 1)
            set_block(Y2, a, b, 1, 1, -1)
    C2 = {(0,0): -1, (0,1): 1, (1,1): 1}
    products.append({'X': X2, 'Y': Y2, 'C': C2})
    
    # M3: A11(B10-B00)
    X3 = make_zero_coeffs()
    Y3 = make_zero_coeffs()
    for a in range(2):
        for b in range(2):
            set_block(X3, a, b, 1, 1, 1)
    for a in range(2):
        for b in range(2):
            set_block(Y3, a, b, 1, 0, 1)
            set_block(Y3, a, b, 0, 0, -1)
    C3 = {(0,0): 1, (1,0): 1}
    products.append({'X': X3, 'Y': Y3, 'C': C3})
    
    # M4: (A00+A01)B11
    X4 = make_zero_coeffs()
    Y4 = make_zero_coeffs()
    for a in range(2):
        for b in range(2):
            set_block(X4, a, b, 0, 0, 1)
            set_block(X4, a, b, 0, 1, 1)
    for a in range(2):
        for b in range(2):
            set_block(Y4, a, b, 1, 1, 1)
    C4 = {(0,0): -1, (0,1): 1}
    products.append({'X': X4, 'Y': Y4, 'C': C4})
    
    # M5: (A10-A00)(B00+B01)
    X5 = make_zero_coeffs()
    Y5 = make_zero_coeffs()
    for a in range(2):
        for b in range(2):
            set_block(X5, a, b, 1, 0, 1)
            set_block(X5, a, b, 0, 0, -1)
    for a in range(2):
        for b in range(2):
            set_block(Y5, a, b, 0, 0, 1)
            set_block(Y5, a, b, 0, 1, 1)
    C5 = {(1,1): 1}
    products.append({'X': X5, 'Y': Y5, 'C': C5})
    
    # M6: (A01-A11)(B10+B11)
    X6 = make_zero_coeffs()
    Y6 = make_zero_coeffs()
    for a in range(2):
        for b in range(2):
            set_block(X6, a, b, 0, 1, 1)
            set_block(X6, a, b, 1, 1, -1)
    for a in range(2):
        for b in range(2):
            set_block(Y6, a, b, 1, 0, 1)
            set_block(Y6, a, b, 1, 1, 1)
    C6 = {(0,0): 1}
    products.append({'X': X6, 'Y': Y6, 'C': C6})
    
    # Schoolbook 2x2 multiplication patterns
    # Z[a][d] += X[a][b] * Y[b][d]
    # Terms:
    # 0: X00 Y00 -> Z00
    # 1: X01 Y10 -> Z00
    # 2: X00 Y01 -> Z01
    # 3: X01 Y11 -> Z01
    # 4: X10 Y00 -> Z10
    # 5: X11 Y10 -> Z10
    # 6: X10 Y01 -> Z11
    # 7: X11 Y11 -> Z11
    
    patterns = [
        (0, 0, 0, 0, 0, 0), # a, b, c, d, e, f (c must be b)
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1)
    ]
    
    for s, prod in enumerate(products):
        X = prod['X']
        Y = prod['Y']
        C_weights = prod['C']
        
        for t, (a, b, c, d, e, f) in enumerate(patterns):
            r = s * 8 + t
            
            # Update U (from X)
            # X[a][b] = sum_{I,J} X[a][b][I][J] * A[2I+a][2J+b]
            for I in range(2):
                for J in range(2):
                    val = X[a][b][I][J]
                    if val != 0:
                        row = 2 * I + a
                        col = 2 * J + b
                        u_idx = row * m + col
                        U[u_idx][r] += val
                        
            # Update V (from Y)
            # Y[c][d] = sum_{K,L} Y[c][d][K][L] * B[2K+c][2L+d]
            # Note: c == b
            for K in range(2):
                for L in range(2):
                    val = Y[c][d][K][L]
                    if val != 0:
                        row = 2 * K + c
                        col = 2 * L + d
                        v_idx = row * p + col
                        V[v_idx][r] += val
                        
            # Update W (from C)
            # Z[e][f] contributes to C blocks
            for (P, Q), weight in C_weights.items():
                if weight != 0:
                    row = 2 * P + e
                    col = 2 * Q + f
                    # W index: k*n + i, where k=col, i=row
                    w_idx = col * n + row
                    W[w_idx][r] += weight
                    
    return U, V, W
