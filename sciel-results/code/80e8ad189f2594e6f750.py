def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0]*rank for _ in range(n*m)]  # 8 x 56
    V = [[0]*rank for _ in range(m*p)]  # 32 x 56
    W = [[0]*rank for _ in range(p*n)]  # 16 x 56
    
    # Strassen coefficients for 2x2 * 2x2
    # For each term t, define:
    # x_coeffs: dict mapping (i', j') to coefficient
    # y_coeffs: dict mapping (j', k') to coefficient
    # z_coeffs: dict mapping (i', k') to coefficient
    
    strassen_terms = [
        # t=0: (x00+x11)(y00+y11)
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        # t=1: (x10+x11)y00
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        # t=2: x00(y01-y11)
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        # t=3: x11(y10-y00)
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        # t=4: (x00+x01)y11
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        # t=5: (x10-x00)(y00+y01)
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        # t=6: (x01-x11)(y10+y11)
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    col = 0
    for a in range(2):  # A block column: 2a, 2a+1
        for b in range(4):  # B column block: 2b, 2b+1
            for t in range(7):
                term = strassen_terms[t]
                # X part: A[i', 2a+j']
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx][col] += coeff
                
                # Y part: B[2a+j', 2b+k']
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx][col] += coeff
                
                # Z part: C[i', 2b+k'] -> W[2b+k', i']
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
