def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    # Rank 49
    rank = 49
    dim_u = n * m # 4
    dim_v = m * p # 26
    dim_w = p * n # 26
    
    U = [[0.0] * rank for _ in range(dim_u)]
    V = [[0.0] * rank for _ in range(dim_v)]
    W = [[0.0] * rank for _ in range(dim_w)]
    
    # Strassen coefficients for A (U)
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 0k = k, 1k = 13+k
    # C indices: ik = 2k+i
    
    for j in range(7):
        c0 = 2 * j
        c1 = 2 * j + 1
        
        for t in range(7):
            col = j * 7 + t
            
            if t == 0:
                # m1
                U[0][col] += 1.0
                U[3][col] += 1.0
                if c0 < p: V[c0][col] += 1.0
                if c1 < p: V[13 + c1][col] += 1.0
                if c0 < p: W[2 * c0][col] += 1.0
                if c1 < p: W[2 * c1 + 1][col] += 1.0
                
            elif t == 1:
                # m2
                U[2][col] += 1.0
                U[3][col] += 1.0
                if c0 < p: V[c0][col] += 1.0
                if c0 < p: W[2 * c0 + 1][col] += 1.0
                if c1 < p: W[2 * c1 + 1][col] -= 1.0
                
            elif t == 2:
                # m3
                U[0][col] += 1.0
                if c1 < p: V[c1][col] += 1.0
                if c1 < p: V[13 + c1][col] -= 1.0
                if c0 < p: W[2 * c0 + 1][col] += 1.0
                
            elif t == 3:
                # m4
                U[3][col] += 1.0
                if c0 < p: V[13 + c0][col] += 1.0
                if c0 < p: V[c0][col] -= 1.0
                if c1 < p: W[2 * c1][col] += 1.0
                
            elif t == 4:
                # m5
                U[0][col] += 1.0
                U[1][col] += 1.0
                if c1 < p: V[13 + c1][col] += 1.0
                if c0 < p: W[2 * c0][col] -= 1.0
                if c1 < p: W[2 * c1][col] += 1.0
                
            elif t == 5:
                # m6
                U[2][col] += 1.0
                U[0][col] -= 1.0
                if c0 < p: V[c0][col] += 1.0
                if c1 < p: V[c1][col] += 1.0
                if c1 < p: W[2 * c1 + 1][col] += 1.0
                
            elif t == 6:
                # m7
                U[1][col] += 1.0
                U[3][col] -= 1.0
                if c0 < p: V[13 + c0][col] += 1.0
                if c1 < p: V[13 + c1][col] += 1.0
                if c0 < p: W[2 * c0][col] += 1.0

    return {"U": U, "V": V, "W": W}
