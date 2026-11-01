def construct():
    n, m, p = 2, 2, 12
    rank = 42
    U = [[0]*rank for _ in range(n*m)]  # 4 x 42
    V = [[0]*rank for _ in range(m*p)]  # 24 x 42
    W = [[0]*rank for _ in range(p*n)]  # 24 x 42
    
    num_pairs = p // 2  # 6
    # Strassen products for 2x2 x 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        # M1
        U[0][col] = 1; U[3][col] = 1
        V[0][col] = 1; V[3][col] = 1
        W[0][col] = 1; W[11][col] = 1  # C00, C11
        col += 1
        # M2
        U[2][col] = 1; U[3][col] = 1
        V[0][col] = 1
        W[10][col] = 1; W[11][col] = -1  # C10, C11
        col += 1
        # M3
        U[0][col] = 1
        V[1][col] = 1; V[3][col] = -1
        W[1][col] = 1; W[11][col] = 1  # C01, C11
        col += 1
        # M4
        U[3][col] = 1
        V[2][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[10][col] = 1  # C00, C10
        col += 1
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[3][col] = 1
        W[0][col] = -1; W[1][col] = 1  # C00, C01
        col += 1
        # M6
        U[2][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[11][col] = 1  # C11
        col += 1
        # M7
        U[1][col] = 1; U[3][col] = -1
        V[2][col] = 1; V[3][col] = 1
        W[0][col] = 1  # C00
        col += 1
    
    return {"U": U, "V": V, "W": W}
