def construct():
    n, m, p = 2, 2, 5
    # Rank 18: 7 (Strassen) + 7 (Strassen) + 4 (Schoolbook)
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: Columns 0, 1 of B
    # Strassen terms
    # M1 = (a+d)(e+h)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1 # e=row 0, h=row 6 (1*5+1)
    W[0][col] = 1  # c00
    W[3][col] = 1  # c11
    W[2][col] = -1 # c01
    W[1][col] = 1  # c10
    col += 1
    
    # M2 = (c+d)e
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1][col] = 1  # c10
    W[3][col] = 1  # c11
    col += 1
    
    # M3 = a(f-h)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1 # f=row 1, h=row 6
    W[2][col] = 1  # c01
    W[3][col] = -1 # c11
    col += 1
    
    # M4 = d(g-e)
    U[3][col] = 1
    V[5][col] = 1; V[0][col] = -1 # g=row 5, e=row 0
    W[0][col] = -1 # c00
    W[1][col] = 1  # c10
    col += 1
    
    # M5 = (a+b)h
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = 1  # c00
    W[2][col] = 1  # c01
    col += 1
    
    # M6 = (c-a)(e+f)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1  # c11
    W[1][col] = -1 # c10
    col += 1
    
    # M7 = (b-d)(g+h)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = -1 # c00
    W[3][col] = -1 # c11
    col += 1
    
    # Block 2: Columns 2, 3 of B
    # Strassen terms
    # e=row 2, f=row 3, g=row 7, h=row 8
    # W offset: c02=4, c12=5, c03=6, c13=7
    
    # M1' = (a+d)(e'+h')
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1  # c02
    W[7][col] = 1  # c13
    W[6][col] = -1 # c03
    W[5][col] = 1  # c12
    col += 1
    
    # M2' = (c+d)e'
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[5][col] = 1  # c12
    W[7][col] = 1  # c13
    col += 1
    
    # M3' = a(f'-h')
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[6][col] = 1  # c03
    W[7][col] = -1 # c13
    col += 1
    
    # M4' = d(g'-e')
    U[3][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[4][col] = -1 # c02
    W[5][col] = 1  # c12
    col += 1
    
    # M5' = (a+b)h'
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = 1  # c02
    W[6][col] = 1  # c03
    col += 1
    
    # M6' = (c-a)(e'+f')
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1  # c13
    W[5][col] = -1 # c12
    col += 1
    
    # M7' = (b-d)(g'+h')
    U[1][col] = 1; U[3][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = -1 # c02
    W[7][col] = -1 # c13
    col += 1
    
    # Block 3: Column 4 of B
    # Schoolbook
    # c04 = a*b04 + b*b14
    # c14 = c*b04 + d*b14
    
    # Term 1: a * b04
    U[0][col] = 1
    V[4][col] = 1 # b04
    W[8][col] = 1 # c04
    col += 1
    
    # Term 2: b * b14
    U[1][col] = 1
    V[9][col] = 1 # b14
    W[8][col] = 1 # c04
    col += 1
    
    # Term 3: c * b04
    U[2][col] = 1
    V[4][col] = 1 # b04
    W[9][col] = 1 # c14
    col += 1
    
    # Term 4: d * b14
    U[3][col] = 1
    V[9][col] = 1 # b14
    W[9][col] = 1 # c14
    col += 1
    
    return {"U": U, "V": V, "W": W}
