def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0]*rank for _ in range(n*m)]  # 6 x 55
    V = [[0]*rank for _ in range(m*p)]  # 30 x 55
    W = [[0]*rank for _ in range(p*n)]  # 20 x 55
    
    col = 0
    # C1 part: 5 blocks, 7 Strassen terms each
    for b in range(5):
        # B indices for this block
        B00 = 0*10 + 2*b      # B[0, 2b]
        B01 = 0*10 + (2*b+1)  # B[0, 2b+1]
        B10 = 1*10 + 2*b      # B[1, 2b]
        B11 = 1*10 + (2*b+1)  # B[1, 2b+1]
        # C indices for this block
        C00 = (2*b)*2 + 0    # C[0, 2b]
        C01 = (2*b+1)*2 + 0  # C[0, 2b+1]
        C10 = (2*b)*2 + 1    # C[1, 2b]
        C11 = (2*b+1)*2 + 1  # C[1, 2b+1]
        # A indices
        A00 = 0*3+0  # A[0,0]
        A01 = 0*3+1  # A[0,1]
        A10 = 1*3+0  # A[1,0]
        A11 = 1*3+1  # A[1,1]
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[A00][col] += 1
        V[B10][col] += 1; V[B11][col] += -1
        W[C01][col] += 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[A11][col] += 1
        V[B01][col] += 1; V[B00][col] += -1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += -1; W[C01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[A10][col] += 1; U[A00][col] += -1
        V[B00][col] += 1; V[B10][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[A01][col] += 1; U[A11][col] += -1
        V[B01][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    # C2 part: 20 terms
    for i in range(2):
        for k in range(10):
            A_idx = i*3 + 2   # A[i,2]
            B_idx = 2*10 + k  # B[2,k]
            C_idx = k*2 + i   # C[i,k]
            U[A_idx][col] += 1
            V[B_idx][col] += 1
            W[C_idx][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
