import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Base case: 2x2x2 Strassen
    def get_strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # A: 00(0), 01(1), 10(2), 11(3)
        # B: 00(0), 01(1), 10(2), 11(3)
        # C: 00(0), 01(1), 10(2), 11(3)
        
        # T1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # T2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # T3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # T4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # T5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # T6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # T7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def get_trivial(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = (i*m + j)*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return U, V, W

    def combine_dim(U1, V1, W1, U2, V2, W2, dim):
        # dim: 0 for n, 1 for m, 2 for p
        # n1, m1, p1 from U1, V1, W1
        # n2, m2, p2 from U2, V2, W2
        # They must match on other dims.
        
        n1 = len(W1) // len(U1) # p*n / n*m -> p/m? No.
        # U: n*m, V: m*p, W: p*n
        # n1 = len(U1) // m1
        # We need to infer dimensions.
        
        # Let's pass dimensions explicitly to be safe.
        pass

    # Better to pass dimensions.
    
    def combine(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2, dim):
        if dim == 0: # n
            assert m1 == m2 and p1 == p2
            n = n1 + n2
            m = m1
            p = p1
            r1 = len(W1[0])
            r2 = len(W2[0])
            rank = r1 + r2
            
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # U
            for i in range(n1):
                for j in range(m):
                    for r in range(r1):
                        U[i*m+j][r] = U1[i*m1+j][r]
            for i in range(n2):
                for j in range(m):
                    for r in range(r2):
                        U[(n1+i)*m+j][r1+r] = U2[i*m2+j][r]
                        
            # V
            for j in range(m):
                for k in range(p):
                    for r in range(r1):
                        V[j*p+k][r] = V1[j*p1+k][r]
                    for r in range(r2):
                        V[j*p+k][r1+r] = V2[j*p2+k][r]
                        
            # W
            for k in range(p):
                for i in range(n1):
                    for r in range(r1):
                        W[k*n+i][r] = W1[k*p1+i][r]
                for i in range(n2):
                    for r in range(r2):
                        W[k*n + (n1+i)][r1+r] = W2[k*p2+i][r]
                        
            return U, V, W, n, m, p

        elif dim == 1: # m
            assert n1 == n2 and p1 == p2
            n = n1
            m = m1 + m2
            p = p1
            r1 = len(W1[0])
            r2 = len(W2[0])
            rank = r1 + r2
            
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # U
            for i in range(n):
                for j in range(m1):
                    for r in range(r1):
                        U[i*m+j][r] = U1[i*m1+j][r]
                for j in range(m2):
                    for r in range(r2):
                        U[i*m + (m1+j)][r1+r] = U2[i*m2+j][r]
                        
            # V
            for j in range(m1):
                for k in range(p):
                    for r in range(r1):
                        V[j*p+k][r] = V1[j*p1+k][r]
            for j in range(m2):
                for k in range(p):
                    for r in range(r2):
                        V[(m1+j)*p+k][r1+r] = V2[j*p2+k][r]
                        
            # W
            for k in range(p):
                for i in range(n):
                    for r in range(r1):
                        W[k*n+i][r] = W1[k*p1+i][r]
                    for r in range(r2):
                        W[k*n+i][r1+r] = W2[k*p2+i][r]
                        
            return U, V, W, n, m, p

        elif dim == 2: # p
            assert n1 == n2 and m1 == m2
            n = n1
            m = m1
            p = p1 + p2
            r1 = len(W1[0])
            r2 = len(W2[0])
            rank = r1 + r2
            
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # U
            for i in range(n):
                for j in range(m):
                    for r in range(r1):
                        U[i*m+j][r] = U1[i*m1+j][r]
                    for r in range(r2):
                        U[i*m+j][r1+r] = U2[i*m2+j][r]
                        
            # V
            for j in range(m):
                for k in range(p1):
                    for r in range(r1):
                        V[j*p+k][r] = V1[j*p1+k][r]
                for k in range(p2):
                    for r in range(r2):
                        V[j*p + (p1+k)][r1+r] = V2[j*p2+k][r]
                        
            # W
            for k in range(p1):
                for i in range(n):
                    for r in range(r1):
                        W[k*n+i][r] = W1[k*p1+i][r]
            for k in range(p2):
                for i in range(n):
                    for r in range(r2):
                        W[(p1+k)*n+i][r1+r] = W2[k*p2+i][r]
                        
            return U, V, W, n, m, p

    # 1. 2x2x2
    U, V, W, n, m, p = get_strassen_222(), None, None, 2, 2, 2
    U222, V222, W222 = get_strassen_222()
    
    # 2. 2x4x2 (double m)
    U242, V242, W242, n242, m242, p242 = combine(U222, V222, W222, 2, 2, 2, U222, V222, W222, 2, 2, 2, 1)
    
    # 3. 2x4x4 (double p)
    U244, V244, W244, n244, m244, p244 = combine(U242, V242, W242, n242, m242, p242, U242, V242, W242, n242, m242, p242, 2)
    
    # 4. 1x4x4 (trivial)
    U144, V144, W144 = get_trivial(1, 4, 4)
    
    # 5. 3x4x4 (combine n: 2+1)
    U344, V344, W344, n344, m344, p344 = combine(U244, V244, W244, n244, m244, p244, U144, V144, W144, 1, 4, 4, 0)
    
    # 6. 3x4x8 (double p)
    U348, V348, W348, n348, m348, p348 = combine(U344, V344, W344, n344, m344, p344, U344, V344, W344, n344, m344, p344, 2)
    
    if time.time() > deadline:
        # Fallback to naive if something went wrong or took too long (unlikely for this small size)
        n, m, p = 3, 4, 8
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U348, "V": V348, "W": W348}
