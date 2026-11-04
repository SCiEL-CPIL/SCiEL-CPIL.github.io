def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        return col + 1

    col = 0
    # Part 1: j=0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            col = add_term(u_idx, v_idx, w_idx)

    # Part 2: j=1,2 (Strassen on 2x2x2 blocks)
    # U indices: A(i,1) -> i*m+1, A(i,2) -> i*m+2
    # V indices: B(1,k) -> 1*p+k, B(2,k) -> 2*p+k
    # W indices: C(i,k) -> k*n+i
    
    for b in range(5):
        k0 = 2 * b
        
        # M1 = (A01 + A12)(B1_k0 + B2_k0+1)  # Indices in U: 1, 5. V: 10+k0, 20+k0+1
        # Note: A01 is i=0, j=1 (idx 1). A12 is i=1, j=2 (idx 5).
        # B1_k0 is j=1, k=k0 (idx 10+k0). B2_k0+1 is j=2, k=k0+1 (idx 20+k0+1).
        
        # Strassen M1: (A00+A11)(B00+B11) -> Here A00=A(0,1), A11=A(1,2). B00=B(1,k0), B11=B(2,k0+1)
        U[1][col] += 1; U[5][col] += 1
        V[10+k0][col] += 1; V[20+k0+1][col] += 1
        W[2*k0][col] += 1; W[2*(k0+1)+1][col] += 1 # C00, C11
        col += 1

        # M2: (A10+A11)B00 -> A10=A(1,1) idx 4, A11=A(1,2) idx 5. B00=B(1,k0) idx 10+k0
        U[4][col] += 1; U[5][col] += 1
        V[10+k0][col] += 1
        W[2*k0+1][col] += 1; W[2*(k0+1)+1][col] -= 1 # C10, C11
        col += 1

        # M3: A00(B01-B11) -> A00=A(0,1) idx 1. B01=B(1,k0+1) idx 10+k0+1. B11=B(2,k0+1) idx 20+k0+1
        U[1][col] += 1
        V[10+k0+1][col] += 1; V[20+k0+1][col] -= 1
        W[2*(k0+1)][col] += 1; W[2*(k0+1)+1][col] += 1 # C01, C11
        col += 1

        # M4: A11(B10-B00) -> A11=A(1,2) idx 5. B10=B(2,k0) idx 20+k0. B00=B(1,k0) idx 10+k0
        U[5][col] += 1
        V[20+k0][col] += 1; V[10+k0][col] -= 1
        W[2*k0][col] += 1; W[2*k0+1][col] += 1 # C00, C10
        col += 1

        # M5: (A00+A01)B11 -> A00=A(0,1) idx 1. A01=A(0,2) idx 2. B11=B(2,k0+1) idx 20+k0+1
        U[1][col] += 1; U[2][col] += 1
        V[20+k0+1][col] += 1
        W[2*k0][col] -= 1; W[2*(k0+1)][col] += 1 # C00, C01
        col += 1

        # M6: (A10-A00)(B00+B01) -> A10=A(1,1) idx 4. A00=A(0,1) idx 1. B00=B(1,k0) idx 10+k0. B01=B(1,k0+1) idx 10+k0+1
        U[4][col] += 1; U[1][col] -= 1
        V[10+k0][col] += 1; V[10+k0+1][col] += 1
        W[2*(k0+1)+1][col] += 1 # C11
        col += 1

        # M7: (A01-A11)(B10+B11) -> A01=A(0,2) idx 2. A11=A(1,2) idx 5. B10=B(2,k0) idx 20+k0. B11=B(2,k0+1) idx 20+k0+1
        U[2][col] += 1; U[5][col] -= 1
        V[20+k0][col] += 1; V[20+k0+1][col] += 1
        W[2*k0][col] += 1 # C00
        col += 1

    return {"U": U, "V": V, "W": W}
