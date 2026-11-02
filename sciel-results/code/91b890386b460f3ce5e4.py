"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (2x2) * B_1 (2x2) -> C_1 (2x2)
    # Strassen 7 multiplications for 2x2
    # Indices for A_1: rows 0,1; cols 0,1 -> U indices 0,1,2,3
    # Indices for B_1: rows 0,1; cols 0,1 -> V indices 0,1,4,5
    # Indices for C_1: cols 0,1; rows 0,1 -> W indices 0,2,1,3 (k*n+i)
    
    # Strassen steps for M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to U, V, W vectors.
    # U vectors are for A, V for B, W for C.
    # Each multiplication step contributes to U, V, W.
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> V indices: 0,1,4,5
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> W indices: 0,2,1,3
    
    # M1
    # U: A11(0), A22(3)
    # V: B11(0), B22(5)
    # W: C11(0), C12(2), C21(1), C22(3)
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # This is 4 terms if expanded, but Strassen uses 1 mult.
    # In tensor decomposition, each "mult" is a rank-1 term.
    # So we need 7 rank-1 terms.
    
    # Term 1: (A11+A22) * (B11+B22)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 1, 1, 1]
    add_term(0, 0, 0, 1, 1, 1) # A11 B11 -> C11
    # Wait, Strassen combines them.
    # The rank-1 term is $u \otimes v \otimes w$.
    # $u = e_{11} + e_{22}$, $v = e_{11} + e_{22}$, $w = e_{11} + e_{12} + e_{21} + e_{22}$.
    # This is ONE rank-1 term? No, $w$ has 4 non-zeros.
    # A rank-1 tensor has $u_i v_j w_k$.
    # If $u$ has 2 non-zeros, $v$ has 2, $w$ has 4, it is still rank 1.
    # So we just set the entries.
    
    # Let's implement the 7 Strassen terms properly.
    
    # 1. (A11+A22)(B11+B22)
    # U: A11(0)=1, A22(3)=1
    # V: B11(0)=1, B22(5)=1
    # W: C11(0)=1, C12(2)=1, C21(1)=1, C22(3)=1
    U[0][col]=1; U[3][col]=1
    V[0][col]=1; V[5][col]=1
    W[0][col]=1; W[2][col]=1; W[1][col]=1; W[3][col]=1
    col+=1

    # 2. (A21+A22)B11
    # U: A21(2)=1, A22(3)=1
    # V: B11(0)=1
    # W: C21(1)=1, C22(3)=1
    U[2][col]=1; U[3][col]=1
    V[0][col]=1
    W[1][col]=1; W[3][col]=1
    col+=1

    # 3. A11(B21-B22)
    # U: A11(0)=1
    # V: B21(4)=1, B22(5)=-1
    # W: C11(0)=1, C12(2)=-1
    U[0][col]=1
    V[4][col]=1; V[5][col]=-1
    W[0][col]=1; W[2][col]=-1
    col+=1

    # 4. A22(B12-B11)
    # U: A22(3)=1
    # V: B12(1)=1, B11(0)=-1
    # W: C21(1)=1, C22(3)=-1
    U[3][col]=1
    V[1][col]=1; V[0][col]=-1
    W[1][col]=1; W[3][col]=-1
    col+=1

    # 5. (A11+A12)B22
    # U: A11(0)=1, A12(1)=1
    # V: B22(5)=1
    # W: C11(0)=1, C12(2)=1
    U[0][col]=1; U[1][col]=1
    V[5][col]=1
    W[0][col]=1; W[2][col]=1
    col+=1

    # 6. (A21-A11)(B11+B12)
    # U: A21(2)=1, A11(0)=-1
    # V: B11(0)=1, B12(1)=1
    # W: C21(1)=1, C22(3)=1
    U[2][col]=1; U[0][col]=-1
    V[0][col]=1; V[1][col]=1
    W[1][col]=1; W[3][col]=1
    col+=1

    # 7. (A12-A22)(B21+B22)
    # U: A12(1)=1, A22(3)=-1
    # V: B21(4)=1, B22(5)=1
    # W: C11(0)=1, C12(2)=1
    U[1][col]=1; U[3][col]=-1
    V[4][col]=1; V[5][col]=1
    W[0][col]=1; W[2][col]=1
    col+=1

    # Part 2: A_1 (2x2) * b_2 (2x1) -> C_col3 (2x1)
    # A_1 indices: 0,1,2,3
    # b_2 indices: V[2] (row 2, col 2 of B? No, b_2 is col 3 of B, rows 1,2)
    # B is 3x3. b_2 is B[0:2, 2]. V indices: 2, 5?
    # V index = j*p + k.
    # j=0, k=2 -> V[2]. j=1, k=2 -> V[5].
    # C_col3 is C[:, 2]. W indices: k*n+i.
    # k=2, i=0 -> W[4]. k=2, i=1 -> W[5].
    
    # 4 multiplications:
    # A11 * B02 -> C02
    U[0][col]=1; V[2][col]=1; W[4][col]=1; col+=1
    # A12 * B12 -> C02
    U[1][col]=1; V[5][col]=1; W[4][col]=1; col+=1
    # A21 * B02 -> C12
    U[2][col]=1; V[2][col]=1; W[5][col]=1; col+=1
    # A22 * B12 -> C12
    U[3][col]=1; V[5][col]=1; W[5][col]=1; col+=1

    # Part 3: a_2 (2x1) * b_3 (1x2) -> C_cols1,2 (2x2)
    # a_2 is A[:, 2]. U indices: 2, 3?
    # U index = i*m + j. i=0, j=2 -> U[2]. i=1, j=2 -> U[5].
    # b_3 is B[2, 0:2]. V indices: j=2, k=0 -> V[6]. j=2, k=1 -> V[7].
    # C_cols1,2 are C[:, 0:2]. W indices:
    # k=0, i=0 -> W[0]. k=1, i=0 -> W[1].
    # k=0, i=1 -> W[2]. k=1, i=1 -> W[3].
    
    # 4 multiplications:
    # A02 * B20 -> C00
    U[2][col]=1; V[6][col]=1; W[0][col]=1; col+=1
    # A02 * B21 -> C01
    U[2][col]=1; V[7][col]=1; W[1][col]=1; col+=1
    # A12 * B20 -> C10
    U[5][col]=1; V[6][col]=1; W[2][col]=1; col+=1
    # A12 * B21 -> C11
    U[5][col]=1; V[7][col]=1; W[3][col]=1; col+=1

    # Part 4: a_2 (2x1) * b_4 (1x1) -> C_col3 (2x1)
    # a_2: U[2], U[5].
    # b_4: B[2,2]. V index: 2*3+2 = 8.
    # C_col3: W[4], W[5].
    
    # 2 multiplications:
    # A02 * B22 -> C02
    U[2][col]=1; V[8][col]=1; W[4][col]=1; col+=1
    # A12 * B22 -> C12
    U[5][col]=1; V[8][col]=1; W[5][col]=1; col+=1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
