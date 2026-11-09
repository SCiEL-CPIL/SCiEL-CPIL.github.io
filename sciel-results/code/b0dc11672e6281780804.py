"""Matrix multiplication 2x5x6 rank-minimal decomposition via Strassen-based recursive splitting."""

def _strassen_222():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    
    U: shape (4, 7) - indices (i, j)
    V: shape (4, 7) - indices (j, k)
    W: shape (4, 7) - indices (k, n) where n=2, row index is k*2+i
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Index conventions:
    # A is 2x2: A[i,j], i,j in {0,1}
    # B is 2x2: B[j,k], j,k in {0,1}
    # C is 2x2: C[i,k], i,k in {0,1}
    #
    # U rows indexed by (i,j) -> row = i*2 + j
    # V rows indexed by (j,k) -> row = j*2 + k
    # W rows indexed by (k,i) -> row = k*2 + i  (trace convention: W[k*n+i, r])
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 = A[0,0], A22 = A[1,1]
    # B11 = B[0,0], B22 = B[1,1]
    # C11 += M1, C22 += M1
    U[0*2+0][0] = 1  # A[0,0]
    U[1*2+1][0] = 1  # A[1,1]
    V[0*2+0][0] = 1  # B[0,0]
    V[1*2+1][0] = 1  # B[1,1]
    W[0*2+0][0] = 1  # C[0,0]
    W[1*2+1][0] = 1  # C[1,1]
    
    # M2 = (A21 + A22)B11
    # A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0]
    # C21 += M2, C22 -= M2
    U[1*2+0][1] = 1  # A[1,0]
    U[1*2+1][1] = 1  # A[1,1]
    V[0*2+0][1] = 1  # B[0,0]
    W[0*2+1][1] = 1  # C[1,0]
    W[1*2+1][1] = -1 # C[1,1]
    
    # M3 = A11(B21 - B22)
    # A11 = A[0,0]
    # B21 = B[1,0], B22 = B[1,1]
    # C12 += M3, C22 += M3
    U[0*2+0][2] = 1  # A[0,0]
    V[1*2+0][2] = 1  # B[1,0]
    V[1*2+1][2] = -1 # B[1,1]
    W[0*2+0][2] = 1  # C[0,1]
    W[1*2+1][2] = 1  # C[1,1]
    
    # M4 = A22(B11 - B21)
    # A22 = A[1,1]
    # B11 = B[0,0], B21 = B[1,0]
    # C11 += M4, C21 += M4
    U[1*2+1][3] = 1  # A[1,1]
    V[0*2+0][3] = 1  # B[0,0]
    V[1*2+0][3] = -1 # B[1,0]
    W[0*2+0][3] = 1  # C[0,0]
    W[0*2+1][3] = 1  # C[1,0]
    
    # M5 = (A11 + A12)B22
    # A11 = A[0,0], A12 = A[0,1]
    # B22 = B[1,1]
    # C11 -= M5, C12 += M5
    U[0*2+0][4] = 1  # A[0,0]
    U[0*2+1][4] = 1  # A[0,1]
    V[1*2+1][4] = 1  # B[1,1]
    W[0*2+0][4] = -1 # C[0,0]
    W[0*2+1][4] = 1  # C[0,1]
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21 = A[1,0], A11 = A[0,0]
    # B11 = B[0,0], B12 = B[0,1]
    # C22 += M6
    U[1*2+0][5] = 1  # A[1,0]
    U[0*2+0][5] = -1 # A[0,0]
    V[0*2+0][5] = 1  # B[0,0]
    V[0*2+1][5] = 1  # B[0,1]
    W[1*2+1][5] = 1  # C[1,1]
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12 = A[0,1], A22 = A[1,1]
    # B21 = B[1,0], B22 = B[1,1]
    # C11 += M7
    U[0*2+1][6] = 1  # A[0,1]
    U[1*2+1][6] = -1 # A[1,1]
    V[1*2+0][6] = 1  # B[1,0]
    V[1*2+1][6] = 1  # B[1,1]
    W[0*2+0][6] = 1  # C[0,0]
    
    return U, V, W


def _trivial_1mp(m, p):
    """Return U, V, W for 1xm x mxp matrix multiplication (trivial/schoolbook).
    
    U: shape (m, m*p)
    V: shape (m*p, m*p)
    W: shape (p, m*p)
    """
    rank = m * p
    U = [[0]*rank for _ in range(m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p)]
    
    for j in range(m):
        for k in range(p):
            col = j * p + k
            U[j][col] = 1
            V[j*p + k][col] = 1
            W[k][col] = 1
    
    return U, V, W


def _matmul_factors(U1, V1, W1, U2, V2, W2):
    """Combine two independent decompositions.
    
    For splitting where the sub-problems have disjoint supports in i (first dimension),
    we concatenate: U = [U1; U2], V = [V1; V2], W = W1 + W2 element-wise (if same size) 
    or handle appropriately.
    
    Actually, for our use case, we'll handle the combination explicitly in construct().
    """
    pass


def _combine_for_n_split(U1, V1, W1, U2, V2, W2, n1, m, p):
    """Combine decompositions for n = n1 + n2 split.
    
    U1: (n1*m, rank1), U2: (n2*m, rank2)
    V1: (m*p, rank1), V2: (m*p, rank2)
    W1: (p*n1, rank1), W2: (p*n2, rank2)
    
    Result: U: (n*m, rank1+rank2), V: (m*p, rank1+rank2), W: (p*n, rank1+rank2)
    
    For rows of U and V: horizontal concatenation
    For W: vertical concatenation
    """
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    
    # U: vertical stack
    U = [[0]*total_rank for _ in range(n1*m + n2*m)]
    for i in range(n1*m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(n2*m):
        for r in range(rank2):
            U[n1*m + i][rank1 + r] = U2[i][r]
    
    # V: horizontal stack (same rows, concatenated columns)
    V = [[0]*total_rank for _ in range(m*p)]
    for i in range(m*p):
        for r in range(rank1):
            V[i][r] = V1[i][r]
        for r in range(rank2):
            V[i][rank1 + r] = V2[i][r]
    
    # W: vertical stack
    W = [[0]*total_rank for _ in range(p*n1 + p*n2)]
    for i in range(p*n1):
        for r in range(rank1):
            W[i][r] = W1[i][r]
    for i in range(p*n2):
        for r in range(rank2):
            W[p*n1 + i][rank1 + r] = W2[i][r]
    
    return U, V, W


def _combine_for_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """Combine decompositions for m = m1 + m2 split.
    
    U1: (n*m1, rank1), U2: (n*m2, rank2)
    V1: (m1*p, rank1), V2: (m2*p, rank2)
    W1: (p*n, rank1), W2: (p*n, rank2)
    
    Result: U: (n*m, rank1+rank2), V: (m*p, rank1+rank2), W: (p*n, rank1+rank2)
    
    For U: vertical stack
    For V: vertical stack
    For W: element-wise addition (horizontal combination of columns doesn't apply; 
           instead, W entries are summed)
    """
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    
    # U: vertical stack
    U = [[0]*total_rank for _ in range(n*m1 + n*m2)]
    for i in range(n*m1):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(n*m2):
        for r in range(rank2):
            U[n*m1 + i][rank1 + r] = U2[i][r]
    
    # V: vertical stack
    V = [[0]*total_rank for _ in range(m1*p + m2*p)]
    for i in range(m1*p):
        for r in range(rank1):
            V[i][r] = V1[i][r]
    for i in range(m2*p):
        for r in range(rank2):
            V[m1*p + i][rank1 + r] = V2[i][r]
    
    # W: sum of columns (each has rank1 and rank2 columns)
    W = [[0]*total_rank for _ in range(p*n)]
    for i in range(p*n):
        for r in range(rank1):
            W[i][r] = W1[i][r]
        for r in range(rank2):
            W[i][rank1 + r] = W2[i][r]
    
    return U, V, W


def _combine_for_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """Combine decompositions for p = p1 + p2 split.
    
    U1: (n*m, rank1), U2: (n*m, rank2)
    V1: (m*p1, rank1), V2: (m*p2, rank2)
    W1: (p1*n, rank1), W2: (p2*n, rank2)
    
    Result: U: (n*m, rank1+rank2), V: (m*p, rank1+rank2), W: (p*n, rank1+rank2)
    
    For U: horizontal stack
    For V: vertical stack
    For W: vertical stack
    """
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    
    # U: horizontal stack
    U = [[0]*total_rank for _ in range(n*m)]
    for i in range(n*m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
        for r in range(rank2):
            U[i][rank1 + r] = U2[i][r]
    
    # V: vertical stack
    V = [[0]*total_rank for _ in range(m*p1 + m*p2)]
    for i in range(m*p1):
        for r in range(rank1):
            V[i][r] = V1[i][r]
    for i in range(m*p2):
        for r in range(rank2):
            V[m*p1 + i][rank1 + r] = V2[i][r]
    
    # W: vertical stack
    W = [[0]*total_rank for _ in range(p1*n + p2*n)]
    for i in range(p1*n):
        for r in range(rank1):
            W[i][r] = W1[i][r]
    for i in range(p2*n):
        for r in range(rank2):
            W[p1*n + i][rank1 + r] = W2[i][r]
    
    return U, V, W


def construct():
    """Construct rank-minimal decomposition for 2x5x6 matrix multiplication.
    
    Strategy: Split the middle dimension m=5 into m1=2 and m2=3, or split
    n=2 and p=6 to get sub-problems that can use Strassen.
    
    Let's try: Split p=6 into p1=2, p2=4.
    Sub-problem 1: 2x5x2
    Sub-problem 2: 2x5x4
    
    For 2x5x2: Split n=2 into n1=1, n2=1.
      Sub 1.1: 1x5x2 (trivial, rank 10)
      Sub 1.2: 1x5x2 (trivial, rank 10)
      Total: rank 20
    
    For 2x5x4: Split n=2 into n1=1, n2=1.
      Sub 2.1: 1x5x4 (trivial, rank 20)
      Sub 2.2: 1x5x4 (trivial, rank 20)
      Total: rank 40
    
    Total: 60. No improvement.
    
    Better: Use Strassen where possible.
    Split m=5 into m1=2, m2=3.
    Sub 1: 2x2x6
    Sub 2: 2x3x6
    
    For 2x2x6: Split p=6 into p1=2, p2=4.
      Sub 1.1: 2x2x2 -> Strassen, rank 7
      Sub 1.2: 2x2x4 -> Split p=4 into p1=2, p2=2.
        Sub 1.2.1: 2x2x2 -> Strassen, rank 7
        Sub 1.2.2: 2x2x2 -> Strassen, rank 7
        Total: rank 14
      Total for 2x2x6: rank 7 + 14 = 21
    
    For 2x3x6: Split m=3 into m1=1, m2=2.
      Sub 2.1: 2x1x6 -> Trivial, rank 12
      Sub 2.2: 2x2x6 -> As above, rank 21
      Total: rank 33
    
    Total: 21 + 33 = 54.
    
    This gives rank 54 vs schoolbook 60.
    """
    n, m, p = 2, 5, 6
    
    # Split m=5 into m1=2, m2=3
    m1, m2 = 2, 3
    
    # Sub-problem 1: n x m1 x p = 2 x 2 x 6
    # Split p=6 into p1=2, p2=4
    p1, p2 = 2, 4
    
    # Sub 1.1: 2x2x2 -> Strassen
    U_11, V_11, W_11 = _strassen_222()
    
    # Sub 1.2: 2x2x4 -> Split p=4 into 2+2
    # Sub 1.2.1: 2x2x2 -> Strassen
    U_121, V_121, W_121 = _strassen_222()
    # Sub 1.2.2: 2x2x2 -> Strassen
    U_122, V_122, W_122 = _strassen_222()
    
    # Combine 1.2.1 and 1.2.2 (p split: p1=2, p2=2)
    U_12, V_12, W_12 = _combine_for_p_split(U_121, V_121, W_121, U_122, V_122, W_122, n, m1, 2, 2)
    
    # Combine 1.1 and 1.2 (p split: p1=2, p2=4)
    U_1, V_1, W_1 = _combine_for_p_split(U_11, V_11, W_11, U_12, V_12, W_12, n, m1, p1, p2)
    
    # Sub-problem 2: n x m2 x p = 2 x 3 x 6
    # Split m=3 into m1=1, m2=2
    mm1, mm2 = 1, 2
    
    # Sub 2.1: 2x1x6 -> Trivial
    U_21, V_21, W_21 = _trivial_1mp(mm1, p)
    
    # Sub 2.2: 2x2x6 -> Same as sub-problem 1
    # Reuse the same structure
    U_221, V_221, W_221 = _strassen_222()
    U_2221, V_2221, W_2221 = _strassen_222()
    U_2222, V_2222, W_2222 = _strassen_222()
    
    U_222, V_222, W_222 = _combine_for_p_split(U_2221, V_2221, W_2221, U_2222, V_2222, W_2222, n, mm2, 2, 2)
    U_22, V_22, W_22 = _combine_for_p_split(U_221, V_221, W_221, U_222, V_222, W_222, n, mm2, p1, p2)
    
    # Combine 2.1 and 2.2 (m split: m1=1, m2=2)
    U_2, V_2, W_2 = _combine_for_m_split(U_21, V_21, W_21, U_22, V_22, W_22, n, mm1, mm2, p)
    
    # Combine sub-problem 1 and 2 (m split: m1=2, m2=3)
    U, V, W = _combine_for_m_split(U_1, V_1, W_1, U_2, V_2, W_2, n, m1, m2, p)
    
    return {"U": U, "V": V, "W": W}
