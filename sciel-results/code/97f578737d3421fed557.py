def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    The decomposition takes the form:
    T = sum_{r=0}^{6} u_r \otimes v_r \otimes w_r
    
    Where T is the matrix multiplication tensor.
    U shape: (4, 7) corresponding to A entries (a00, a01, a10, a11)
    V shape: (4, 7) corresponding to B entries (b00, b01, b10, b11)
    W shape: (4, 7) corresponding to C entries (c00, c01, c10, c11)
    
    Note on W indexing: The task specifies W[k*n+i, r] where C = A*B.
    Standard convention: C[i][k] = sum_j A[i][j]B[j][k].
    The tensor T_{i,j,k} = 1 if C[i][k] depends on A[i][j]B[j][k].
    In the decomposition sum u_r[i] v_r[j] w_r[k], the index for W is k*n+i?
    Let's check the prompt: "output factor follows the official trace convention W[k*n+i,r]".
    This implies the index into the W vector is `k*n + i`.
    Let's verify the Strassen terms against this convention.
    
    A = [[a00, a01], [a10, a11]] -> indices 0, 1, 2, 3
    B = [[b00, b01], [b10, b11]] -> indices 0, 1, 2, 3
    C = [[c00, c01], [c10, c11]]
    
    C[0][0] = a00*b00 + a01*b10
    C[0][1] = a00*b01 + a01*b11
    C[1][0] = a10*b00 + a11*b10
    C[1][1] = a10*b01 + a11*b11
    
    Strassen:
    m0 = (a00+a11)(b00+b11)
    m1 = (a10+a11)b00
    m2 = a00(b01-b11)
    m3 = a11(b10-b00)
    m4 = (a00+a01)b11
    m5 = (a10-a00)(b00+b01)
    m6 = (a01-a11)(b10+b11)
    
    c00 = m0 + m3 - m4 + m6
    c01 = m2 + m4
    c10 = m1 + m3
    c11 = m0 - m1 + m2 + m5
    
    Mapping to W index `k*n + i`:
    c00 is i=0, k=0 -> index 0*2+0 = 0
    c01 is i=0, k=1 -> index 1*2+0 = 2
    c10 is i=1, k=0 -> index 0*2+1 = 1
    c11 is i=1, k=1 -> index 1*2+1 = 3
    
    So W vector indices:
    0: c00
    1: c10
    2: c01
    3: c11
    
    Let's construct U, V, W based on this mapping.
    U indices: a00=0, a01=1, a10=2, a11=3
    V indices: b00=0, b01=1, b10=2, b11=3
    W indices: c00=0, c10=1, c01=2, c11=3
    """
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # m0 = (a00+a11)(b00+b11)
    # U: a00(0), a11(3) -> 1
    # V: b00(0), b11(3) -> 1
    # W: c00(0), c11(3) -> 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m1 = (a10+a11)b00
    # U: a10(2), a11(3) -> 1
    # V: b00(0) -> 1
    # W: c10(1) -> 1, c11(3) -> -1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # m2 = a00(b01-b11)
    # U: a00(0) -> 1
    # V: b01(1) -> 1, b11(3) -> -1
    # W: c01(2) -> 1, c11(3) -> 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # m3 = a11(b10-b00)
    # U: a11(3) -> 1
    # V: b10(2) -> 1, b00(0) -> -1
    # W: c00(0) -> 1, c10(1) -> 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # m4 = (a00+a01)b11
    # U: a00(0), a01(1) -> 1
    # V: b11(3) -> 1
    # W: c00(0) -> -1, c01(2) -> 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # m5 = (a10-a00)(b00+b01)
    # U: a10(2) -> 1, a00(0) -> -1
    # V: b00(0) -> 1, b01(1) -> 1
    # W: c11(3) -> 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m6 = (a01-a11)(b10+b11)
    # U: a01(1) -> 1, a11(3) -> -1
    # V: b10(2) -> 1, b11(3) -> 1
    # W: c00(0) -> 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
