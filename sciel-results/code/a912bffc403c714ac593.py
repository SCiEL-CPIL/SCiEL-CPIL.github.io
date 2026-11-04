"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition using Strassen on 2x2 blocks
    # A is 3x3, B is 3x4
    # A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # B11 (2x2), B12 (2x2), B21 (1x2), B22 (1x2)
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Initialize U, V, W as lists of zeros
    # We will build them column by column
    
    # Helper to create a vector of length n*m with a value at index
    def make_u(idx, val):
        vec = [0.0] * (n * m)
        vec[idx] = val
        return vec
        
    def make_v(idx, val):
        vec = [0.0] * (m * p)
        vec[idx] = val
        return vec
        
    def make_w(idx, val):
        vec = [0.0] * (p * n)
        vec[idx] = val
        return vec

    # Indices for A (3x3)
    # A[i, j] -> index i*m + j
    # A11: (0,0), (0,1), (1,0), (1,1)
    a00 = 0
    a01 = 1
    a10 = 3
    a11 = 4
    # A12: (0,2), (1,2)
    a02 = 2
    a12 = 5
    # A21: (2,0), (2,1)
    a20 = 6
    a21 = 7
    # A22: (2,2)
    a22 = 8

    # Indices for B (3x4)
    # B[i, j] -> index i*p + j
    # B11: (0,0), (0,1), (1,0), (1,1)
    b00 = 0
    b01 = 1
    b10 = 4
    b11 = 5
    # B12: (0,2), (0,3), (1,2), (1,3)
    b02 = 2
    b03 = 3
    b12 = 6
    b13 = 7
    # B21: (2,0), (2,1)
    b20 = 8
    b21 = 9
    # B22: (2,2), (2,3)
    b22 = 10
    b23 = 11

    # Indices for W (p*n = 12)
    # W[k*n + i, r] corresponds to C[i, k]
    # C[i, k] -> index i*p + k? No, W is p*n.
    # The tensor is T[i, j, k] = A[i, j] B[j, k].
    # W index is k*n + i.
    
    # Strassen for A11 * X where X = B11 + B12
    # A11 = [[a00, a01], [a10, a11]]
    # X = [[b00+b02, b01+b03], [b10+b12, b11+b13]]
    
    # m1 = (a00+a11) * (x00+x11)
    u = [0.0]*9; u[a00]=1; u[a11]=1
    v = [0.0]*12; v[b00]=1; v[b11]=1; v[b02]=1; v[b13]=1
    w = [0.0]*12
    # C[i, k] gets contribution from m1
    # m1 contributes to C11, C12, C21, C22
    # C11 (i=0, k=0): W[0*3+0] = W[0]
    # C12 (i=0, k=1): W[1*3+0] = W[3]
    # C21 (i=1, k=0): W[0*3+1] = W[1]
    # C22 (i=1, k=1): W[1*3+1] = W[4]
    # Strassen reconstruction:
    # C11 = (m1 - m2 + m3 + m4) / 2 ? No.
    # Standard Strassen:
    # C11 = m1 - m4 + m6 - m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m5 - m6
    # We need to map these to W.
    
    # Let's just add the 7 terms for m1..m7 for X
    # And 7 terms for Y = B12 - B11
    
    # X terms
    # m1_X = (a00+a11) * (b00+b11+b02+b13)
    u = [0.0]*9; u[a00]=1; u[a11]=1
    v = [0.0]*12; v[b00]=1; v[b11]=1; v[b02]=1; v[b13]=1
    w = [0.0]*12
    w[0] = 1   # C11
    w[3] = 1   # C12
    w[1] = 1   # C21
    w[4] = 1   # C22
    add_term(u, v, w)

    # m2_X = (a10+a11) * (b00+b02)
    u = [0.0]*9; u[a10]=1; u[a11]=1
    v = [0.0]*12; v[b00]=1; v[b02]=1
    w = [0.0]*12
    w[1] = 1   # C21
    w[4] = 1   # C22
    add_term(u, v, w)

    # m3_X = a00 * (b01+b03 - b11-b13)
    u = [0.0]*9; u[a00]=1
    v = [0.0]*12; v[b01]=1; v[b03]=1; v[b11]=-1; v[b13]=-1
    w = [0.0]*12
    w[3] = 1   # C12
    w[4] = -1  # C22
    add_term(u, v, w)

    # m4_X = a11 * (b10+b12 - b00-b02)
    u = [0.0]*9; u[a11]=1
    v = [0.0]*12; v[b10]=1; v[b12]=1; v[b00]=-1; v[b02]=-1
    w = [0.0]*12
    w[0] = -1  # C11
    w[1] = 1   # C21
    add_term(u, v, w)

    # m5_X = (a00+a01) * (b11+b13)
    u = [0.0]*9; u[a00]=1; u[a01]=1
    v = [0.0]*12; v[b11]=1; v[b13]=1
    w = [0.0]*12
    w[3] = 1   # C12
    w[4] = 1   # C22
    add_term(u, v, w)

    # m6_X = (a10-a00) * (b00+b02 + b01+b03)
    u = [0.0]*9; u[a10]=1; u[a00]=-1
    v = [0.0]*12; v[b00]=1; v[b02]=1; v[b01]=1; v[b03]=1
    w = [0.0]*12
    w[0] = -1  # C11
    w[3] = 1   # C12
    add_term(u, v, w)

    # m7_X = (a01-a11) * (b10+b12 + b11+b13)
    u = [0.0]*9; u[a01]=1; u[a11]=-1
    v = [0.0]*12; v[b10]=1; v[b12]=1; v[b11]=1; v[b13]=1
    w = [0.0]*12
    w[1] = -1  # C21
    w[4] = -1  # C22
    add_term(u, v, w)
    
    # Y terms (B12 - B11)
    # m1_Y = (a00+a11) * (b02+b13 - b00-b11)
    u = [0.0]*9; u[a00]=1; u[a11]=1
    v = [0.0]*12; v[b02]=1; v[b13]=1; v[b00]=-1; v[b11]=-1
    w = [0.0]*12
    w[0] = 1   # C11
    w[3] = 1   # C12
    w[1] = 1   # C21
    w[4] = 1   # C22
    add_term(u, v, w)

    # m2_Y = (a10+a11) * (b02-b00)
    u = [0.0]*9; u[a10]=1; u[a11]=1
    v = [0.0]*12; v[b02]=1; v[b00]=-1
    w = [0.0]*12
    w[1] = 1   # C21
    w[4] = 1   # C22
    add_term(u, v, w)

    # m3_Y = a00 * (b03-b13 - (b01-b11))
    u = [0.0]*9; u[a00]=1
    v = [0.0]*12; v[b03]=1; v[b13]=-1; v[b01]=-1; v[b11]=1
    w = [0.0]*12
    w[3] = 1   # C12
    w[4] = -1  # C22
    add_term(u, v, w)

    # m4_Y = a11 * (b12-b02 - (b10-b00))
    u = [0.0]*9; u[a11]=1
    v = [0.0]*12; v[b12]=1; v[b02]=-1; v[b10]=-1; v[b00]=1
    w = [0.0]*12
    w[0] = -1  # C11
    w[1] = 1   # C21
    add_term(u, v, w)

    # m5_Y = (a00+a01) * (b13-b11)
    u = [0.0]*9; u[a00]=1; u[a01]=1
    v = [0.0]*12; v[b13]=1; v[b11]=-1
    w = [0.0]*12
    w[3] = 1   # C12
    w[4] = 1   # C22
    add_term(u, v, w)

    # m6_Y = (a10-a00) * (b02-b00 + b03-b01)
    u = [0.0]*9; u[a10]=1; u[a00]=-1
    v = [0.0]*12; v[b02]=1; v[b00]=-1; v[b03]=1; v[b01]=-1
    w = [0.0]*12
    w[0] = -1  # C11
    w[3] = 1   # C12
    add_term(u, v, w)

    # m7_Y = (a01-a11) * (b12-b02 + b13-b11)
    u = [0.0]*9; u[a01]=1; u[a11]=-1
    v = [0.0]*12; v[b12]=1; v[b02]=-1; v[b13]=1; v[b11]=-1
    w = [0.0]*12
    w[1] = -1  # C21
    w[4] = -1  # C22
    add_term(u, v, w)

    # Scalar terms
    # A12 B21: A[i,2] B[2,j] for i in 0,1 and j in 0,1
    for i in range(2):
        for j in range(2):
            u = [0.0]*9; u[i*m + 2] = 1
            v = [0.0]*12; v[2*p + j] = 1
            w = [0.0]*12
            w[j*n + i] = 1
            add_term(u, v, w)

    # A12 B22: A[i,2] B[2,j] for i in 0,1 and j in 2,3
    for i in range(2):
        for j in range(2, 4):
            u = [0.0]*9; u[i*m + 2] = 1
            v = [0.0]*12; v[2*p + j] = 1
            w = [0.0]*12
            w[j*n + i] = 1
            add_term(u, v, w)

    # A21 B11: A[2,i] B[i,j] for i in 0,1 and j in 0,1
    for i in range(2):
        for j in range(2):
            u = [0.0]*9; u[2*m + i] = 1
            v = [0.0]*12; v[i*p + j] = 1
            w = [0.0]*12
            w[j*n + 2] = 1
            add_term(u, v, w)

    # A21 B12: A[2,i] B[i,j] for i in 0,1 and j in 2,3
    for i in range(2):
        for j in range(2, 4):
            u = [0.0]*9; u[2*m + i] = 1
            v = [0.0]*12; v[i*p + j] = 1
            w = [0.0]*12
            w[j*n + 2] = 1
            add_term(u, v, w)

    # A22 B21: A[2,2] B[2,j] for j in 0,1
    for j in range(2):
        u = [0.0]*9; u[2*m + 2] = 1
        v = [0.0]*12; v[2*p + j] = 1
        w = [0.0]*12
        w[j*n + 2] = 1
        add_term(u, v, w)

    # A22 B22: A[2,2] B[2,j] for j in 2,3
    for j in range(2, 4):
        u = [0.0]*9; u[2*m + 2] = 1
        v = [0.0]*12; v[2*p + j] = 1
        w = [0.0]*12
        w[j*n + 2] = 1
        add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
