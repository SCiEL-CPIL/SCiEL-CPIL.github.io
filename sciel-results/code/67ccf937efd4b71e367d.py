import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 2, 2, 11
    rank = 39
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Strassen terms for 2x2 blocks
    # U indices: 0=a, 1=b, 2=c, 3=d
    # V indices: 0=e, 1=f, 2=g, 3=h (local to block)
    # W indices: 0=C00, 1=C01, 2=C10, 3=C11 (local to block)
    
    # We have 5 blocks of 2 columns, and 1 remaining column.
    for block in range(5):
        k_offset = block * 2
        
        # Map local indices to global V and W indices
        # V: row j, col k. Global index j*p + k.
        # Block columns are k_offset, k_offset+1.
        # Local V indices:
        # e -> (0, k_offset) -> k_offset
        # f -> (0, k_offset+1) -> k_offset + 1
        # g -> (1, k_offset) -> p + k_offset
        # h -> (1, k_offset+1) -> p + k_offset + 1
        
        v_e = k_offset
        v_f = k_offset + 1
        v_g = p + k_offset
        v_h = p + k_offset + 1
        
        # W: row k, col i. Global index k*n + i.
        # Block columns are k_offset, k_offset+1.
        # Local W indices:
        # C00 -> (k_offset, 0) -> k_offset * n
        # C01 -> (k_offset+1, 0) -> (k_offset+1) * n
        # C10 -> (k_offset, 1) -> k_offset * n + 1
        # C11 -> (k_offset+1, 1) -> (k_offset+1) * n + 1
        
        w_c00 = k_offset * n
        w_c01 = (k_offset + 1) * n
        w_c10 = k_offset * n + 1
        w_c11 = (k_offset + 1) * n + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

    # Last column k=10
    k2 = 10
    v_e2 = k2       # B[0, 10]
    v_g2 = p + k2   # B[1, 10]
    w_c02 = k2 * n       # C[0, 10]
    w_c12 = k2 * n + 1   # C[1, 10]
    
    # Schoolbook for 2x2 * 2x1
    # C[0, 10] = A[0,0]B[0,10] + A[0,1]B[1,10]
    # C[1, 10] = A[1,0]B[0,10] + A[1,1]B[1,10]
    
    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1

    return {"U": U, "V": V, "W": W}
