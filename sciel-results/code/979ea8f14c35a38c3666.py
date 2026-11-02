def construct():
    """
    Returns a rank-44 exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    1. Split the 3x8 matrix B into B1 (1x8) and B2 (2x8).
    2. Split the 2x3 matrix A into A1 (2x1) and A2 (2x2).
    3. The product is C = A1*B1 + A2*B2.
    4. A1*B1 is a 2x1 * 1x8 multiplication, which has rank 16 (schoolbook).
    5. A2*B2 is a 2x2 * 2x8 multiplication. This can be viewed as 4 independent 2x2 * 2x2 multiplications.
    6. Each 2x2 * 2x2 multiplication has rank 7 (Strassen's algorithm).
    7. Total rank = 16 + 4 * 7 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    # U is (n*m, rank) = (6, 44)
    # V is (m*p, rank) = (24, 44)
    # W is (p*n, rank) = (16, 44)
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
    # A1 corresponds to indices 0, 3 in U (A[0,0], A[1,0])
    # B1 corresponds to indices 0, 1, ..., 7 in V (B[0,0], ..., B[0,7])
    # Output C1 corresponds to indices 0..7 in W (C[0,0], C[1,0], C[0,1], C[1,1], ...)
    # Wait, W index is k*n + i.
    # For C[j, k], index is k*n + j.
    # A1 is column 0 of A. A[i, 0]. Index in U: i*m + 0 = i*3.
    #   A[0,0] -> U[0]
    #   A[1,0] -> U[3]
    # B1 is row 0 of B. B[0, k]. Index in V: 0*p + k = k.
    #   B[0,k] -> V[k]
    # C1 is the contribution to C from this part.
    # C[i, k] += A[i, 0] * B[0, k]
    # Index in W: k*n + i.
    
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0..7
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
    # A2 is A[:, 1:3].
    #   A[0,1] -> U[1]
    #   A[0,2] -> U[2]
    #   A[1,1] -> U[4]
    #   A[1,2] -> U[5]
    # B2 is B[1:3, :].
    #   B[1,k] -> V[1*p + k] = V[8+k]
    #   B[2,k] -> V[2*p + k] = V[16+k]
    # Output C2 is the contribution to C from this part.
    # C[i, k] += A[i, 1] * B[1, k] + A[i, 2] * B[2, k]
    
    # We split the 8 columns into 4 blocks of 2.
    # Block b covers columns 2b and 2b+1.
    # For each block, we have a 2x2 * 2x2 multiplication.
    # A_sub is 2x2:
    #   [[A[0,1], A[0,2]],
    #    [A[1,1], A[1,2]]]
    # B_sub is 2x2:
    #   [[B[1,2b], B[1,2b+1]],
    #    [B[2,2b], B[2,2b+1]]]
    # C_sub is 2x2:
    #   [[C[0,2b], C[0,2b+1]],
    #    [C[1,2b], C[1,2b+1]]]
    
    # Strassen's 7 terms for 2x2 * 2x2:
    # Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)h
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    
    # C = [[m1 + m4 - m5 + m7, m3 + m5],
    #      [m2 - m4 + m6, m1 - m2 + m3 + m6]]
    
    # Mapping to our indices:
    # a = A[0,1] -> U[1]
    # b = A[0,2] -> U[2]
    # c = A[1,1] -> U[4]
    # d = A[1,2] -> U[5]
    
    # e = B[1,2b] -> V[8+2b]
    # f = B[1,2b+1] -> V[8+2b+1]
    # g = B[2,2b] -> V[16+2b]
    # h = B[2,2b+1] -> V[16+2b+1]
    
    # C[0,2b] -> W[2b*2 + 0] = W[4b]
    # C[0,2b+1] -> W[(2b+1)*2 + 0] = W[4b+2]
    # C[1,2b] -> W[2b*2 + 1] = W[4b+1]
    # C[1,2b+1] -> W[(2b+1)*2 + 1] = W[4b+3]
    
    for b in range(4):
        # Indices for B
        e_idx = 8 + 2*b
        f_idx = 8 + 2*b + 1
        g_idx = 16 + 2*b
        h_idx = 16 + 2*b + 1
        
        # Indices for C
        c00_idx = 4*b
        c01_idx = 4*b + 2
        c10_idx = 4*b + 1
        c11_idx = 4*b + 3
        
        # Term 1: (a + d)(e + h)
        U[1][col] = 1
        U[5][col] = 1
        V[e_idx][col] = 1
        V[h_idx][col] = 1
        W[c00_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # Term 2: (c + d)e
        U[4][col] = 1
        U[5][col] = 1
        V[e_idx][col] = 1
        W[c10_idx][col] = 1
        W[c11_idx][col] = -1
        col += 1
        
        # Term 3: a(f - h)
        U[1][col] = 1
        V[f_idx][col] = 1
        V[h_idx][col] = -1
        W[c01_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # Term 4: d(g - e)
        U[5][col] = 1
        V[g_idx][col] = 1
        V[e_idx][col] = -1
        W[c00_idx][col] = 1
        W[c10_idx][col] = 1
        col += 1
        
        # Term 5: (a + b)h
        U[1][col] = 1
        U[2][col] = 1
        V[h_idx][col] = 1
        W[c00_idx][col] = -1
        W[c01_idx][col] = 1
        col += 1
        
        # Term 6: (c - a)(e + f)
        U[4][col] = 1
        U[1][col] = -1
        V[e_idx][col] = 1
        V[f_idx][col] = 1
        W[c10_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # Term 7: (b - d)(g + h)
        U[2][col] = 1
        U[5][col] = -1
        V[g_idx][col] = 1
        V[h_idx][col] = 1
        W[c00_idx][col] = 1
        W[c01_idx][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
