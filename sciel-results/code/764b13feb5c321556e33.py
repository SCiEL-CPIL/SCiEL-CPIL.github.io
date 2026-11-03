import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks (rank 35)
    # Part 2: A2 (2x1) * B2 (1x10) using schoolbook (rank 20)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    strassen_U = [
        [1, 0, 1, 0, 0, 0, 0], # a
        [0, 0, 0, 0, 1, 0, 0], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 0, 0, 1, 0, 0, 0]  # d
    ]
    # Note: Indices in strassen_U correspond to a,b,c,d of the 2x2 matrix A1
    # A1 is rows 0,1 of A, cols 0,1.
    # U indices: 0=A11, 1=A12, 2=A13, 3=A21, 4=A22, 5=A23
    # A1 corresponds to U indices 0,1,3,4.
    
    strassen_V = [
        [1, 0, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 0, 1], # f
        [1, 1, 0, 1, 0, 0, 0], # g
        [1, 0, 1, 1, 1, 0, 0]  # h
    ]
    # V indices for B1 (rows 0,1 of B, cols 0..9)
    # B1 is V indices 0..9 (row 0) and 10..19 (row 1).
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    # W indices for C1 (2x10)
    # C1 is rows 0,1 of C, cols 0..9.
    # W index for C[i,k] is k*n + i.
    # C1[0,0] -> W[0], C1[1,0] -> W[1], C1[0,1] -> W[2], etc.
    
    col = 0
    for r in range(5):
        # Block r uses B cols 2r, 2r+1
        b_col_start = 2 * r
        b_col_end = 2 * r + 2
        
        # Map strassen_U to U
        # A1[0,0] = U[0], A1[0,1] = U[1], A1[1,0] = U[3], A1[1,1] = U[4]
        u_map = [0, 1, 3, 4]
        
        # Map strassen_V to V
        # B1[0,0] = V[b_col_start], B1[0,1] = V[b_col_start+1]
        # B1[1,0] = V[10+b_col_start], B1[1,1] = V[10+b_col_start+1]
        v_map = [
            b_col_start, 
            b_col_start + 1, 
            10 + b_col_start, 
            10 + b_col_start + 1
        ]
        
        # Map strassen_W to W
        # C1[0,0] = W[b_col_start*2 + 0]
        # C1[1,0] = W[b_col_start*2 + 1]
        # C1[0,1] = W[(b_col_start+1)*2 + 0]
        # C1[1,1] = W[(b_col_start+1)*2 + 1]
        w_map = [
            b_col_start * 2 + 0,
            b_col_start * 2 + 1,
            (b_col_start + 1) * 2 + 0,
            (b_col_start + 1) * 2 + 1
        ]
        
        for s in range(7):
            # U
            for u_idx, coeff in zip(u_map, strassen_U[s]):
                if coeff != 0:
                    U[u_idx][col] = coeff
            # V
            for v_idx, coeff in zip(v_map, strassen_V[s]):
                if coeff != 0:
                    V[v_idx][col] = coeff
            # W
            for w_idx, coeff in zip(w_map, strassen_W[s]):
                if coeff != 0:
                    W[w_idx][col] = coeff
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x10)
    # A2 is U indices 2 (A13) and 5 (A23)
    # B2 is V indices 20..29 (row 2 of B)
    # C2 is W indices 0..19 (all of C, since C = C1 + C2)
    
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(10):
            u_idx = 2 + i # A13=2, A23=5
            v_idx = 20 + k # B31..B310
            w_idx = k * 2 + i # C[i,k]
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
