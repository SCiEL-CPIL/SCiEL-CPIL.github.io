import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 84
    # Part 1: i in {0, 1}
    #   Sub 1a: A[0:2, 0:2] * B[0:2, 0:6] -> 3 Strassen blocks (rank 21)
    #   Sub 1b: A[0:2, 2:4] * B[2:4, 0:6] -> 3 Strassen blocks (rank 21)
    #   Sub 1c: A[0:2, 4:5] * B[4:5, 0:6] -> Schoolbook (rank 12)
    # Part 2: i in {2}
    #   Sub 2a: A[2:3, 0:5] * B[0:5, 0:6] -> Schoolbook (rank 30)
    
    rank_total = 84
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A indices: i0, i1 (rows), j0, j1 (cols)
    # B indices: j0, j1 (rows), k0, k1 (cols)
    # C indices: i0, i1 (rows), k0, k1 (cols)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # This formulation is tricky to map to U,V,W directly without expansion.
        # Standard Strassen decomposition:
        # U, V, W are 0/1/-1 matrices.
        
        # m1: (a+d)(e+h)
        # U: A[i0,j0], A[i1,j1] -> 1
        # V: B[j0,k0], B[j1,k1] -> 1
        # W: C[i0,k0] += 1, C[i0,k1] += 1, C[i1,k0] += 1, C[i1,k1] += 1
        # Wait, C00 = m1..., C01 = m1..., C10 = m1..., C11 = m1...
        # Actually:
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # We need to define the 7 products.
        # Product r: U_r * V_r * W_r
        # U_r has non-zeros at A indices.
        # V_r has non-zeros at B indices.
        # W_r has non-zeros at C indices.
        
        # m1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=1, C10=1, C11=1 (Because m1 contributes to all 4? No.)
        # Let's check Strassen formulas again.
        # m1 = (a+d)(e+h)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # So m1 contributes to C00 and C11.
        # W1: C00=1, C11=1.
        
        # m2 = (c+d)e
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=-1
        
        # m3 = a(f-h)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1, C11=1
        
        # m4 = d(g-f)
        # U: A11=1
        # V: B10=1, B01=-1
        # W: C00=1, C10=1
        
        # m5 = (a+b)h
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        
        # m6 = (c-a)(e+f)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        
        # m7 = (b-d)(g+h)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1, C10=1
        
        # Let's implement these 7 steps.
        
        # Step 1: m1
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # Step 2: m2
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = -1
        col += 1
        
        # Step 3: m3
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # Step 4: m4
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k1][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # Step 5: m5
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1
        
        # Step 6: m6
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # Step 7: m7
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
    # Helper to add Schoolbook block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Part 1: i in {0, 1}
    # Sub 1a: A[0:2, 0:2] * B[0:2, 0:6]
    # B cols split into 0:2, 2:4, 4:6
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_strassen(0, 1, 0, 1, 4, 5)
    
    # Sub 1b: A[0:2, 2:4] * B[2:4, 0:6]
    add_strassen(0, 1, 2, 3, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3)
    add_strassen(0, 1, 2, 3, 4, 5)
    
    # Sub 1c: A[0:2, 4:5] * B[4:5, 0:6]
    add_schoolbook(0, 2, 4, 5, 0, 6)
    
    # Part 2: i in {2}
    # Sub 2a: A[2:3, 0:5] * B[0:5, 0:6]
    add_schoolbook(2, 3, 0, 5, 0, 6)
    
    # Verify col == rank_total
    if col != rank_total:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
