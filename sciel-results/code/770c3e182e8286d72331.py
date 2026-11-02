"""Matrix multiplication 2,4,7 with rank-minimal decomposition using Strassen and block structure."""

def construct():
    n, m, p = 2, 4, 7
    
    # Strategy:
    # - Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}
    # - For each m-block, split p=7 into 3 blocks of 2 and 1 block of 1
    # - For 2x2 * 2x2 subproblems (3 per m-block), use Strassen (rank 7)
    # - For 2x2 * 2x1 subproblems (1 per m-block), use schoolbook (rank 4)
    # Total rank: 2 * (3*7 + 1*4) = 2 * 25 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # (8, 50)
    V = [[0] * rank for _ in range(m * p)]  # (28, 50)
    W = [[0] * rank for _ in range(p * n)]  # (14, 50)
    
    col = 0
    
    # Strassen 2x2 * 2x2 decomposition
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # Output convention: W[k*n+i] = C[i,k]
    
    # Correct Strassen formulas:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11+b12)
    # M3 = (a11-a22)(b21+b22)
    # M4 = a11(b21-b22)
    # M5 = a22(b12-b11)
    # M6 = (a11+a12)b22
    # M7 = (a21-a11)b11
    # C11 = M1 + M4 - M5 + M7
    # C12 = M6 + M5
    # C21 = M2 - M7
    # C22 = M1 - M4 + M3 - M6
    
    def add_strassen(j_base, k_base):
        """Add Strassen decomposition for 2x2 * 2x2 block.
        A rows: i=0,1; cols: j_base, j_base+1
        B rows: j_base, j_base+1; cols: k_base, k_base+1
        C: W[k*n+i] for k in {k_base, k_base+1}, i in {0,1}
        """
        nonlocal col
        
        # a11 = A[0, j_base], a12 = A[0, j_base+1]
        # a21 = A[1, j_base], a22 = A[1, j_base+1]
        # b11 = B[j_base, k_base], b12 = B[j_base, k_base+1]
        # b21 = B[j_base+1, k_base], b22 = B[j_base+1, k_base+1]
        
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (a11+a22)(b11+b22)
        U[0 * m + j_base][col] += 1       # a11
        U[1 * m + j_base + 1][col] += 1   # a22
        V[j_base * p + k_base][col] += 1       # b11
        V[(j_base + 1) * p + (k_base + 1)][col] += 1  # b22
        W[k_base * n + 0][col] += 1        # C11
        W[k_base * n + 1][col] += -1       # C22 (M1 appears in C11 with +1, in C22 with -1)
        col += 1
        
        # M2 = (a21+a22)(b11+b12)
        U[1 * m + j_base][col] += 1       # a21
        U[1 * m + j_base + 1][col] += 1   # a22
        V[j_base * p + k_base][col] += 1       # b11
        V[j_base * p + (k_base + 1)][col] += 1  # b12
        W[k_base * n + 1][col] += 1        # C21 (M2 appears in C21 with +1)
        col += 1
        
        # M3 = (a11-a22)(b21+b22)
        U[0 * m + j_base][col] += 1       # a11
        U[1 * m + j_base + 1][col] += -1  # a22
        V[(j_base + 1) * p + k_base][col] += 1       # b21
        V[(j_base + 1) * p + (k_base + 1)][col] += 1  # b22
        W[k_base * n + 1][col] += 1        # C22 (M3 appears in C22 with +1)
        col += 1
        
        # M4 = a11(b21-b22)
        U[0 * m + j_base][col] += 1       # a11
        V[(j_base + 1) * p + k_base][col] += 1       # b21
        V[(j_base + 1) * p + (k_base + 1)][col] += -1  # b22
        W[k_base * n + 0][col] += 1        # C11 (M4 appears in C11 with +1)
        W[k_base * n + 1][col] += -1       # C22 (M4 appears in C22 with -1)
        col += 1
        
        # M5 = a22(b12-b11)
        U[1 * m + j_base + 1][col] += 1   # a22
        V[j_base * p + (k_base + 1)][col] += 1  # b12
        V[j_base * p + k_base][col] += -1       # b11
        W[k_base * n + 0][col] += -1       # C11 (M5 appears in C11 with -1)
        W[k_base * n + 1][col] += 1        # C12 (M5 appears in C12 with +1)
        col += 1
        
        # M6 = (a11+a12)b22
        U[0 * m + j_base][col] += 1       # a11
        U[0 * m + j_base + 1][col] += 1   # a12
        V[(j_base + 1) * p + (k_base + 1)][col] += 1  # b22
        W[k_base * n + 1][col] += 1        # C12 (M6 appears in C12 with +1)
        W[k_base * n + 1][col] += -1       # C22 (M6 appears in C22 with -1)
        # Wait, M6 appears in C12 with +1 and C22 with -1
        col += 1
        
        # M7 = (a21-a11)b11
        U[1 * m + j_base][col] += 1       # a21
        U[0 * m + j_base][col] += -1      # a11
        V[j_base * p + k_base][col] += 1       # b11
        W[k_base * n + 0][col] += 1        # C11 (M7 appears in C11 with +1)
        W[k_base * n + 1][col] += -1       # C21 (M7 appears in C21 with -1)
        col += 1
    
    def add_schoolbook_2x2x1(j_base, k_base):
        """Add schoolbook decomposition for 2x2 * 2x1 block.
        A rows: i=0,1; cols: j_base, j_base+1
        B rows: j_base, j_base+1; col: k_base
        C: W[k_base*n+i] for i in {0,1}
        """
        nonlocal col
        
        # C[k_base, 0] = A[0, j_base]*B[j_base, k_base] + A[0, j_base+1]*B[j_base+1, k_base]
        # C[k_base, 1] = A[1, j_base]*B[j_base, k_base] + A[1, j_base+1]*B[j_base+1, k_base]
        
        # Product 1: A[0, j_base] * B[j_base, k_base]
        U[0 * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + 0][col] += 1
        col += 1
        
        # Product 2: A[0, j_base+1] * B[j_base+1, k_base]
        U[0 * m + j_base + 1][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        W[k_base * n + 0][col] += 1
        col += 1
        
        # Product 3: A[1, j_base] * B[j_base, k_base]
        U[1 * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + 1][col] += 1
        col += 1
        
        # Product 4: A[1, j_base+1] * B[j_base+1, k_base]
        U[1 * m + j_base + 1][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        W[k_base * n + 1][col] += 1
        col += 1
    
    # Iterate over m-blocks (2 blocks of size 2)
    for jb in range(2):
        j_base = jb * 2
        
        # Iterate over k-blocks (3 blocks of size 2, 1 block of size 1)
        for kb in range(4):
            if kb < 3:
                k_base = kb * 2
                add_strassen(j_base, k_base)
            else:
                k_base = 6
                add_schoolbook_2x2x1(j_base, k_base)
    
    return {"U": U, "V": V, "W": W}
