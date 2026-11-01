"""Rewritten program for matrix multiplication 2,2,9 tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Use Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    # combined with the 9 columns of B.
    # For each column pair (2j, 2j+1) of B, we can use Strassen to compute
    # the 2x2 product A * B[:, 2j:2j+2] with 7 multiplications.
    # Total rank = 7 * 4 = 28 (for the 4 pairs of columns, since 9 = 2*4 + 1)
    # Wait, 9 columns means we have 4 pairs and 1 single column.
    # For the 4 pairs, we use Strassen (7 mults each) = 28.
    # For the last single column, we need 4 multiplications (2x2 * 1x1 vector).
    # Total rank = 28 + 4 = 32.
    
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]  # shape (4, 32)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (18, 32)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (18, 32)
    
    # Strassen's algorithm for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(h-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each pair of columns (2j, 2j+1) where j = 0, 1, 2, 3:
    for j in range(4):
        base = j * 7
        # Column indices in B: 2j and 2j+1
        # Row indices in A: 0, 1, 2, 3 (corresponding to a, b, c, d)
        # Output indices in W: k*n + i where k is column index, i is row index
        
        # M1 = (a+d)(e+g)
        U[0][base + 0] += 1.0  # a
        U[3][base + 0] += 1.0  # d
        V[2*j][base + 0] += 1.0  # e (column 2j, row 0 of B)
        V[2*j + 1][base + 0] += 1.0  # g (column 2j+1, row 0 of B)
        # C00 += M1, C11 += M1
        W[0 * n + 0][base + 0] += 1.0  # C00
        W[1 * n + 1][base + 0] += 1.0  # C11
        
        # M2 = (c+d)e
        U[2][base + 1] += 1.0  # c
        U[3][base + 1] += 1.0  # d
        V[2*j][base + 1] += 1.0  # e
        # C10 += M2, C11 -= M2
        W[0 * n + 1][base + 1] += 1.0  # C10
        W[1 * n + 1][base + 1] -= 1.0  # C11
        
        # M3 = a(f-g)
        U[0][base + 2] += 1.0  # a
        V[2*j + 1][base + 2] += 1.0  # f (column 2j+1, row 0 of B)
        V[2*j + 1][base + 2] -= 1.0  # -g (column 2j+1, row 1 of B)
        # Wait, V is indexed by j*p + k where j is row of A (0 or 1), k is column of B.
        # Let me reconsider the indexing.
        # A is 2x2: rows 0,1; cols 0,1.
        # B is 2x9: rows 0,1; cols 0..8.
        # V has shape (m*p, rank) = (2*9, rank) = (18, rank).
        # V[j*p + k] corresponds to B[j, k].
        # So for column pair (2j, 2j+1), the columns are 2j and 2j+1.
        # e = B[0, 2j], f = B[1, 2j], g = B[0, 2j+1], h = B[1, 2j+1].
        
        # Let me redo this with correct indexing.
        pass
    
    # Reset and redo with correct indexing
    U = [[0.0] * rank for _ in range(n * m)]  # shape (4, 32)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (18, 32)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (18, 32)
    
    for j in range(4):
        base = j * 7
        # Column indices in B: col0 = 2j, col1 = 2j + 1
        col0 = 2 * j
        col1 = 2 * j + 1
        
        # e = B[0, col0], f = B[1, col0], g = B[0, col1], h = B[1, col1]
        # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        
        # M1 = (a+d)(e+g)
        U[0][base + 0] += 1.0  # a
        U[3][base + 0] += 1.0  # d
        V[0 * p + col0][base + 0] += 1.0  # e = B[0, col0]
        V[0 * p + col1][base + 0] += 1.0  # g = B[0, col1]
        # C00 += M1, C11 += M1
        W[col0 * n + 0][base + 0] += 1.0  # C[0, col0]
        W[col1 * n + 1][base + 0] += 1.0  # C[1, col1]
        
        # M2 = (c+d)e
        U[2][base + 1] += 1.0  # c
        U[3][base + 1] += 1.0  # d
        V[0 * p + col0][base + 1] += 1.0  # e
        # C10 += M2, C11 -= M2
        W[col0 * n + 1][base + 1] += 1.0  # C[1, col0]
        W[col1 * n + 1][base + 1] -= 1.0  # C[1, col1]
        
        # M3 = a(f-g)
        U[0][base + 2] += 1.0  # a
        V[1 * p + col0][base + 2] += 1.0  # f = B[1, col0]
        V[1 * p + col1][base + 2] -= 1.0  # -g = -B[1, col1]
        # C01 += M3, C11 += M3
        W[col0 * n + 0][base + 2] += 1.0  # C[0, col0]
        W[col1 * n + 1][base + 2] += 1.0  # C[1, col1]
        
        # M4 = d(h-f)
        U[3][base + 3] += 1.0  # d
        V[1 * p + col1][base + 3] += 1.0  # h = B[1, col1]
        V[1 * p + col0][base + 3] -= 1.0  # -f = -B[1, col0]
        # C00 += M4, C10 += M4
        W[col0 * n + 0][base + 3] += 1.0  # C[0, col0]
        W[col0 * n + 1][base + 3] += 1.0  # C[1, col0]
        
        # M5 = (a+b)h
        U[0][base + 4] += 1.0  # a
        U[1][base + 4] += 1.0  # b
        V[1 * p + col1][base + 4] += 1.0  # h
        # C00 -= M5, C01 += M5
        W[col0 * n + 0][base + 4] -= 1.0  # C[0, col0]
        W[col0 * n + 0][base + 4] += 1.0  # C[0, col0] -- wait, C01 is C[0, col1]
        W[col1 * n + 0][base + 4] += 1.0  # C[0, col1]
        
        # M6 = (c-a)(e+f)
        U[2][base + 5] += 1.0  # c
        U[0][base + 5] -= 1.0  # -a
        V[0 * p + col0][base + 5] += 1.0  # e
        V[1 * p + col0][base + 5] += 1.0  # f
        # C11 += M6
        W[col1 * n + 1][base + 5] += 1.0  # C[1, col1]
        
        # M7 = (b-d)(g+h)
        U[1][base + 6] += 1.0  # b
        U[3][base + 6] -= 1.0  # -d
        V[0 * p + col1][base + 6] += 1.0  # g
        V[1 * p + col1][base + 6] += 1.0  # h
        # C00 += M7
        W[col0 * n + 0][base + 6] += 1.0  # C[0, col0]
    
    # For the last column (col 8), use schoolbook (4 multiplications)
    base = 4 * 7  # 28
    col = 8
    # C[0, col] = a*e + b*g where e = B[0, col], g = B[1, col]
    # C[1, col] = c*e + d*g
    
    U[0][base + 0] += 1.0  # a
    V[0 * p + col][base + 0] += 1.0  # e
    W[col * n + 0][base + 0] += 1.0  # C[0, col]
    
    U[1][base + 1] += 1.0  # b
    V[1 * p + col][base + 1] += 1.0  # g
    W[col * n + 0][base + 1] += 1.0  # C[0, col]
    
    U[2][base + 2] += 1.0  # c
    V[0 * p + col][base + 2] += 1.0  # e
    W[col * n + 1][base + 2] += 1.0  # C[1, col]
    
    U[3][base + 3] += 1.0  # d
    V[1 * p + col][base + 3] += 1.0  # g
    W[col * n + 1][base + 3] += 1.0  # C[1, col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
