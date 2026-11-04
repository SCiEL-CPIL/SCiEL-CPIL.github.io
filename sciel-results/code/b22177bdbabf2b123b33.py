"""Program for rank-minimal tensor decomposition of 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-wise Strassen decomposition for 4x4 matrix multiplication.
    
    Partition 4x4 matrices into 2x2 blocks of 2x2 sub-matrices.
    Apply Strassen's 7 multiplications at the block level.
    Each block multiplication is a 2x2 * 2x2 = 2x2 multiplication (rank 4 schoolbook).
    Total rank = 7 * 4 = 28.
    
    The output follows trace convention W[k*n + i, r] where the result is C[i,j] = sum_k A[i,k]*B[k,j].
    With n=m=p=4, C has shape (4,4), and W has shape (p*n, rank) = (16, rank).
    W[k*n + i, r] corresponds to output element C[i, j] where... let me re-examine.
    
    Actually, looking at the seed: W[k*n+i, c] is set to 1 for the schoolbook where
    A[i*m+j] * B[j*p+k] contributes to output. The output index in W is k*n+i.
    So W row index = k*n + i, which means the output is indexed by (k, i) where
    k is the column of B (which becomes column of C) and i is the row of A (row of C).
    So W[k*n + i] corresponds to C[i, k].
    
    For the Strassen approach:
    A is 4x4, partitioned into:
        A11 A12
        A21 A22
    where each is 2x2.
    
    Similarly B is 4x4:
        B11 B12
        B21 B22
    
    C = A*B is 4x4:
        C11 C12
        C21 C22
    
    Strassen:
        M1 = (A11 + A22) * (B11 + B22)
        M2 = (A21 + A22) * B11
        M3 = A11 * (B12 - B22)
        M4 = A22 * (B21 - B11)
        M5 = (A11 + A12) * B22
        M6 = (A21 - A11) * (B11 + B12)
        M7 = (A12 - A22) * (B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
    
    Each Mk is a 2x2 matrix product, requiring rank 4 (schoolbook for 2x2).
    So total rank = 7 * 4 = 28.
    
    Let me define the linear forms carefully.
    
    A entries: a[i,j] for i,j in 0..3
    B entries: b[i,j] for i,j in 0..3
    
    Block indexing:
    A11 = [[a00, a01], [a10, a11]]
    A12 = [[a02, a03], [a12, a13]]
    A21 = [[a20, a21], [a30, a31]]
    A22 = [[a22, a23], [a32, a33]]
    
    B11 = [[b00, b01], [b10, b11]]
    B12 = [[b02, b03], [b12, b13]]
    B21 = [[b20, b21], [b30, b31]]
    B22 = [[b22, b23], [b32, b33]]
    
    For each 2x2*2x2 multiplication, the 4 scalar multiplications are:
    For X * Y where X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]:
    Product P = [[x00*y00 + x01*y10, x00*y01 + x01*y11],
                 [x10*y00 + x11*y10, x10*y01 + x11*y11]]
    
    The 4 rank-1 terms are:
    c0 = x00 * y00 -> contributes to P[0,0]
    c1 = x00 * y01 -> contributes to P[0,1]
    c2 = x01 * y10 -> contributes to P[0,0]
    c3 = x01 * y11 -> contributes to P[0,1]
    c4 = x10 * y00 -> contributes to P[1,0]
    c5 = x10 * y01 -> contributes to P[1,1]
    c6 = x11 * y10 -> contributes to P[1,0]
    c7 = x11 * y11 -> contributes to P[1,1]
    
    Wait, that's 8 terms. The schoolbook for 2x2 is:
    P[0,0] = x00*y00 + x01*y10
    P[0,1] = x00*y01 + x01*y11
    P[1,0] = x10*y00 + x11*y10
    P[1,1] = x10*y01 + x11*y11
    
    So 8 scalar multiplications, each is a product of one entry of X and one of Y.
    But we can express this as rank 4? No, the rank of 2x2 matrix multiplication tensor is 7 (Strassen), but the schoolbook uses 8 or we can use 4 if we allow the factors to be linear combinations... 
    
    Wait, the rank of the 2x2 matrix multiplication tensor is actually 7 (Strassen proved lower bound 7, and it's been shown to be 7). But the "schoolbook" uses 8 multiplications. However, for our purposes, we need an exact decomposition.
    
    Actually, the standard schoolbook for 2x2 uses 8 scalar multiplications. But we can use a rank-7 Strassen decomposition for each 2x2 multiplication, giving total rank 7*7=49, which is worse than 28.
    
    Let me reconsider. The schoolbook for 2x2*2x2=2x2 requires 8 scalar multiplications (each being a product of one A entry and one B entry). So if we use schoolbook for each of the 7 Strassen block multiplications, total rank = 7 * 8 = 56.
    
    But wait, the original 4x4 schoolbook is rank 64 (4*4*4=64). So 56 is better than 64.
    
    Actually, I realize I made an error. Let me recount. For 2x2 matrix multiplication:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    That's 8 products, so rank 8 schoolbook.
    
    So 7 Strassen multiplications * 8 schoolbook multiplications each = 56 total rank.
    
    But we can do better. The rank of 2x2 matrix multiplication is 7, so 7*7=49.
    
    Hmm, but implementing Strassen within Strassen is complex. Let me just use the 8-term schoolbook for each block, giving rank 56.
    
    Actually, let me think about this differently. We can directly construct the rank-56 decomposition.
    
    Let me enumerate:
    For each of the 7 Strassen multiplications Mk = Xk * Yk where Xk and Yk are 2x2 linear combinations of A and B blocks.
    
    For each Mk, we have 8 scalar multiplications (schoolbook 2x2).
    
    Total: 7 * 8 = 56 terms.
    
    Let me build the U, V, W matrices.
    
    U has shape (n*m, rank) = (16, 56)
    V has shape (m*p, rank) = (16, 56)  
    W has shape (p*n, rank) = (16, 56)
    
    U[i*m + j, c] is the coefficient of A[i,j] in the c-th multiplication's X factor
    V[j*p + k, c] is the coefficient of B[j,k] in the c-th multiplication's Y factor
    W[k*n + i, c] is the coefficient that adds to C[i,k]
    
    Wait, I need to be more careful. Let me use indices:
    A is (n,m) = (4,4), B is (m,p) = (4,4), C is (n,p) = (4,4).
    
    U has (n*m) = 16 rows, indexed by (i,j) -> i*m+j
    V has (m*p) = 16 rows, indexed by (j,k) -> j*p+k
    W has (p*n) = 16 rows, indexed by (k,i) -> k*n+i, representing C[i,k]
    
    For each rank-1 term c, we have:
    U[i*m+j, c] * V[j'*p+k', c] contributes to W[k'*n+i, c]
    
    The sum over c of U[i*m+j, c] * V[j'*p+k', c] * W[k'*n+i, c] should equal C[i,k'] = sum_j A[i,j]*B[j,k']
    
    Let me construct this explicitly.
    
    Define the 7 Strassen inputs:
    X0 = A11 + A22
    X1 = A21 + A22
    X2 = A11
    X3 = A22
    X4 = A11 + A12
    X5 = A21 - A11
    X6 = A12 - A22
    
    Y0 = B11 + B22
    Y1 = B11
    Y2 = B12 - B22
    Y3 = B21 - B11
    Y4 = B22
    Y5 = B11 + B12
    Y6 = B21 + B22
    
    M0 = X0 * Y0 -> contributes to C11, C22
    M1 = X1 * Y1 -> contributes to C21, C22
    M2 = X2 * Y2 -> contributes to C11, C12
    M3 = X3 * Y3 -> contributes to C11, C21
    M4 = X4 * Y4 -> contributes to C11, C12
    M5 = X5 * Y5 -> contributes to C22
    M6 = X6 * Y6 -> contributes to C11, C22
    
    C11 = M0 + M3 - M4 + M6
    C12 = M2 + M4
    C21 = M1 + M3
    C22 = M0 - M1 + M2 + M5
    
    Now, each Mk is a 2x2 matrix. Let me write out each entry.
    
    For Mk = Xk * Yk where Xk = [[x00, x01], [x10, x11]] and Yk = [[y00, y01], [y10, y11]]:
    Mk[0,0] = x00*y00 + x01*y10
    Mk[0,1] = x00*y01 + x01*y11
    Mk[1,0] = x10*y00 + x11*y10
    Mk[1,1] = x10*y01 + x11*y11
    
    Now I need to express each x_ij and y_ij as linear combinations of A and B entries.
    
    Let me code this up by building the factor matrices directly.
    """
    n, m, p = DIMENSIONS
    # We'll build rank-56 decomposition (7 Strassen * 8 schoolbook per block)
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper: add a rank-1 term
    # A entry (i, j), B entry (j2, k), contributes to C[i, k]
    def add_term(a_i, a_j, b_j, b_k, c_i, c_k, sign=1):
        """Add a term: sign * A[a_i][a_j] * B[b_j][b_k] to C[c_i][c_k]"""
        nonlocal U, V, W, rank
        col = rank
        U[a_i * m + a_j][col] = sign
        V[b_j * p + b_k][col] = 1
        W[b_k * n + c_i][col] = 1
        # We'll handle this with pre-allocated lists and track columns
    
    # Actually, let me just build it directly with explicit enumeration.
    
    # Let me restructure: I'll create lists of (U_row, V_row, W_row, U_val, V_val, W_val) for each term.
    terms = []
    
    # Block definitions in terms of A indices (i,j) where i,j in 0..3
    # A11: (0,0),(0,1),(1,0),(1,1)
    # A12: (0,2),(0,3),(1,2),(1,3)
    # A21: (2,0),(2,1),(3,0),(3,1)
    # A22: (2,2),(2,3),(3,2),(3,3)
    
    # B11: (0,0),(0,1),(1,0),(1,1)
    # B12: (0,2),(0,3),(1,2),(1,3)
    # B21: (2,0),(2,1),(3,0),(3,1)
    # B22: (2,2),(2,3),(3,2),(3,3)
    
    # For each Strassen multiplication k (0-6), define Xk and Yk as 2x2 matrices
    # Each entry is a list of (a_i, a_j, coeff) for X, and (b_j, b_k, coeff) for Y
    
    # X0 = A11 + A22
    X0 = [
        [(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)],
        [(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]
    ]
    # Y0 = B11 + B22
    Y0 = [
        [(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)],
        [(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]
    ]
    # X1 = A21 + A22
    X1 = [
        [(2,0,1), (2,2,1)], [(2,1,1), (2,3,1)],
        [(3,0,1), (3,2,1)], [(3,1,1), (3,3,1)]
    ]
    # Y1 = B11
    Y1 = [
        [(0,0,1)], [(0,1,1)],
        [(1,0,1)], [(1,1,1)]
    ]
    # X2 = A11
    X2 = [
        [(0,0,1)], [(0,1,1)],
        [(1,0,1)], [(1,1,1)]
    ]
    # Y2 = B12 - B22
    Y2 = [
        [(0,2,1), (2,2,-1)], [(0,3,1), (2,3,-1)],
        [(1,2,1), (3,2,-1)], [(1,3,1), (3,3,-1)]
    ]
    # X3 = A22
    X3 = [
        [(2,2,1)], [(2,3,1)],
        [(3,2,1)], [(3,3,1)]
    ]
    # Y3 = B21 - B11
    Y3 = [
        [(2,0,1), (0,0,-1)], [(2,1,1), (0,1,-1)],
        [(3,0,1), (1,0,-1)], [(3,1,1), (1,1,-1)]
    ]
    # X4 = A11 + A12
    X4 = [
        [(0,0,1), (0,2,1)], [(0,1,1), (0,3,1)],
        [(1,0,1), (1,2,1)], [(1,1,1), (1,3,1)]
    ]
    # Y4 = B22
    Y4 = [
        [(2,2,1)], [(2,3,1)],
        [(3,2,1)], [(3,3,1)]
    ]
    # X5 = A21 - A11
    X5 = [
        [(2,0,1), (0,0,-1)], [(2,1,1), (0,1,-1)],
        [(3,0,1), (1,0,-1)], [(3,1,1), (1,1,-1)]
    ]
    # Y5 = B11 + B12
    Y5 = [
        [(0,0,1), (0,2,1)], [(0,1,1), (0,3,1)],
        [(1,0,1), (1,2,1)], [(1,1,1), (1,3,1)]
    ]
    # X6 = A12 - A22
    X6 = [
        [(0,2,1), (2,2,-1)], [(0,3,1), (2,3,-1)],
        [(1,2,1), (3,2,-1)], [(1,3,1), (3,3,-1)]
    ]
    # Y6 = B21 + B22
    Y6 = [
        [(2,0,1), (2,2,1)], [(2,1,1), (2,3,1)],
        [(3,0,1), (3,2,1)], [(3,1,1), (3,3,1)]
    ]
    
    X_list = [X0, X1, X2, X3, X4, X5, X6]
    Y_list = [Y0, Y1, Y2, Y3, Y4, Y5, Y6]
    
    # Output coefficients: which C blocks get which Mk with what sign
    # C11 = M0 + M3 - M4 + M6
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M1 + M2 + M5
    #
    # C11 is block (0,0): rows 0-1, cols 0-1
    # C12 is block (0,1): rows 0-1, cols 2-3
    # C21 is block (1,0): rows 2-3, cols 0-1
    # C22 is block (1,1): rows 2-3, cols 2-3
    #
    # For Mk contributing to Cblock(r,c) where r,c in {0,1}:
    # Mk[0,0] -> C[2r, 2c]
    # Mk[0,1] -> C[2r, 2c+1]
    # Mk[1,0] -> C[2r+1, 2c]
    # Mk[1,1] -> C[2r+1, 2c+1]
    
    # contributions[k] = list of (c_block_row, c_block_col, sign) for Mk
    contributions = [
        [(0,0,1), (1,1,1)],       # M0 -> C11 +, C22 +
        [(1,0,1), (1,1,-1)],      # M1 -> C21 +, C22 -
        [(0,0,1), (0,1,1)],       # M2 -> C11 +, C12 +
        [(0,0,1), (1,0,1)],       # M3 -> C11 +, C21 +
        [(0,0,-1), (0,1,1)],      # M4 -> C11 -, C12 +
        [(1,1,1)],                # M5 -> C22 +
        [(0,0,1), (1,1,1)]        # M6 -> C11 +, C22 +
    ]
    
    # Now build the terms
    col = 0
    for k in range(7):
        Xk = X_list[k]
        Yk = Y_list[k]
        # For each of the 8 scalar multiplications in the 2x2*2x2 product
        for xi in range(2):
            for yj in range(2):
                # Xk[xi, yj] * Yk[?, ?]
                # Product Mk[xi, ?] = sum_yj' Xk[xi, yj'] * Yk[yj', ?]
                # So for each xi, yj': term Xk[xi,yj'] * Yk[yj', yj_out]
                for yj_prime in range(2):
                    for yj_out in range(2):
                        # This is getting confused. Let me be explicit.
                        # Mk[xi, yj_out] = sum over yj_prime of Xk[xi, yj_prime] * Yk[yj_prime, yj_out]
                        # So each pair (xi, yj_prime, yj_out) gives a term.
                        # But that's 2*2*2=8 terms, which is what we want.
                        pass
        
        # Actually, the 8 terms for Mk are:
        # (0,0): X[0,0]*Y[0,0]
        # (0,1): X[0,0]*Y[0,1]
        # (0,0) also: X[0,1]*Y[1,0]
        # (0,1) also: X[0,1]*Y[1,1]
        # (1,0): X[1,0]*Y[0,0]
        # (1,1): X[1,0]*Y[0,1]
        # (1,0) also: X[1,1]*Y[1,0]
        # (1,1) also: X[1,1]*Y[1,1]
        
        # Let me enumerate all 8:
        mult_pairs = [
            (0,0,0,0),  # X[0,0]*Y[0,0] -> Mk[0,0]
            (0,0,0,1),  # X[0,0]*Y[0,1] -> Mk[0,1]
            (0,1,1,0),  # X[0,1]*Y[1,0] -> Mk[0,0]
            (0,1,1,1),  # X[0,1]*Y[1,1] -> Mk[0,1]
            (1,0,0,0),  # X[1,0]*Y[0,0] -> Mk[1,0]
            (1,0,0,1),  # X[1,0]*Y[0,1] -> Mk[1,1]
            (1,1,1,0),  # X[1,1]*Y[1,0] -> Mk[1,0]
            (1,1,1,1),  # X[1,1]*Y[1,1] -> Mk[1,1]
        ]
        
        for (xi, xj, yi, yj) in mult_pairs:
            # X entry is at position (xi, xj) in the 2x2 matrix Xk
            # Y entry is at position (yi, yj) in the 2x2 matrix Yk
            # Product contributes to Mk[xi, yj]
            
            # Get the linear forms for Xk[xi, xj] and Yk[yi, yj]
            x_form = Xk[xi][xj]  # list of (a_i, a_j, coeff)
            y_form = Yk[yi][yj]  # list of (b_j, b_k, coeff)
            
            # For each combination of terms in x_form and y_form
            for (ai, aj, cx) in x_form:
                for (bj, bk, cy) in y_form:
                    coeff = cx * cy
                    if coeff == 0:
                        continue
                    
                    # This term is: coeff * A[ai, aj] * B[bj, bk]
                    # It contributes to Mk[xi, yj]
                    # Now distribute Mk[xi, yj] to C blocks
                    
                    for (cbr, cbc, s) in contributions[k]:
                        # cbr, cbc: which C block (row block, col block)
                        # s: sign
                        # Mk[xi, yj] -> C[2*cbr + xi, 2*cbc + yj]
                        c_i = 2 * cbr + xi
                        c_k = 2 * cbc + yj
                        total_sign = coeff * s
                        
                        U[ai * m + aj][col] = total_sign
                        V[bj * p + bk][col] = 1
                        W[bk * n + c_i][col] = 1
                    
                    col += 1
    
    # Trim to actual rank used
    rank = col
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
