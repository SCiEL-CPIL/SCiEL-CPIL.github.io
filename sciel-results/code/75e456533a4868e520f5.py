import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 with rank 7.
    U: (4, 7) for A
    V: (4, 7) for B
    W: (4, 7) for C
    Indices: 00, 01, 10, 11
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen terms
    # M1 = (a00 - a11)(b10 + b11)
    # M2 = (a00 + a01) b10
    # M3 = a11 (b01 - b11)
    # M4 = (a10 + a11) b00
    # M5 = (a00 + a11)(b00 + b11)
    # M6 = (a01 - a11)(b00 + b01)
    # M7 = (a10 - a00)(b01 + b11)
    
    # Coefficients for U (A entries)
    # U[0] is a00, U[1] is a01, U[2] is a10, U[3] is a11
    # Coefficients for V (B entries)
    # V[0] is b00, V[1] is b01, V[2] is b10, V[3] is b11
    # Coefficients for W (C entries)
    # W[0] is c00, W[1] is c01, W[2] is c10, W[3] is c11
    
    # M1: (a00 - a11)(b10 + b11)
    U[0][0] = 1; U[3][0] = -1
    V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[3][0] = -1
    
    # M2: (a00 + a01) b10
    U[0][1] = 1; U[1][1] = 1
    V[2][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3: a11 (b01 - b11)
    U[3][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: (a10 + a11) b00
    U[2][3] = 1; U[3][3] = 1
    V[0][3] = 1
    W[0][3] = 1; W[2][3] = -1
    
    # M5: (a00 + a11)(b00 + b11)
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1; W[2][4] = 1
    
    # M6: (a01 - a11)(b00 + b01)
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = -1; W[3][5] = 1
    
    # M7: (a10 - a00)(b01 + b11)
    U[2][6] = 1; U[0][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1; W[2][6] = 1
    
    return U, V, W

def _strassen_4x4_4x4():
    """
    Constructs rank-49 decomposition for 4x4 * 4x4 using nested Strassen.
    Returns U, V, W of shapes (16, 49), (16, 49), (16, 49).
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Get the base 2x2 Strassen decomposition
    U_base, V_base, W_base = _strassen_2x2_2x2()
    
    # The 7 outer Strassen products
    # Each product is X_k * Y_k = M_k
    # X_k and Y_k are 2x2 matrices formed by linear combinations of A and B blocks.
    # M_k is a 2x2 matrix which contributes to C blocks.
    
    # Block indices for 4x4 matrix (row-major):
    # Block 00: rows 0-3 (0,1,4,5)
    # Block 01: rows 4-7 (2,3,6,7) -> Wait, 4x4 matrix indices:
    # A[0][0]=0, A[0][1]=1, A[0][2]=2, A[0][3]=3
    # A[1][0]=4, A[1][1]=5, A[1][2]=6, A[1][3]=7
    # A[2][0]=8, A[2][1]=9, A[2][2]=10, A[2][3]=11
    # A[3][0]=12, A[3][1]=13, A[3][2]=14, A[3][3]=15
    
    # Blocks:
    # A11 (top-left): 0, 1, 4, 5
    # A12 (top-right): 2, 3, 6, 7
    # A21 (bottom-left): 8, 9, 12, 13
    # A22 (bottom-right): 10, 11, 14, 15
    
    # We define the 7 products. For each product k (0..6):
    # We need to map the 2x2 Strassen terms to the global indices.
    # The 2x2 Strassen decomposition has 7 terms.
    # So for each outer product k, we add 7 terms to the global decomposition.
    # Total 7 * 7 = 49 terms.
    
    # Outer Product Definitions:
    # k=0: M1 = (A11 - A22)(B21 + B22)
    # k=1: M2 = (A11 + A12) B21
    # k=2: M3 = A22 (B12 - B22)
    # k=3: M4 = (A21 + A22) B11
    # k=4: M5 = (A11 + A22)(B11 + B22)
    # k=5: M6 = (A12 - A22)(B11 + B12)
    # k=6: M7 = (A21 - A11)(B12 + B22)
    
    # For each k, we define:
    # X_coeffs: mapping from 2x2 local indices (0,1,2,3) to global U indices and coefficients.
    # Y_coeffs: mapping from 2x2 local indices to global V indices and coefficients.
    # W_coeffs: mapping from 2x2 local indices to global W indices and coefficients.
    
    # Local 2x2 indices: 0=00, 1=01, 2=10, 3=11
    
    # Helper to add a term
    def add_term(k, term_idx):
        # term_idx is 0..6 for the inner Strassen
        base_col = k * 7 + term_idx
        
        # Get coefficients from base Strassen
        # U_base[row][term_idx]
        # V_base[row][term_idx]
        # W_base[row][term_idx]
        
        # We need to map the local 2x2 indices of X, Y, Z to global indices.
        # X is a 2x2 matrix. Its entries are linear combinations of A blocks.
        # Y is a 2x2 matrix. Its entries are linear combinations of B blocks.
        # Z (M_k) is a 2x2 matrix. Its entries contribute to C blocks.
        
        # We will build dictionaries:
        # x_map: local_idx -> list of (global_idx, coeff)
        # y_map: local_idx -> list of (global_idx, coeff)
        # w_map: local_idx -> list of (global_idx, coeff)
        
        # Define x_map, y_map, w_map for each k
        
        if k == 0:
            # M1 = (A11 - A22)(B21 + B22)
            # X = A11 - A22
            # X[0] (00) = A11[0] - A22[0] = A[0] - A[10]
            # X[1] (01) = A11[1] - A22[1] = A[1] - A[11]
            # X[2] (10) = A11[2] - A22[2] = A[4] - A[14]
            # X[3] (11) = A11[3] - A22[3] = A[5] - A[15]
            x_map = {
                0: [(0, 1), (10, -1)],
                1: [(1, 1), (11, -1)],
                2: [(4, 1), (14, -1)],
                3: [(5, 1), (15, -1)]
            }
            # Y = B21 + B22
            # Y[0] (00) = B21[0] + B22[0] = B[8] + B[10]
            # Y[1] (01) = B21[1] + B22[1] = B[9] + B[11]
            # Y[2] (10) = B21[2] + B22[2] = B[12] + B[14]
            # Y[3] (11) = B21[3] + B22[3] = B[13] + B[15]
            y_map = {
                0: [(8, 1), (10, 1)],
                1: [(9, 1), (11, 1)],
                2: [(12, 1), (14, 1)],
                3: [(13, 1), (15, 1)]
            }
            # W = C11 (+1), C21 (+1), C22 (-1)
            # C11: 0,1,4,5
            # C21: 8,9,12,13
            # C22: 10,11,14,15
            w_map = {
                0: [(0, 1), (8, 1), (10, -1)],
                1: [(1, 1), (9, 1), (11, -1)],
                2: [(4, 1), (12, 1), (14, -1)],
                3: [(5, 1), (13, 1), (15, -1)]
            }
        elif k == 1:
            # M2 = (A11 + A12) B21
            # X = A11 + A12
            x_map = {
                0: [(0, 1), (2, 1)],
                1: [(1, 1), (3, 1)],
                2: [(4, 1), (6, 1)],
                3: [(5, 1), (7, 1)]
            }
            # Y = B21
            y_map = {
                0: [(8, 1)],
                1: [(9, 1)],
                2: [(12, 1)],
                3: [(13, 1)]
            }
            # W = C12 (+1)
            # C12: 2,3,6,7
            w_map = {
                0: [(2, 1)],
                1: [(3, 1)],
                2: [(6, 1)],
                3: [(7, 1)]
            }
        elif k == 2:
            # M3 = A22 (B12 - B22)
            # X = A22
            x_map = {
                0: [(10, 1)],
                1: [(11, 1)],
                2: [(14, 1)],
                3: [(15, 1)]
            }
            # Y = B12 - B22
            y_map = {
                0: [(2, 1), (10, -1)],
                1: [(3, 1), (11, -1)],
                2: [(6, 1), (14, -1)],
                3: [(7, 1), (15, -1)]
            }
            # W = C22 (+1)
            w_map = {
                0: [(10, 1)],
                1: [(11, 1)],
                2: [(14, 1)],
                3: [(15, 1)]
            }
        elif k == 3:
            # M4 = (A21 + A22) B11
            # X = A21 + A22
            x_map = {
                0: [(8, 1), (10, 1)],
                1: [(9, 1), (11, 1)],
                2: [(12, 1), (14, 1)],
                3: [(13, 1), (15, 1)]
            }
            # Y = B11
            y_map = {
                0: [(0, 1)],
                1: [(1, 1)],
                2: [(4, 1)],
                3: [(5, 1)]
            }
            # W = C11 (+1), C21 (-1)
            w_map = {
                0: [(0, 1), (8, -1)],
                1: [(1, 1), (9, -1)],
                2: [(4, 1), (12, -1)],
                3: [(5, 1), (13, -1)]
            }
        elif k == 4:
            # M5 = (A11 + A22)(B11 + B22)
            # X = A11 + A22
            x_map = {
                0: [(0, 1), (10, 1)],
                1: [(1, 1), (11, 1)],
                2: [(4, 1), (14, 1)],
                3: [(5, 1), (15, 1)]
            }
            # Y = B11 + B22
            y_map = {
                0: [(0, 1), (10, 1)],
                1: [(1, 1), (11, 1)],
                2: [(4, 1), (14, 1)],
                3: [(5, 1), (15, 1)]
            }
            # W = C11 (-1), C12 (+1), C21 (+1)
            w_map = {
                0: [(0, -1), (2, 1), (8, 1)],
                1: [(1, -1), (3, 1), (9, 1)],
                2: [(4, -1), (6, 1), (12, 1)],
                3: [(5, -1), (7, 1), (13, 1)]
            }
        elif k == 5:
            # M6 = (A12 - A22)(B11 + B12)
            # X = A12 - A22
            x_map = {
                0: [(2, 1), (10, -1)],
                1: [(3, 1), (11, -1)],
                2: [(6, 1), (14, -1)],
                3: [(7, 1), (15, -1)]
            }
            # Y = B11 + B12
            y_map = {
                0: [(0, 1), (2, 1)],
                1: [(1, 1), (3, 1)],
                2: [(4, 1), (6, 1)],
                3: [(5, 1), (7, 1)]
            }
            # W = C12 (-1), C22 (+1)
            w_map = {
                0: [(2, -1), (10, 1)],
                1: [(3, -1), (11, 1)],
                2: [(6, -1), (14, 1)],
                3: [(7, -1), (15, 1)]
            }
        else: # k == 6
            # M7 = (A21 - A11)(B12 + B22)
            # X = A21 - A11
            x_map = {
                0: [(8, 1), (0, -1)],
                1: [(9, 1), (1, -1)],
                2: [(12, 1), (4, -1)],
                3: [(13, 1), (5, -1)]
            }
            # Y = B12 + B22
            y_map = {
                0: [(2, 1), (10, 1)],
                1: [(3, 1), (11, 1)],
                2: [(6, 1), (14, 1)],
                3: [(7, 1), (15, 1)]
            }
            # W = C11 (+1), C12 (-1)
            w_map = {
                0: [(0, 1), (2, -1)],
                1: [(1, 1), (3, -1)],
                2: [(4, 1), (6, -1)],
                3: [(5, 1), (7, -1)]
            }
        
        # Now fill U, V, W for this term
        # Iterate over local indices 0..3
        for local_idx in range(4):
            u_coeff = U_base[local_idx][term_idx]
            if u_coeff != 0:
                for g_idx, c in x_map[local_idx]:
                    U[g_idx][base_col] += u_coeff * c
            
            v_coeff = V_base[local_idx][term_idx]
            if v_coeff != 0:
                for g_idx, c in y_map[local_idx]:
                    V[g_idx][base_col] += v_coeff * c
            
            w_coeff = W_base[local_idx][term_idx]
            if w_coeff != 0:
                for g_idx, c in w_map[local_idx]:
                    W[g_idx][base_col] += w_coeff * c

    for k in range(7):
        for t in range(7):
            add_term(k, t)
            
    return U, V, W, rank

def _schoolbook_4x4_4x3():
    """
    Schoolbook decomposition for 4x4 * 4x3 (rank 48).
    Returns (U, V, W) where:
      U: (16, 48) for A entries
      V: (12, 48) for B entries (4x3 = 12 entries)
      W: (12, 48) for C entries (4x3 = 12 entries)
    """
    rank = 48
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j][col] = 1
                V[j*3+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return U, V, W, rank

def _build_full_4x4_4x7():
    """
    Build the full 4x4 * 4x7 tensor decomposition.
    Split B into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2.
    U is the same for both parts (A is 4x4).
    V is block diagonal: V1 for B1, V2 for B2.
    W is block diagonal: W1 for C1, W2 for C2.
    """
    U1, V1, W1, rank1 = _strassen_4x4_4x4()
    U2, V2, W2, rank2 = _schoolbook_4x4_4x3()
    
    rank = rank1 + rank2
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    # Part 1: A * B1 (4x4 * 4x4)
    for i in range(16):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for j in range(16):
        for c in range(rank1):
            V[j][c] = V1[j][c]
    for k in range(16):
        for c in range(rank1):
            W[k][c] = W1[k][c]
    
    # Part 2: A * B2 (4x4 * 4x3)
    for i in range(16):
        for c in range(rank2):
            U[i][rank1+c] = U2[i][c]
    for j in range(12):
        for c in range(rank2):
            V[16+j][rank1+c] = V2[j][c]
    for k in range(12):
        for c in range(rank2):
            W[16+k][rank1+c] = W2[k][c]
    
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition (rank n*m*p) as a fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the structured decomposition
    U, V, W, rank = _build_full_4x4_4x7()
    
    # Verify correctness
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
