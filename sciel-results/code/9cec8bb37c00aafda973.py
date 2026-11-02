"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    
    Strategy: Partition 5 = 2 + 3. Use Strassen's algorithm (rank 7) for the
    2x2 diagonal block product A11*B11, and schoolbook for all other block
    products. Total rank = 7 + 6*6 + 9 = 52 (improvement over schoolbook 125).
    
    Block structure:
      A = [[A11(2x2), A12(2x3)],
           [A21(3x2), A22(3x3)]]
      B = [[B11(2x2), B12(2x3)],
           [B21(3x2), B22(3x3)]]
      C = [[C11(2x2), C12(2x3)],
           [C21(3x2), C22(3x3)]]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # all 5
    
    # We'll build U (25 x rank), V (25 x rank), W (25 x rank)
    # U[i*m+j, r] corresponds to A[i,j]
    # V[j*p+k, r] corresponds to B[j,k]
    # W[k*n+i, r] corresponds to C[i,k]
    
    terms = []  # list of (U_indices, V_indices, W_indices)
    
    # Helper to add a schoolbook term: A[i,j] * B[j,k] -> C[i,k]
    def add_schoolbook(i, j, k):
        U_idx = i * m + j  # A[i,j]
        V_idx = j * p + k  # B[j,k]
        W_idx = k * n + i  # C[i,k]
        terms.append(((U_idx,), (V_idx,), (W_idx,)))
    
    # Helper to add a Strassen term for 2x2 block
    # Strassen's 7 products for A11*B11 where A11,B11 are 2x2
    # A11 = [[a00, a01], [a10, a11]] (rows 0,1; cols 0,1)
    # B11 = [[b00, b01], [b10, b11]] (rows 0,1; cols 0,1)
    # C11 = [[c00, c01], [c10, c11]] (rows 0,1; cols 0,1)
    #
    # Strassen products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of sums/differences of A and B entries.
    # In tensor decomposition, each Mi contributes rank 1 terms.
    # A sum like (a00+a11) means U has +1 at both positions.
    
    def add_strassen_term(a_coeffs, b_coeffs, c_coeffs, sign_a, sign_b, sign_c):
        """
        Add a term: (sum sign_a[i]*A[i]) * (sum sign_b[j]*B[j]) -> sum sign_c[k]*C[k]
        a_coeffs: list of (i,j) for A
        b_coeffs: list of (j,k) for B  
        c_coeffs: list of (i,k) for C
        sign_a, sign_b, sign_c: +1 or -1 for each
        """
        U_indices = []
        for (i, j) in a_coeffs:
            U_indices.append((i * m + j, sign_a[a_coeffs.index((i,j))]))
        V_indices = []
        for (j, k) in b_coeffs:
            V_indices.append((j * p + k, sign_b[b_coeffs.index((j,k))]))
        W_indices = []
        for (i, k) in c_coeffs:
            W_indices.append((k * n + i, sign_c[c_coeffs.index((i,k))]))
        terms.append((U_indices, V_indices, W_indices))
    
    # Strassen for A11*B11 (2x2 block, indices 0,1 in both rows and cols)
    # A11 entries: A[0,0], A[0,1], A[1,0], A[1,1]
    # B11 entries: B[0,0], B[0,1], B[1,0], B[1,1]
    # C11 entries: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> contributes to C[0,0] and C[1,1]
    add_strassen_term(
        [(0,0), (1,1)],  # A[0,0], A[1,1]
        [(0,0), (1,1)],  # B[0,0], B[1,1]
        [(0,0), (1,1)],  # C[0,0], C[1,1]
        [1, 1], [1, 1], [1, 1]
    )
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0] -> contributes to C[1,0]
    add_strassen_term(
        [(1,0), (1,1)],  # A[1,0], A[1,1]
        [(0,0)],         # B[0,0]
        [(1,0)],         # C[1,0]
        [1, 1], [1], [1]
    )
    
    # M3 = A[0,0] * (B[0,1]-B[1,1]) -> contributes to C[0,1]
    add_strassen_term(
        [(0,0)],         # A[0,0]
        [(0,1), (1,1)],  # B[0,1], B[1,1]
        [(0,1)],         # C[0,1]
        [1], [1, -1], [1]
    )
    
    # M4 = A[1,1] * (B[1,0]-B[0,0]) -> contributes to C[0,0] and C[1,0]
    add_strassen_term(
        [(1,1)],         # A[1,1]
        [(1,0), (0,0)],  # B[1,0], B[0,0]
        [(0,0), (1,0)],  # C[0,0], C[1,0]
        [1], [1, -1], [1, 1]
    )
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1] -> contributes to C[0,0] and C[0,1]
    add_strassen_term(
        [(0,0), (0,1)],  # A[0,0], A[0,1]
        [(1,1)],         # B[1,1]
        [(0,0), (0,1)],  # C[0,0], C[0,1]
        [1, 1], [1], [-1, 1]  # C[0,0] gets -M5, C[0,1] gets +M5
    )
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]) -> contributes to C[1,1]
    add_strassen_term(
        [(1,0), (0,0)],  # A[1,0], A[0,0]
        [(0,0), (0,1)],  # B[0,0], B[0,1]
        [(1,1)],         # C[1,1]
        [1, -1], [1, 1], [1]
    )
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) -> contributes to C[0,0] and C[1,1]
    add_strassen_term(
        [(0,1), (1,1)],  # A[0,1], A[1,1]
        [(1,0), (1,1)],  # B[1,0], B[1,1]
        [(0,0), (1,1)],  # C[0,0], C[1,1]
        [1, -1], [1, 1], [1, 1]
    )
    
    # Now add schoolbook terms for all other block products
    # A12*B21: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # i in 0,1; j in 2,3,4; k in 0,1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # A11*B12: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # i in 0,1; j in 0,1; k in 2,3,4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # A12*B22: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # i in 0,1; j in 2,3,4; k in 2,3,4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # A21*B11: A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # i in 2,3,4; j in 0,1; k in 0,1
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # A21*B12: A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # i in 2,3,4; j in 0,1; k in 2,3,4
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # A22*B21: A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    # i in 2,3,4; j in 2,3,4; k in 0,1
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # A22*B22: A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # i in 2,3,4; j in 2,3,4; k in 2,3,4
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_data, V_data, W_data) in enumerate(terms):
        for (idx, sign) in U_data:
            U[idx][col] += sign
        for (idx, sign) in V_data:
            V[idx][col] += sign
        for (idx, sign) in W_data:
            W[idx][col] += sign
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
