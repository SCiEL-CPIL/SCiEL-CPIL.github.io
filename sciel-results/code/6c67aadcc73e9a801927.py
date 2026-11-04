"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-style rank-19 decomposition for 3x3 matrix multiplication.
    # U has shape (n*m, rank) = (9, 19), V has shape (m*p, rank) = (9, 19),
    # W has shape (p*n, rank) = (9, 19).
    # W[k*n+i, r] is the coefficient of C_{i,k} from the r-th rank-1 term.
    n, m, p = 3, 3, 3
    rank = 19

    # Each rank-1 term is (u, v, w) where:
    #   u: vector of length 9 (A indices: row i, col j -> index i*3+j)
    #   v: vector of length 9 (B indices: row j, col k -> index j*3+k)
    #   w: vector of length 9 (C indices: row i, col k -> index k*3+i per convention W[k*n+i])
    #
    # The 19 terms (using the known optimal 3x3 decomposition):
    # Indices for A: a00=0, a01=1, a02=2, a10=3, a11=4, a12=5, a20=6, a21=7, a22=8
    # Indices for B: b00=0, b01=1, b02=2, b10=3, b11=4, b12=5, b20=6, b21=7, b22=8
    # Indices for C (W index = k*3 + i): 
    #   C00: k=0,i=0 -> 0; C01: k=1,i=0 -> 3; C02: k=2,i=0 -> 6
    #   C10: k=0,i=1 -> 1; C11: k=1,i=1 -> 4; C12: k=2,i=1 -> 7
    #   C20: k=0,i=2 -> 2; C21: k=1,i=2 -> 5; C22: k=2,i=2 -> 8

    terms = [
        # Term 1: A00 * B00 -> C00
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0]),

        # Term 2: A01 * B10 -> C00 (wait, A01*B10 gives C[0][0]? No: C[i][k] = sum_j A[i][j]B[j][k]
        # A[0][1]*B[1][0] contributes to C[0][0]. So C index for C00 is 0.
        ([0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0]),

        # Actually, let me use the well-known 19-term decomposition.
        # I'll construct it carefully.
    ]

    # Let me use a verified 19-term decomposition for 3x3 matrix multiplication.
    # Using the approach from the literature, the 19 terms are:

    terms = [
        # 1
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0]),  # a00*b00 -> C00

        # 2
        ([0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0]),  # a01*b10 -> C00

        # 3
        ([0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0]),  # a02*b20 -> C00

        # 4
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0]),  # a00*b01 -> C01

        # 5
        ([0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0]),  # a01*b11 -> C01

        # 6
        ([0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0]),  # a02*b21 -> C01

        # 7
        ([1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0]),  # a00*b02 -> C02

        # 8
        ([0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0]),  # a01*b12 -> C02

        # 9
        ([0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 0, 0]),  # a02*b22 -> C02

        # 10
        ([0, 0, 0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0]),  # a10*b00 -> C10

        # 11
        ([0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0]),  # a11*b10 -> C10

        # 12
        ([0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0]),  # a20*b20 -> C10

        # 13
        ([0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0]),  # a10*b01 -> C11

        # 14
        ([0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0]),  # a11*b11 -> C11

        # 15
        ([0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0]),  # a20*b21 -> C11

        # 16
        ([0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0]),  # a10*b02 -> C12

        # 17
        ([0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0]),  # a11*b12 -> C12

        # 18
        ([0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 0]),  # a20*b22 -> C12

        # 19
        ([0, 0, 0, 0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0]),  # a12*b00 -> C20
    ]

    # Wait, I only have 19 terms but they look like just the schoolbook terms with some rearrangement.
    # This is NOT a rank-19 decomposition; it's still rank 19 but these are just 19 of the 27 schoolbook terms,
    # which would NOT correctly compute the full matrix product.
    
    # Let me reconsider. The schoolbook is rank 27. A true rank-19 decomposition must use 
    # linear combinations, not just subsets of the 27 elementary terms.
    
    # Since I don't have a verified explicit rank-19 decomposition memorized with integer coefficients,
    # and the directive asks for the Strassen 1968 result (which was for 2x2, rank 7) or a 3x3 rank 19,
    # let me fall back to a correct decomposition. The minimal known rank for 3x3 over reals is not 
    # definitively 19 in a simple hardcoded form I can verify here.
    
    # Actually, the problem states the seed is rank 27 and wants improvement. Let me at least return
    # a valid decomposition. If I can't guarantee rank 19 with integer coefficients correctly,
    # I should return a correct but possibly higher-rank decomposition, or the schoolbook.
    
    # However, the directive explicitly asks for rank-19. Let me try to use a known construction.
    
    # I'll use the schoolbook (rank 27) as a safe fallback since I cannot verify a rank-19 
    # decomposition with integer coefficients from memory.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
