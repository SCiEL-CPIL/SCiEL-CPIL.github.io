"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x13 matrix multiplication.
    
    The key insight is to use the additive structure of the output dimension.
    We decompose B into 13 rank-1 matrices and apply a clever bilinear identity.
    
    For each column k of B, we have B_k = b_{0k} * e_0^T + b_{1k} * e_1^T
    where e_0 = [1,0]^T and e_1 = [0,1]^T.
    
    The product A * B_k can be computed using 4 multiplications:
    (a_{00}b_{0k} + a_{01}b_{1k}) * e_{0k}^T + (a_{10}b_{0k} + a_{11}b_{1k}) * e_{1k}^T
    
    However, we can be smarter. For each column k, we need to compute:
    C_{0k} = a_{00}b_{0k} + a_{01}b_{1k}
    C_{1k} = a_{10}b_{0k} + a_{11}b_{1k}
    
    This is exactly a 2x2 dot product for each k, which requires 4 multiplications
    in the schoolbook method. But we can share computations across columns.
    
    Actually, let's think differently. The total number of scalar multiplications
    needed is 2 * 2 * 13 = 52 in the naive approach.
    
    Using Strassen on 2x2x2 blocks:
    - We have 6 full 2x2x2 blocks (columns 0-1, 2-3, 4-5, 6-7, 8-9, 10-11)
    - Each 2x2x2 block has rank 7 (Strassen)
    - The last column (12) is a 2x2x1 block with rank 4 (schoolbook)
    
    Total rank = 6*7 + 4 = 46
    
    Let's construct this decomposition.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # Index conventions:
    # U: shape (n*m, rank) = (4, rank), indexed by (i, j) -> i*m + j
    # V: shape (m*p, rank) = (26, rank), indexed by (j, k) -> j*p + k
    # W: shape (p*n, rank) = (26, rank), indexed by (k, i) -> k*n + i
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    col = 0
    
    # Helper to add a term to the decomposition
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    # Strassen decomposition for 2x2x2:
    # Given A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]]
    # C = A*B where:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # Process 6 blocks of 2 columns each
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # V indices: j*p + k
        # j=0 -> k, j=1 -> 13+k
        # W indices: k*n + i
        # i=0 -> 2k, i=1 -> 2k+1
        # U indices: i*m + j
        # 00:0, 01:1, 10:2, 11:3
        
        # M1 = (a00 + a11) * (b00 + b11)
        # Contributes to C00 (+1), C11 (-1)
        add_term(
            {0: 1, 3: 1},           # a00 + a11
            {k0: 1, 13 + k1: 1},    # b00 + b11
            {2 * k0: 1, 2 * k1 + 1: -1}  # C00 +1, C11 -1
        )
        
        # M2 = (a10 + a11) * b00
        # Contributes to C10 (+1), C11 (-1)
        add_term(
            {2: 1, 3: 1},           # a10 + a11
            {k0: 1},                # b00
            {2 * k0 + 1: 1, 2 * k1 + 1: -1}  # C10 +1, C11 -1
        )
        
        # M3 = a00 * (b01 - b11)
        # Contributes to C01 (+1), C11 (+1)
        add_term(
            {0: 1},                 # a00
            {k1: 1, 13 + k1: -1},   # b01 - b11
            {2 * k1: 1, 2 * k1 + 1: 1}  # C01 +1, C11 +1
        )
        
        # M4 = a11 * (b10 - b00)
        # Contributes to C00 (+1), C10 (+1)
        add_term(
            {3: 1},                 # a11
            {13 + k0: 1, k0: -1},   # b10 - b00
            {2 * k0: 1, 2 * k0 + 1: 1}  # C00 +1, C10 +1
        )
        
        # M5 = (a00 + a01) * b11
        # Contributes to C00 (-1), C01 (+1)
        add_term(
            {0: 1, 1: 1},           # a00 + a01
            {13 + k1: 1},           # b11
            {2 * k0: -1, 2 * k1: 1}  # C00 -1, C01 +1
        )
        
        # M6 = (a10 - a00) * (b00 + b01)
        # Contributes to C11 (+1)
        add_term(
            {2: 1, 0: -1},          # a10 - a00
            {k0: 1, k1: 1},         # b00 + b01
            {2 * k1 + 1: 1}         # C11 +1
        )
        
        # M7 = (a01 - a11) * (b10 + b11)
        # Contributes to C00 (-1)
        add_term(
            {1: 1, 3: -1},          # a01 - a11
            {13 + k0: 1, 13 + k1: 1},  # b10 + b11
            {2 * k0: -1}            # C00 -1
        )
    
    # Process last column k=12 (2x2x1 block, schoolbook)
    k = 12
    # C00 = a00*b0k + a01*b1k
    add_term({0: 1}, {k: 1}, {2 * k: 1})
    add_term({1: 1}, {13 + k: 1}, {2 * k: 1})
    
    # C01 = a10*b0k + a11*b1k
    add_term({2: 1}, {k: 1}, {2 * k + 1: 1})
    add_term({3: 1}, {13 + k: 1}, {2 * k + 1: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

