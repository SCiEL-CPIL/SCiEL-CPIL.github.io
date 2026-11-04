"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 intermediate products
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # Output:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    rank = 7
    U = [[0.0] * rank for _ in range(4)]  # 2*2 = 4 rows
    V = [[0.0] * rank for _ in range(4)]  # 2*2 = 4 rows
    W = [[0.0] * rank for _ in range(4)]  # 2*2 = 4 rows
    
    # Column 0: M1 = (a00+a11)(b00+b11)
    U[0][0] = 1.0  # a00
    U[3][0] = 1.0  # a11
    V[0][0] = 1.0  # b00
    V[3][0] = 1.0  # b11
    W[0][0] = 1.0  # c00
    W[3][0] = 1.0  # c11
    
    # Column 1: M2 = (a10+a11)*b00
    U[2][1] = 1.0  # a10
    U[3][1] = 1.0  # a11
    V[0][1] = 1.0  # b00
    W[2][1] = 1.0  # c10
    
    # Column 2: M3 = a00*(b01-b11)
    U[0][2] = 1.0  # a00
    V[1][2] = 1.0  # b01
    V[3][2] = -1.0 # b11
    W[1][2] = 1.0  # c01
    W[3][2] = 1.0  # c11
    
    # Column 3: M4 = a11*(b10-b00)
    U[3][3] = 1.0  # a11
    V[2][3] = 1.0  # b10
    V[0][3] = -1.0 # b00
    W[0][3] = 1.0  # c00
    W[2][3] = 1.0  # c10
    
    # Column 4: M5 = (a00+a01)*b11
    U[0][4] = 1.0  # a00
    U[1][4] = 1.0  # a01
    V[3][4] = 1.0  # b11
    W[0][4] = -1.0 # c00
    W[1][4] = 1.0  # c01
    
    # Column 5: M6 = (a10-a00)*(b00+b01)
    U[2][5] = 1.0  # a10
    U[0][5] = -1.0 # a00
    V[0][5] = 1.0  # b00
    V[1][5] = 1.0  # b01
    W[3][5] = 1.0  # c11
    
    # Column 6: M7 = (a01-a11)*(b10+b11)
    U[1][6] = 1.0  # a01
    U[3][6] = -1.0 # a11
    V[2][6] = 1.0  # b10
    V[3][6] = 1.0  # b11
    W[0][6] = 1.0  # c00
    W[3][6] = -1.0 # c11
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m times m x p.
    Returns (U, V, W) with shapes (n*m, n*m*p), (m*p, n*m*p), (p*n, n*m*p).
    """
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _block_decompose_5x5():
    """Decompose 5x5 matrix multiplication using block structure 2+3.
    Blocks:
      - A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : 2x2x2, Strassen rank 7
      - A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2] : 2x3x2, schoolbook rank 12
      - A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5] : 3x2x3, schoolbook rank 18
      - A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] : 2x2x3, schoolbook rank 12
      - A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5] : 3x3x3, schoolbook rank 27
      - A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5] : 2x3x3, schoolbook rank 18
      - A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2] : 3x2x2, schoolbook rank 12
      - A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2] : 3x3x2, schoolbook rank 18
    
    Total rank: 7 + 12 + 18 + 12 + 27 + 18 + 12 + 18 = 124? No wait.
    
    Let me reconsider. The 5x5 matrix multiplication tensor has:
    - Input A: 5x5 = 25 entries, indexed as (i, j) where i is row, j is col
    - Input B: 5x5 = 25 entries, indexed as (j, k) where j is row, k is col
    - Output C: 5x5 = 25 entries, indexed as (i, k)
    
    The blocks are:
    - A block (i1:i2, j1:j2), B block (j1:j2, k1:k2) contributes to C block (i1:i2, k1:k2)
    
    For 5 = 2 + 3:
    Row/col splits: first 2, next 3.
    
    A blocks: A[0:2,0:2], A[0:2,2:5], A[2:5,0:2], A[2:5,2:5]
    B blocks: B[0:2,0:2], B[0:2,2:5], B[2:5,0:2], B[2:5,2:5]
    
    Contribution (A_block, B_block) -> C_block where:
    A_block = A[i1:i2, j1:j2], B_block = B[j1:j2, k1:k2], C_block = C[i1:i2, k1:k2]
    
    Valid combinations (j must match):
    1. A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]: 2x2x2, rank 7 (Strassen)
    2. A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]: 2x2x3, rank 12 (schoolbook)
    3. A[0:2,2:5] * B[2:5,0:2] -> C[0:2,0:2]: 2x3x2, rank 12 (schoolbook)
    4. A[0:2,2:5] * B[2:5,2:5] -> C[0:2,2:5]: 2x3x3, rank 18 (schoolbook)
    5. A[2:5,0:2] * B[0:2,0:2] -> C[2:5,0:2]: 3x2x2, rank 12 (schoolbook)
    6. A[2:5,0:2] * B[0:2,2:5] -> C[2:5,2:5]: 3x2x3, rank 18 (schoolbook)
    7. A[2:5,2:5] * B[2:5,0:2] -> C[2:5,0:2]: 3x3x2, rank 18 (schoolbook)
    8. A[2:5,2:5] * B[2:5,2:5] -> C[2:5,2:5]: 3x3x3, rank 27 (schoolbook)
    
    Total rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    Hmm, that's only 1 less than 125. Let me double-check the indexing.
    
    Actually wait - for the W matrix, the convention is W[k*n+i, c] where the output is C[i,k].
    So W has p*n = 5*5 = 25 rows, indexed as k*n + i = k*5 + i.
    
    Let me be more careful. Let me just use schoolbook for all blocks and then replace the 2x2x2 block with Strassen.
    """
    n, m, p = 5, 5, 5
    total_rank = 0
    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25
    
    # We'll build the full U, V, W by summing contributions from each block
    # First, compute total rank needed
    blocks = [
        # (i1, i2, j1, j2, k1, k2, rank)
        (0, 2, 0, 2, 0, 2, 7),    # 2x2x2 Strassen
        (0, 2, 0, 2, 2, 5, 12),   # 2x2x3 schoolbook
        (0, 2, 2, 5, 0, 2, 12),   # 2x3x2 schoolbook
        (0, 2, 2, 5, 2, 5, 18),   # 2x3x3 schoolbook
        (2, 5, 0, 2, 0, 2, 12),   # 3x2x2 schoolbook
        (2, 5, 0, 2, 2, 5, 18),   # 3x2x3 schoolbook
        (2, 5, 2, 5, 0, 2, 18),   # 3x3x2 schoolbook
        (2, 5, 2, 5, 2, 5, 27),   # 3x3x3 schoolbook
    ]
    total_rank = sum(b[6] for b in blocks)
    
    U = [[0.0] * total_rank for _ in range(U_rows)]
    V = [[0.0] * total_rank for _ in range(V_rows)]
    W = [[0.0] * total_rank for _ in range(W_rows)]
    
    col_offset = 0
    for (i1, i2, j1, j2, k1, k2, block_rank) in blocks:
        ni = i2 - i1
        nj = j2 - j1
        nk = k2 - k1
        
        if ni == 2 and nj == 2 and nk == 2:
            # Use Strassen
            bU, bV, bW = _strassen_2x2()
            for local_i in range(ni):
                for local_j in range(nj):
                    gi = i1 + local_i
                    gj = j1 + local_j
                    gU_row = gi * m + gj
                    for c in range(block_rank):
                        U[gU_row][col_offset + c] += bU[local_i * nj + local_j][c]
            for local_j in range(nj):
                for local_k in range(nk):
                    gj = j1 + local_j
                    gk = k1 + local_k
                    gV_row = gj * p + gk
                    for c in range(block_rank):
                        V[gV_row][col_offset + c] += bV[local_j * nk + local_k][c]
            for local_k in range(nk):
                for local_i in range(ni):
                    gk = k1 + local_k
                    gi = i1 + local_i
                    gW_row = gk * n + gi
                    for c in range(block_rank):
                        W[gW_row][col_offset + c] += bW[local_k * ni + local_i][c]
        else:
            # Schoolbook for this block
            bU, bV, bW = _schoolbook_nm_p(ni, nj, nk)
            for local_i in range(ni):
                for local_j in range(nj):
                    gi = i1 + local_i
                    gj = j1 + local_j
                    gU_row = gi * m + gj
                    for c in range(block_rank):
                        U[gU_row][col_offset + c] += bU[local_i * nj + local_j][c]
            for local_j in range(nj):
                for local_k in range(nk):
                    gj = j1 + local_j
                    gk = k1 + local_k
                    gV_row = gj * p + gk
                    for c in range(block_rank):
                        V[gV_row][col_offset + c] += bV[local_j * nk + local_k][c]
            for local_k in range(nk):
                for local_i in range(ni):
                    gk = k1 + local_k
                    gi = i1 + local_i
                    gW_row = gk * n + gi
                    for c in range(block_rank):
                        W[gW_row][col_offset + c] += bW[local_k * ni + local_i][c]
        
        col_offset += block_rank
    
    return U, V, W

def construct():
    """Construct a tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with the decomposition factors.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Use block decomposition with Strassen for 2x2 blocks
    U, V, W = _block_decompose_5x5()
    
    if time.time() > deadline:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
