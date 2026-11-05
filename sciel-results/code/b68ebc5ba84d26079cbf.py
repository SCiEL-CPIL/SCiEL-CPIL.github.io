"""Strassen block decomposition for 5x5x5 matrix multiplication."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _block_decompose(n, m, p):
    """
    Return (U, V, W) for n x m @ m x p -> n x p.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: W[k*n + i, r]
    """
    if m == 1:
        # Rank 1: A[i,0] * B[0,p] -> C[i,p]
        U = [[1] for _ in range(n * m)]  # n*1 entries, each [1]
        V = [[1] for _ in range(m * p)]  # 1*p entries, each [1]
        W = [[1] for _ in range(p * n)]  # p*n entries, each [1]
        return U, V, W, 1

    if n == 1 and p == 1:
        # Rank 1
        U = [[1] for _ in range(n * m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p * n)]
        return U, V, W, 1

    if n == 1:
        # Row vector times matrix: rank p
        # U[i*m+j] has 1 at position j, V[j*p+k] has 1 at position k, W[k*n+i]=W[k] has 1 at position k
        rank = p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for j in range(m):
            for k in range(p):
                U[j][k] = 1
                V[j*p + k][k] = 1
                W[k][k] = 1
        return U, V, W, rank

    if p == 1:
        # Matrix times column vector: rank n
        rank = n
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                U[i*m + j][i] = 1
                V[j*p][i] = 1
                W[i][i] = 1
        return U, V, W, rank

    # General case: use Strassen if n,m,p are all even, or fall back to schoolbook
    # For now, handle the 2x2x2 case with Strassen, and schoolbook otherwise
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    if n == 1 and m == 1:
        # 1x1 @ 1xp -> rank p
        rank = p
        U = [[1]*rank for _ in range(1)]
        V = [[1]*rank for _ in range(p)]
        W = [[1]*rank for _ in range(p)]
        return U, V, W, rank
    if n == 1:
        rank = p
        U = [[0]*rank for _ in range(m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p)]
        for j in range(m):
            for k in range(p):
                U[j][k] = 1
                V[j*p+k][k] = 1
                W[k][k] = 1
        return U, V, W, rank

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank


def _strassen_2x2x2():
    """
    Strassen rank-7 decomposition for 2x2x2.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M5 + M6
    C[1,0] = M3 + M4
    C[1,1] = M1 - M3 + M6 + M7
    """
    # U rows: a00, a01, a10, a11
    # V rows: b00, b01, b10, b11
    # W rows: c00 (k=0,i=0), c10 (k=1,i=0), c01 (k=0,i=1), c11 (k=1,i=1)
    # Wait: W[k*n+i, r] with n=2, so W[0]=c00, W[1]=c10, W[2]=c01, W[3]=c11
    
    U = [
        [1, 0, 0, 1, 1, -1, 0],   # a00: M1, M4(no), M5, M6(no), M7(no)
        [0, 1, 1, 0, 1, 0, 1],    # a01: M2, M3, M5, M7
        [0, 0, 1, 1, 0, 1, 0],    # a10: M3, M4, M6
        [1, 0, 0, 1, 0, 0, -1],   # a11: M1, M4, M7
    ]
    
    # Let me recompute carefully.
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10+b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U (rows = a00, a01, a10, a11):
    # M1: a00=1, a11=1 -> [1, 0, 0, 1]
    # M2: a10=1, a11=1 -> [0, 0, 1, 1]
    # M3: a00=1 -> [1, 0, 0, 0]
    # M4: a11=1 -> [0, 0, 0, 1]
    # M5: a00=1, a01=1 -> [1, 1, 0, 0]
    # M6: a10=1, a00=-1 -> [-1, 0, 1, 0]
    # M7: a01=1, a11=-1 -> [0, 1, 0, -1]
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1],   # a11
    ]
    
    # V (rows = b00, b01, b10, b11):
    # M1: b00=1, b11=1 -> [1, 0, 0, 1]
    # M2: b00=1 -> [1, 0, 0, 0]
    # M3: b10=1, b11=1 -> [0, 0, 1, 1]
    # M4: b01=1, b00=-1 -> [-1, 1, 0, 0]
    # M5: b11=1 -> [0, 0, 0, 1]
    # M6: b00=1, b01=1 -> [1, 1, 0, 0]
    # M7: b10=1, b11=1 -> [0, 0, 1, 1]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00
        [0, 0, 0, 1, 0, 1, 0],    # b01
        [0, 0, 1, 0, 0, 0, 1],    # b10
        [1, 0, 1, 0, 1, 0, 1],    # b11
    ]
    
    # W (rows = c00, c10, c01, c11) where c_ij is C[i,j]
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M3 + M4
    # C[0,1] = M5 + M6
    # C[1,1] = M1 - M3 + M6 + M7
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 1, 0, 0, 0],    # c10
        [0, 0, 0, 0, 1, 1, 0],    # c01
        [1, 0, -1, 0, 0, 1, 1],   # c11
    ]
    
    return U, V, W, 7


def _combine_blocks(n, m, p, blocks):
    """
    Combine block decompositions into a single decomposition.
    blocks: list of (n_i, m_j, p_k, U_ij, V_jk, W_ki)
    """
    total_rank = sum(b[3] for b in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for (n_i, m_j, p_k, U_ij, V_jk, W_ki, rank) in blocks:
        for i in range(n_i):
            for j in range(m_j):
                for r in range(rank):
                    U[i*m + j][offset + r] = U_ij[i*m_j + j][r]
        for j in range(m_j):
            for k in range(p_k):
                for r in range(rank):
                    V[j*p + k][offset + r] = V_jk[j*p_k + k][r]
        for k in range(p_k):
            for i in range(n_i):
                for r in range(rank):
                    W[k*n + i][offset + r] = W_ki[k*n_i + i][r]
        offset += rank
    
    return U, V, W, total_rank


def _block_decompose_full(n, m, p):
    """
    Full block decomposition for n x m @ m x p.
    Partition into 2x2x2 blocks where possible, use Strassen for those,
    schoolbook for the remainder.
    """
    # Determine block structure
    # n = 2*2 + 1, m = 2*2 + 1, p = 2*2 + 1
    # Blocks: (2,2,2) x 8, (2,2,1) x 2, (2,1,2) x 2, (2,1,1) x 1,
    #         (1,2,2) x 2, (1,2,1) x 2, (1,1,2) x 2, (1,1,1) x 1
    
    blocks = []
    
    # 8 blocks of size (2,2,2)
    for i_blk in range(2):
        for j_blk in range(2):
            for k_blk in range(2):
                U_ij, V_jk, W_ki, rank = _strassen_2x2x2()
                blocks.append((2, 2, 2, U_ij, V_jk, W_ki, rank))
    
    # 2 blocks of size (2,2,1): i in 0-3, j in 0-3, k in 4
    for i_blk in range(2):
        for j_blk in range(2):
            U_ij, V_jk, W_ki, rank = _block_decompose(2, 2, 1)
            blocks.append((2, 2, 1, U_ij, V_jk, W_ki, rank))
    
    # 2 blocks of size (2,1,2): i in 0-3, j in 4, k in 0-3
    for i_blk in range(2):
        for k_blk in range(2):
            U_ij, V_jk, W_ki, rank = _block_decompose(2, 1, 2)
            blocks.append((2, 1, 2, U_ij, V_jk, W_ki, rank))
    
    # 1 block of size (2,1,1): i in 0-3, j in 4, k in 4
    U_ij, V_jk, W_ki, rank = _block_decompose(2, 1, 1)
    blocks.append((2, 1, 1, U_ij, V_jk, W_ki, rank))
    
    # 2 blocks of size (1,2,2): i in 4, j in 0-3, k in 0-3
    for j_blk in range(2):
        for k_blk in range(2):
            U_ij, V_jk, W_ki, rank = _block_decompose(1, 2, 2)
            blocks.append((1, 2, 2, U_ij, V_jk, W_ki, rank))
    
    # 2 blocks of size (1,2,1): i in 4, j in 0-3, k in 4
    for j_blk in range(2):
        U_ij, V_jk, W_ki, rank = _block_decompose(1, 2, 1)
        blocks.append((1, 2, 1, U_ij, V_jk, W_ki, rank))
    
    # 2 blocks of size (1,1,2): i in 4, j in 4, k in 0-3
    for k_blk in range(2):
        U_ij, V_jk, W_ki, rank = _block_decompose(1, 1, 2)
        blocks.append((1, 1, 2, U_ij, V_jk, W_ki, rank))
    
    # 1 block of size (1,1,1): i in 4, j in 4, k in 4
    U_ij, V_jk, W_ki, rank = _block_decompose(1, 1, 1)
    blocks.append((1, 1, 1, U_ij, V_jk, W_ki, rank))
    
    return _combine_blocks(n, m, p, blocks)


def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _block_decompose_full(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
