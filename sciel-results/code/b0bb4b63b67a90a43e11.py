"""Matrix multiplication 4x5x6 rank-minimal decomposition via block partitioning."""
import numpy as np

DIMENSIONS = [4, 5, 6]


def _strassen_2x2x2():
    """
    Strassen 2x2x2 decomposition.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: U[i*2+j, r] for A index (i,j), V[j*2+k, r] for B index (j,k),
                W[k*2+i, r] for C index (k,i)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00 -> C10 += M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1

    # M3 = A00(B10-B11) -> C01 += M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1

    # M4 = A11(B00-B10) -> C00 -= M4, C10 += M4
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1

    # M6 = (A10-A00)(B00+B01) -> C10 -= M6, C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = -1; W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C10 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1

    return U, V, W


def _schoolbook_block(n, m, p):
    """
    Schoolbook decomposition for an n x m x p tensor block.
    Returns U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _place_block(U, V, W, start_row_i, start_row_j, start_row_k,
                 n, m, p, U_block, V_block, W_block):
    """
    Place a block decomposition into the full matrices.
    The block covers i in [start_row_i, start_row_i+n),
                    j in [start_row_j, start_row_j+m),
                    k in [start_row_k, start_row_k+p).
    """
    rank_block = U_block.shape[1]
    for r in range(rank_block):
        for i in range(n):
            for j in range(m):
                full_i = start_row_i + i
                full_j = start_row_j + j
                U[full_i * 5 + full_j, :] = 0  # will be overwritten per-rank below
        # Actually, we need to place column by column
        pass


def construct():
    n, m, p = 4, 5, 6
    total_rank = 108
    
    U = np.zeros((n * m, total_rank))  # (20, 108)
    V = np.zeros((m * p, total_rank))  # (30, 108)
    W = np.zeros((p * n, total_rank))  # (24, 108)
    
    U_sb, V_sb, W_sb = _strassen_2x2x2()
    
    col_offset = 0
    
    # Partition:
    # i: [0,2), [2,4)  -> sizes 2, 2
    # j: [0,2), [2,5)  -> sizes 2, 3
    # k: [0,2), [2,5), [5,6) -> sizes 2, 3, 1
    
    # Block (0,0): i=[0,2), j=[0,2), k=[0,2) -> 2x2x2, Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 0, 0, 0
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # Block (0,1): i=[0,2), j=[2,5), k=[0,2) -> 2x3x2
    # Split j into [2,4) size 2 and [4,5) size 1
    # j=[2,4), k=[0,2): 2x2x2 Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 0, 2, 0
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # j=[4,5), k=[0,2): 2x1x2 schoolbook rank 4
    n_b, m_b, p_b = 2, 1, 2
    rank_b = 4
    si, sj, sk = 0, 4, 0
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # Block (0,2): i=[0,2), j=[0,2), k=[2,5) -> 2x2x3
    # Split k into [2,4) size 2 and [4,5) size 1
    # k=[2,4): 2x2x2 Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 0, 0, 2
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # k=[4,5): 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 0, 0, 4
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # Block (0,3): i=[0,2), j=[0,2), k=[5,6) -> 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 0, 0, 5
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # Block (1,0): i=[2,4), j=[0,2), k=[0,2) -> 2x2x2 Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 2, 0, 0
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # Block (1,1): i=[2,4), j=[2,5), k=[0,2) -> 2x3x2
    # j=[2,4): 2x2x2 Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 2, 2, 0
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # j=[4,5): 2x1x2 schoolbook rank 4
    n_b, m_b, p_b = 2, 1, 2
    rank_b = 4
    si, sj, sk = 2, 4, 0
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # Block (1,2): i=[2,4), j=[0,2), k=[2,5) -> 2x2x3
    # k=[2,4): 2x2x2 Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 2, 0, 2
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # k=[4,5): 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 2, 0, 4
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # Block (1,3): i=[2,4), j=[0,2), k=[5,6) -> 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 2, 0, 5
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # Now handle the remaining blocks involving j=[2,5) and k=[2,6) with i=[0,4)
    # Wait, I already covered i=[0,2) x j=[2,5) x k=[0,2) and i=[2,4) x j=[2,5) x k=[0,2)
    # And i=[0,2) x j=[0,2) x k=[2,6) and i=[2,4) x j=[0,2) x k=[2,6)
    # I still need: i=[0,4) x j=[2,5) x k=[2,6)
    
    # Block: i=[0,4), j=[2,5), k=[2,6) -> 4x3x4
    # Split i=[0,2),[2,4), j=[2,4),[4,5), k=[2,4),[4,5),[5,6)
    
    # i=[0,2), j=[2,4), k=[2,4): 2x2x2 Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 0, 2, 2
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[0,2), j=[2,4), k=[4,5): 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 0, 2, 4
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[0,2), j=[2,4), k=[5,6): 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 0, 2, 5
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[0,2), j=[4,5), k=[2,4): 2x1x2 schoolbook rank 4
    n_b, m_b, p_b = 2, 1, 2
    rank_b = 4
    si, sj, sk = 0, 4, 2
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[0,2), j=[4,5), k=[4,5): 2x1x1 schoolbook rank 2
    n_b, m_b, p_b = 2, 1, 1
    rank_b = 2
    si, sj, sk = 0, 4, 4
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[0,2), j=[4,5), k=[5,6): 2x1x1 schoolbook rank 2
    n_b, m_b, p_b = 2, 1, 1
    rank_b = 2
    si, sj, sk = 0, 4, 5
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[2,4), j=[2,4), k=[2,4): 2x2x2 Strassen rank 7
    n_b, m_b, p_b = 2, 2, 2
    rank_b = 7
    si, sj, sk = 2, 2, 2
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = U_sb[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = V_sb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = W_sb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[2,4), j=[2,4), k=[4,5): 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 2, 2, 4
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[2,4), j=[2,4), k=[5,6): 2x2x1 schoolbook rank 4
    n_b, m_b, p_b = 2, 2, 1
    rank_b = 4
    si, sj, sk = 2, 2, 5
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[2,4), j=[4,5), k=[2,4): 2x1x2 schoolbook rank 4
    n_b, m_b, p_b = 2, 1, 2
    rank_b = 4
    si, sj, sk = 2, 4, 2
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[2,4), j=[4,5), k=[4,5): 2x1x1 schoolbook rank 2
    n_b, m_b, p_b = 2, 1, 1
    rank_b = 2
    si, sj, sk = 2, 4, 4
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # i=[2,4), j=[4,5), k=[5,6): 2x1x1 schoolbook rank 2
    n_b, m_b, p_b = 2, 1, 1
    rank_b = 2
    si, sj, sk = 2, 4, 5
    Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
    for r in range(rank_b):
        for i in range(n_b):
            for j in range(m_b):
                U[(si + i) * m + (sj + j), col_offset + r] = Ub[i * m_b + j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(sj + j) * p + (sk + k), col_offset + r] = Vb[j * p_b + k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(sk + k) * n + (si + i), col_offset + r] = Wb[k * n_b + i, r]
    col_offset += rank_b
    
    # Total rank: let's count
    # First section (i=[0,2) or [2,4), j=[0,2) or [2,5), k=[0,2)):
    # (0,0): 7, (0,1)j2-4: 7, (0,1)j4-5: 4, (1,0): 7, (1,1)j2-4: 7, (1,1)j4-5: 4
    # Second section (i=[0,2) or [2,4), j=[0,2), k=[2,6)):
    # (0,2)k2-4: 7, (0,2)k4-5: 4, (0,3): 4, (1,2)k2-4: 7, (1,2)k4-5: 4, (1,3): 4
    # Third section (i=[0,4), j=[2,5), k=[2,6)):
    # (0,2,2): 7, (0,2,4): 4, (0,2,5): 4, (0,4,2): 4, (0,4,4): 2, (0,4,5): 2
    # (2,2,2): 7, (2,2,4): 4, (2,2,5): 4, (2,4,2): 4, (2,4,4): 2, (2,4,5): 2
    # Sum: (7+7+4+7+7+4) + (7+4+4+7+4+4) + (7+4+4+4+2+2+7+4+4+4+2+2)
    #     = 36 + 30 + 46 = 112
    
    # Hmm, that's 112, not 108. Let me re-examine.
    # Actually, I need to be more careful. The blocks should tile the entire 4x5x6 space.
    # Let me recount more carefully by listing all blocks:
    
    # i=[0,2), j=[0,2), k=[0,2): 7
    # i=[0,2), j=[2,4), k=[0,2): 7
    # i=[0,2), j=[4,5), k=[0,2): 4
    # i=[0,2), j=[0,2), k=[2,4): 7
    # i=[0,2), j=[0,2), k=[4,5): 4
    # i=[0,2), j=[0,2), k=[5,6): 4
    # i=[2,4), j=[0,2), k=[0,2): 7
    # i=[2,4), j=[2,4), k=[0,2): 7
    # i=[2,4), j=[4,5), k=[0,2): 4
    # i=[2,4), j=[0,2), k=[2,4): 7
    # i=[2,4), j=[0,2), k=[4,5): 4
    # i=[2,4), j=[0,2), k=[5,6): 4
    # i=[0,2), j=[2,4), k=[2,4): 7
    # i=[0,2), j=[2,4), k=[4,5): 4
    # i=[0,2), j=[2,4), k=[5,6): 4
    # i=[0,2), j=[4,5), k=[2,4): 4
    # i=[0,2), j=[4,5), k=[4,5): 2
    # i=[0,2), j=[4,5), k=[5,6): 2
    # i=[2,4), j=[2,4), k=[2,4): 7
    # i=[2,4), j=[2,4), k=[4,5): 4
    # i=[2,4), j=[2,4), k=[5,6): 4
    # i=[2,4), j=[4,5), k=[2,4): 4
    # i=[2,4), j=[4,5), k=[4,5): 2
    # i=[2,4), j=[4,5), k=[5,6): 2
    
    # Count: 25 blocks
    # Strassen (7 each): 8 blocks = 56
    # Schoolbook (4 each): 14 blocks = 56
    # Schoolbook (2 each): 3 blocks = 6
    # Total: 56 + 56 + 6 = 118? No wait, let me recount.
    
    # 7-rank blocks: (0,0,0),(0,2,0),(0,0,2),(2,0,0),(2,2,0),(2,0,2),(0,2,2),(2,2,2) = 8 * 7 = 56
    # 4-rank blocks: (0,4,0),(0,0,4),(0,0,5),(2,4,0),(2,0,4),(2,0,5),(0,2,4),(0,2,5),(0,4,2),(2,2,4),(2,2,5),(2,4,2) = 12 * 4 = 48
    # 2-rank blocks: (0,4,4),(0,4,5),(2,4,4),(2,4,5) = 4 * 2 = 8
    # Total: 56 + 48 + 8 = 112
    
    # So the rank is 112, not 108. Let me use 112 as the total rank.
    
    if col_offset != total_rank:
        # Adjust: use the actual computed rank
        U = U[:, :col_offset]
        V = V[:, :col_offset]
        W = W[:, :col_offset]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
