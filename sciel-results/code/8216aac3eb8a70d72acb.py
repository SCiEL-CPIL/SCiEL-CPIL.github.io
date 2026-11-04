import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_block_indices(n, block_r, block_c):
    """
    Get flattened indices for a block in an n x n matrix.
    block_r: 0 for top, 1 for bottom.
    block_c: 0 for left, 1 for right.
    Returns list of row indices in the flattened matrix.
    """
    half = n // 2
    start_r = block_r * half
    start_c = block_c * half
    indices = []
    for i in range(half):
        for j in range(half):
            indices.append((start_r + i) * n + (start_c + j))
    return indices

def _strassen_4x4():
    """
    Construct 4x4x4 decomposition using Strassen recursively.
    Returns U, V, W of shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    n = 4
    half = 2
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block indices for n=4
    # A blocks
    A11 = _get_block_indices(n, 0, 0)
    A12 = _get_block_indices(n, 0, 1)
    A21 = _get_block_indices(n, 1, 0)
    A22 = _get_block_indices(n, 1, 1)
    
    # B blocks
    B11 = _get_block_indices(n, 0, 0)
    B12 = _get_block_indices(n, 0, 1)
    B21 = _get_block_indices(n, 1, 0)
    B22 = _get_block_indices(n, 1, 1)
    
    # C blocks
    C11 = _get_block_indices(n, 0, 0)
    C12 = _get_block_indices(n, 0, 1)
    C21 = _get_block_indices(n, 1, 0)
    C22 = _get_block_indices(n, 1, 1)
    
    # Terms definition:
    # (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (indices, coeff)
    # B_blocks: list of (indices, coeff)
    # C_blocks: list of (indices, coeff)
    
    terms = [
        # M1
        ([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)], [(C11, 1), (C22, 1)]),
        # M2
        ([(A21, 1), (A22, 1)], [(B11, 1)], [(C21, 1)]),
        # M3
        ([(A11, 1)], [(B21, 1), (B22, -1)], [(C12, 1)]),
        # M4
        ([(A22, 1)], [(B12, 1), (B11, -1)], [(C11, 1), (C21, 1)]),
        # M5
        ([(A11, 1), (A12, 1)], [(B22, 1)], [(C11, -1), (C12, 1)]),
        # M6
        ([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)], [(C22, 1)]),
        # M7
        ([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)], [(C11, 1)]),
    ]
    
    for t, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        start_col = t * 7
        for c in range(7):
            col = start_col + c
            
            # U
            for indices, coeff in a_blocks:
                # U2 row i corresponds to A[i//2, i%2]
                # We need to add U2[:, c] * coeff to U4[indices, col]
                # But U2 is 4x7. indices are 4 values.
                # U2[0] -> A11[0,0], U2[1] -> A11[0,1], U2[2] -> A11[1,0], U2[3] -> A11[1,1]
                # indices list is ordered: [r0c0, r0c1, r1c0, r1c1]
                # So we can just add element-wise.
                U4[indices, col] += coeff * U2[:, c]
                
            # V
            for indices, coeff in b_blocks:
                V4[indices, col] += coeff * V2[:, c]
                
            # W
            for indices, coeff in c_blocks:
                W4[indices, col] += coeff * W2[:, c]
                
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Partition 5 = 4 + 1
    n1, n2 = 4, 1
    m1, m2 = 4, 1
    p1, p2 = 4, 1
    
    # Get 4x4x4 Strassen
    U4, V4, W4 = _strassen_4x4()
    
    # Total rank
    rank_444 = 49
    rank_414 = 4 * 1 * 4 # 16
    rank_441 = 4 * 4 * 1 # 16
    rank_411 = 4 * 1 * 1 # 4
    rank_144 = 1 * 4 * 4 # 16
    rank_114 = 1 * 1 * 4 # 4
    rank_141 = 1 * 4 * 1 # 4
    rank_111 = 1 * 1 * 1 # 1
    
    total_rank = rank_444 + rank_414 + rank_441 + rank_411 + rank_144 + rank_114 + rank_141 + rank_111
    # 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col = 0
    
    # Helper to add schoolbook block
    def add_schoolblock(u_rows, v_rows, w_rows, rank):
        nonlocal col
        for i in range(len(u_rows)):
            for j in range(len(v_rows)):
                # Wait, schoolbook for rectangular blocks
                # U shape (n*m, rank), V (m*p, rank), W (p*n, rank)
                # For A(n1 x m1) B(m1 x p1) -> C(n1 x p1)
                # u_rows: indices in A
                # v_rows: indices in B
                # w_rows: indices in C
                # rank = len(u_rows) * len(v_rows) ? No.
                # If A is 4x1, B is 1x4.
                # u_rows has 4 elements. v_rows has 4 elements.
                # Product is 4x4.
                # Schoolbook rank is 4*1*4 = 16.
                # Iteration:
                # for i in 0..3 (A row)
                #   for k in 0..3 (B col)
                #     term i, k
                #     U[u_rows[i], col] = 1
                #     V[v_rows[k], col] = 1  # v_rows[k] corresponds to B[0, k]?
                #     W[w_rows[i*4+k], col] = 1?
                
                # This generic helper is tricky because dimensions vary.
                # Let's just hardcode the 7 terms.
                pass
        col += rank

    # Term 1: A11 B11 (4x4 x 4x4) -> C11 (4x4)
    # U4, V4, W4 are 16x49
    # Map to global
    # A11 rows: 0..15
    # B11 rows: 0..15
    # C11 rows: 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18
    U[0:16, col:col+49] = U4
    V[0:16, col:col+49] = V4
    c11_indices = [i*5+j for i in range(4) for j in range(4)]
    # W4 is 16x49, rows 0..15 correspond to C11 local indices 0..15
    # We need to scatter W4 into W[c11_indices, col:col+49]
    W[c11_indices, col:col+49] = W4
    col += 49
    
    # Term 2: A12 B21 (4x1 x 1x4) -> C11 (4x4)
    # A12: rows 4,9,14,19
    # B21: rows 20,21,22,23
    # C11: same as above
    a12_rows = [4, 9, 14, 19]
    b21_rows = [20, 21, 22, 23]
    for i in range(4):
        for k in range(4):
            U[a12_rows[i], col] = 1
            V[b21_rows[k], col] = 1
            W[c11_indices[i*4+k], col] = 1
            col += 1
            
    # Term 3: A11 B12 (4x4 x 4x1) -> C12 (4x1)
    # A11: 0..15
    # B12: 4,9,14,19
    # C12: 4,9,14,19
    b12_rows = [4, 9, 14, 19]
    c12_rows = [4, 9, 14, 19]
    for i in range(4):
        for j in range(4):
            # A[i, j] * B[j, 0] -> C[i, 0]
            U[i*4+j, col] = 1
            V[b12_rows[j], col] = 1
            W[c12_rows[i], col] = 1
            col += 1
            
    # Term 4: A12 B22 (4x1 x 1x1) -> C12 (4x1)
    # A12: 4,9,14,19
    # B22: 24
    # C12: 4,9,14,19
    b22_row = 24
    for i in range(4):
        U[a12_rows[i], col] = 1
        V[b22_row, col] = 1
        W[c12_rows[i], col] = 1
        col += 1
        
    # Term 5: A21 B11 (1x4 x 4x4) -> C21 (1x4)
    # A21: 20,21,22,23
    # B11: 0..15
    # C21: 20,21,22,23
    a21_rows = [20, 21, 22, 23]
    c21_rows = [20, 21, 22, 23]
    for j in range(4):
        for k in range(4):
            # A[0, j] * B[j, k] -> C[0, k]
            U[a21_rows[j], col] = 1
            V[j*4+k, col] = 1
            W[c21_rows[k], col] = 1
            col += 1
            
    # Term 6: A22 B21 (1x1 x 1x4) -> C21 (1x4)
    # A22: 24
    # B21: 20,21,22,23
    # C21: 20,21,22,23
    for k in range(4):
        U[b22_row, col] = 1
        V[b21_rows[k], col] = 1
        W[c21_rows[k], col] = 1
        col += 1
        
    # Term 7: A21 B12 (1x4 x 4x1) -> C22 (1x1)
    # A21: 20,21,22,23
    # B12: 4,9,14,19
    # C22: 24
    for j in range(4):
        U[a21_rows[j], col] = 1
        V[b12_rows[j], col] = 1
        W[b22_row, col] = 1
        col += 1
        
    # Term 8: A22 B22 (1x1 x 1x1) -> C22 (1x1)
    U[b22_row, col] = 1
    V[b22_row, col] = 1
    W[b22_row, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
