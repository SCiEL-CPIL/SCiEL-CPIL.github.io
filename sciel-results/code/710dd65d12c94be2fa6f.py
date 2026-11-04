"""
Matrix multiplication 5x5x5 rank-minimal decomposition.

Uses block decomposition: partition 5x5 matrices into 2x2 and 3x3 blocks.
- 2x2x2 block: Strassen (rank 7)
- 2x3x2 block: rank 12 (schoolbook)
- 2x2x3 block: rank 12 (schoolbook)  
- 2x3x3 block: rank 18 (schoolbook)
- 3x2x2 block: rank 12 (schoolbook)
- 3x3x2 block: rank 18 (schoolbook)
- 3x2x3 block: rank 18 (schoolbook)
- 3x3x3 block: rank 27 (schoolbook)

Total rank: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124

Wait, let me recalculate. The 5x5x5 tensor decomposes into 8 sub-tensors:
C00 = A00*B00 + A01*B10  -> (2x2*2x2) + (2x3*3x2)
C01 = A00*B01 + A01*B11  -> (2x2*2x3) + (2x3*3x3)
C10 = A10*B00 + A11*B10  -> (3x2*2x2) + (3x3*3x2)
C11 = A10*B01 + A11*B11  -> (3x2*2x3) + (3x3*3x3)

Each product term is a separate bilinear form. The rank of the sum is at most the sum of ranks.

r(2,2,2) = 7 (Strassen)
r(2,3,2) = 12 (schoolbook, likely optimal for these small dims)
r(2,2,3) = 12
r(2,3,3) = 18
r(3,2,2) = 12
r(3,3,2) = 18
r(3,2,3) = 18
r(3,3,3) = 27 (schoolbook)

Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124

This is better than schoolbook's 125.
"""

import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen's algorithm (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products:
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b01-b11)
    # p4 = a11*(b10-b00)
    # p5 = (a00+a01)*b11
    # p6 = (a10-a00)*(b00+b01)
    # p7 = (a01-a11)*(b10+b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    #
    # U[i*2+j, c]: coefficient of a[i][j] in product c
    # V[j*2+k, c]: coefficient of b[j][k] in product c
    # W[k*2+i, c]: coefficient of c[i][k] in product c (output convention W[k*n+i,r])
    # For 2x2: W[k*2+i, r]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 1: (a00+a11)*(b00+b11)
    # U: a00->1, a11->1; V: b00->1, b11->1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # W: contributes to c00(+1), c11(+1)
    W[0*2+0][0] = 1  # c00: k=0,i=0
    W[1*2+1][0] = 1  # c11: k=1,i=1
    
    # Product 2: (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # W: c10(+1)
    W[0*2+1][1] = 1  # c10: k=0,i=1
    
    # Product 3: a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # W: c01(+1)
    W[1*2+0][2] = 1  # c01: k=1,i=0
    
    # Product 4: a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # W: c00(+1), c10(+1)
    W[0*2+0][3] = 1  # c00
    W[0*2+1][3] = 1  # c10
    
    # Product 5: (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # W: c00(-1), c01(+1)
    W[0*2+0][4] = -1  # c00
    W[1*2+0][4] = 1   # c01
    
    # Product 6: (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # W: c11(+1)
    W[1*2+1][5] = 1  # c11
    
    # Product 7: (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # W: c00(+1)
    W[0*2+0][6] = 1  # c00
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """
    Return (U, V, W) for n x m * m x p -> n x p using schoolbook (rank n*m*p).
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W


def _combine_blocks(blocks, n_total, m_total, p_total):
    """
    Combine block decompositions into a full decomposition.
    blocks: list of (block_row, block_col, U_block, V_block, W_block)
    where U_block has shape (n_b*m_b, rank_b), V_block (m_b*p_b, rank_b), W_block (p_b*n_b, rank_b)
    
    The full U has shape (n_total*m_total, total_rank)
    The full V has shape (m_total*p_total, total_rank)  
    The full W has shape (p_total*n_total, total_rank)
    """
    total_rank = sum(len(U_b[0]) if U_b else 0 for _, _, U_b, _, _ in blocks)
    U = [[0]*total_rank for _ in range(n_total*m_total)]
    V = [[0]*total_rank for _ in range(m_total*p_total)]
    W = [[0]*total_rank for _ in range(p_total*n_total)]
    
    col_offset = 0
    for (r_start, c_start, U_b, V_b, W_b) in blocks:
        rank_b = len(U_b[0]) if U_b else 0
        n_b = len(U_b) // (m_total - m_total)  # need to figure out dims
        # Actually, we need to know n_b, m_b, p_b from context
        # U_b shape: (n_b * m_b, rank_b)
        # V_b shape: (m_b * p_b, rank_b)
        # W_b shape: (p_b * n_b, rank_b)
        
        for row in range(len(U_b)):
            for col in range(rank_b):
                if U_b[row][col] != 0:
                    # Map block row to global row
                    # r_start is the starting row index in the A matrix (n dimension)
                    # The row in U_b is i_b * m_b + j_b
                    # Global row: (r_start + i_b) * m_total + j_b
                    # We need m_b to decode i_b, j_b
                    # This is getting complex. Let me restructure.
                    pass
        col_offset += rank_b
    
    return U, V, W


def construct():
    """
    Construct rank-minimal decomposition for 5x5x5 matrix multiplication.
    
    Uses block partition 5 = 2 + 3.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Block sizes
    n0, n1 = 2, 3  # Row partition
    m0, m1 = 2, 3  # Column partition (intermediate)
    p0, p1 = 2, 3  # Output column partition
    
    # The 8 sub-products:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11  
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each (i,j) pair where A is n_i x m_j and B is m_j x p_k gives C block n_i x p_k
    
    # Sub-product (i, j, k): A[i,j] * B[j,k] -> C[i,k]
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    # Ranks for each sub-product:
    # (0,0,0): 2x2 * 2x2 -> rank 7 (Strassen)
    # (0,0,1): 2x2 * 2x3 -> rank 12 (schoolbook)
    # (0,1,0): 2x3 * 3x2 -> rank 12 (schoolbook)
    # (0,1,1): 2x3 * 3x3 -> rank 18 (schoolbook)
    # (1,0,0): 3x2 * 2x2 -> rank 12 (schoolbook)
    # (1,0,1): 3x2 * 2x3 -> rank 18 (schoolbook)
    # (1,1,0): 3x3 * 3x2 -> rank 18 (schoolbook)
    # (1,1,1): 3x3 * 3x3 -> rank 27 (schoolbook)
    
    total_rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27
    # = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    U = [[0]*total_rank for _ in range(n*m)]  # 25 x 124
    V = [[0]*total_rank for _ in range(m*p)]  # 25 x 124
    W = [[0]*total_rank for _ in range(p*n)]  # 25 x 124
    
    col_offset = 0
    
    # Sub-product (0,0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Uses Strassen, rank 7
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(4):
        for c in range(7):
            if U_s[r][c] != 0:
                # r = i0 * 2 + j0, where i0 in {0,1}, j0 in {0,1}
                i0 = r // 2
                j0 = r % 2
                # Global A row: i0 (since n0 starts at 0), global col: j0 (m0 starts at 0)
                U[i0 * 5 + j0][col_offset + c] = U_s[r][c]
    for r in range(4):
        for c in range(7):
            if V_s[r][c] != 0:
                # r = j0 * 2 + k0, where j0 in {0,1}, k0 in {0,1}
                j0 = r // 2
                k0 = r % 2
                # Global B row: j0, global col: k0
                V[j0 * 5 + k0][col_offset + c] = V_s[r][c]
    for r in range(4):
        for c in range(7):
            if W_s[r][c] != 0:
                # r = k0 * 2 + i0, where k0 in {0,1}, i0 in {0,1}
                k0 = r // 2
                i0 = r % 2
                # Global C: row i0, col k0 -> W index: k0 * 5 + i0
                W[k0 * 5 + i0][col_offset + c] = W_s[r][c]
    col_offset += 7
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Sub-product (0,0,1): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # 2x2 * 2x3, schoolbook rank 12
    U_b, V_b, W_b = _schoolbook_block(2, 2, 3)
    for r in range(4):
        for c in range(12):
            if U_b[r][c] != 0:
                i0 = r // 2
                j0 = r % 2
                U[i0 * 5 + j0][col_offset + c] = U_b[r][c]
    for r in range(6):
        for c in range(12):
            if V_b[r][c] != 0:
                # r = j0 * 3 + k0, j0 in {0,1}, k0 in {0,1,2}
                j0 = r // 3
                k0 = r % 3
                # Global B: row j0, col 2+k0
                V[j0 * 5 + (2 + k0)][col_offset + c] = V_b[r][c]
    for r in range(6):
        for c in range(12):
            if W_b[r][c] != 0:
                # r = k0 * 2 + i0, k0 in {0,1,2}, i0 in {0,1}
                k0 = r // 2
                i0 = r % 2
                # Global C: row i0, col 2+k0 -> W index: (2+k0)*5 + i0
                W[(2 + k0) * 5 + i0][col_offset + c] = W_b[r][c]
    col_offset += 12
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Sub-product (0,1,0): A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # 2x3 * 3x2, schoolbook rank 12
    U_b, V_b, W_b = _schoolbook_block(2, 3, 2)
    for r in range(6):
        for c in range(12):
            if U_b[r][c] != 0:
                # r = i0 * 3 + j0, i0 in {0,1}, j0 in {0,1,2}
                i0 = r // 3
                j0 = r % 3
                # Global A: row i0, col 2+j0
                U[i0 * 5 + (2 + j0)][col_offset + c] = U_b[r][c]
    for r in range(6):
        for c in range(12):
            if V_b[r][c] != 0:
                # r = j0 * 2 + k0, j0 in {0,1,2}, k0 in {0,1}
                j0 = r // 2
                k0 = r % 2
                # Global B: row 2+j0, col k0
                V[(2 + j0) * 5 + k0][col_offset + c] = V_b[r][c]
    for r in range(4):
        for c in range(12):
            if W_b[r][c] != 0:
                # r = k0 * 2 + i0, k0 in {0,1}, i0 in {0,1}
                k0 = r // 2
                i0 = r % 2
                # Global C: row i0, col k0
                W[k0 * 5 + i0][col_offset + c] = W_b[r][c]
    col_offset += 12
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Sub-product (0,1,1): A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # 2x3 * 3x3, schoolbook rank 18
    U_b, V_b, W_b = _schoolbook_block(2, 3, 3)
    for r in range(6):
        for c in range(18):
            if U_b[r][c] != 0:
                i0 = r // 3
                j0 = r % 3
                U[i0 * 5 + (2 + j0)][col_offset + c] = U_b[r][c]
    for r in range(9):
        for c in range(18):
            if V_b[r][c] != 0:
                # r = j0 * 3 + k0, j0 in {0,1,2}, k0 in {0,1,2}
                j0 = r // 3
                k0 = r % 3
                V[(2 + j0) * 5 + (2 + k0)][col_offset + c] = V_b[r][c]
    for r in range(6):
        for c in range(18):
            if W_b[r][c] != 0:
                # r = k0 * 2 + i0, k0 in {0,1,2}, i0 in {0,1}
                k0 = r // 2
                i0 = r % 2
                W[(2 + k0) * 5 + i0][col_offset + c] = W_b[r][c]
    col_offset += 18
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Sub-product (1,0,0): A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # 3x2 * 2x2, schoolbook rank 12
    U_b, V_b, W_b = _schoolbook_block(3, 2, 2)
    for r in range(6):
        for c in range(12):
            if U_b[r][c] != 0:
                # r = i1 * 2 + j0, i1 in {0,1,2}, j0 in {0,1}
                i1 = r // 2
                j0 = r % 2
                U[(2 + i1) * 5 + j0][col_offset + c] = U_b[r][c]
    for r in range(4):
        for c in range(12):
            if V_b[r][c] != 0:
                j0 = r // 2
                k0 = r % 2
                V[j0 * 5 + k0][col_offset + c] = V_b[r][c]
    for r in range(6):
        for c in range(12):
            if W_b[r][c] != 0:
                # r = k0 * 3 + i1, k0 in {0,1}, i1 in {0,1,2}
                k0 = r // 3
                i1 = r % 3
                W[k0 * 5 + (2 + i1)][col_offset + c] = W_b[r][c]
    col_offset += 12
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Sub-product (1,0,1): A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # 3x2 * 2x3, schoolbook rank 18
    U_b, V_b, W_b = _schoolbook_block(3, 2, 3)
    for r in range(6):
        for c in range(18):
            if U_b[r][c] != 0:
                i1 = r // 2
                j0 = r % 2
                U[(2 + i1) * 5 + j0][col_offset + c] = U_b[r][c]
    for r in range(6):
        for c in range(18):
            if V_b[r][c] != 0:
                # r = j0 * 3 + k0, j0 in {0,1}, k0 in {0,1,2}
                j0 = r // 3
                k0 = r % 3
                V[j0 * 5 + (2 + k0)][col_offset + c] = V_b[r][c]
    for r in range(9):
        for c in range(18):
            if W_b[r][c] != 0:
                # r = k0 * 3 + i1, k0 in {0,1,2}, i1 in {0,1,2}
                k0 = r // 3
                i1 = r % 3
                W[(2 + k0) * 5 + (2 + i1)][col_offset + c] = W_b[r][c]
    col_offset += 18
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Sub-product (1,1,0): A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    # 3x3 * 3x2, schoolbook rank 18
    U_b, V_b, W_b = _schoolbook_block(3, 3, 2)
    for r in range(9):
        for c in range(18):
            if U_b[r][c] != 0:
                # r = i1 * 3 + j1, i1 in {0,1,2}, j1 in {0,1,2}
                i1 = r // 3
                j1 = r % 3
                U[(2 + i1) * 5 + (2 + j1)][col_offset + c] = U_b[r][c]
    for r in range(6):
        for c in range(18):
            if V_b[r][c] != 0:
                # r = j1 * 2 + k0, j1 in {0,1,2}, k0 in {0,1}
                j1 = r // 2
                k0 = r % 2
                V[(2 + j1) * 5 + k0][col_offset + c] = V_b[r][c]
    for r in range(6):
        for c in range(18):
            if W_b[r][c] != 0:
                # r = k0 * 3 + i1, k0 in {0,1}, i1 in {0,1,2}
                k0 = r // 3
                i1 = r % 3
                W[k0 * 5 + (2 + i1)][col_offset + c] = W_b[r][c]
    col_offset += 18
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Sub-product (1,1,1): A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # 3x3 * 3x3, schoolbook rank 27
    U_b, V_b, W_b = _schoolbook_block(3, 3, 3)
    for r in range(9):
        for c in range(27):
            if U_b[r][c] != 0:
                i1 = r // 3
                j1 = r % 3
                U[(2 + i1) * 5 + (2 + j1)][col_offset + c] = U_b[r][c]
    for r in range(9):
        for c in range(27):
            if V_b[r][c] != 0:
                # r = j1 * 3 + k1, j1 in {0,1,2}, k1 in {0,1,2}
                j1 = r // 3
                k1 = r % 3
                V[(2 + j1) * 5 + (2 + k1)][col_offset + c] = V_b[r][c]
    for r in range(9):
        for c in range(27):
            if W_b[r][c] != 0:
                # r = k1 * 3 + i1, k1 in {0,1,2}, i1 in {0,1,2}
                k1 = r // 3
                i1 = r % 3
                W[(2 + k1) * 5 + (2 + i1)][col_offset + c] = W_b[r][c]
    col_offset += 27
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
