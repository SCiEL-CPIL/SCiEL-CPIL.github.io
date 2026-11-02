"""Kronecker-structured Strassen decomposition for 4x5x6 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _strassen_2x2_terms():
    """Return 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (U_vec, V_vec, W_vec) of length 4 each.
    U_vec indexes [a11, a12, a21, a22]
    V_vec indexes [b11, b12, b21, b22]
    W_vec indexes [c11, c12, c21, c22]
    """
    terms = []
    # P1 = (a11+a22)(b11+b22) -> contributes to c11, c22
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # P2 = (a21+a22)b11 -> contributes to c21, c22
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # P3 = a11(b12-b22) -> contributes to c12, c22
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    # P4 = a22(b21-b11) -> contributes to c11, c21
    terms.append(([0,0,0,1], [0,-1,1,0], [1,0,1,0]))
    # P5 = (a11+a12)b22 -> contributes to c11, c12
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # P6 = (a21-a11)(b11+b12) -> contributes to c22
    terms.append(([0,-1,1,0], [1,1,0,0], [0,0,0,1]))
    # P7 = (a12-a22)(b21+b22) -> contributes to c11
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _embed_2x2x2_into_block(U, V, W, i0, j0, k0, n, m, p, col_offset):
    """Embed a 2x2x2 Strassen block at position (i0,j0,k0) into full matrices.
    U has shape (n*m, total_rank), V has shape (m*p, total_rank), W has shape (p*n, total_rank)
    """
    terms = _strassen_2x2_terms()
    for t, (u_vec, v_vec, w_vec) in enumerate(terms):
        col = col_offset + t
        # U: A block is 2x2, rows i0..i0+1, cols j0..j0+1
        for li in range(2):
            for lj in range(2):
                local_idx = li * 2 + lj
                if u_vec[local_idx] != 0:
                    gi = i0 + li
                    gj = j0 + lj
                    U[gi * m + gj][col] = u_vec[local_idx]
        # V: B block is 2x2, rows j0..j0+1, cols k0..k0+1
        for lj in range(2):
            for lk in range(2):
                local_idx = lj * 2 + lk
                if v_vec[local_idx] != 0:
                    gj = j0 + lj
                    gk = k0 + lk
                    V[gj * p + gk][col] = v_vec[local_idx]
        # W: C block is 2x2, rows k0..k0+1, cols i0..i0+1
        for lk in range(2):
            for li in range(2):
                local_idx = lk * 2 + li
                if w_vec[local_idx] != 0:
                    gk = k0 + lk
                    gi = i0 + li
                    W[gk * n + gi][col] = w_vec[local_idx]

def _embed_4x4x4_strassen(U, V, W, n, m, p, col_offset):
    """Embed 4x4x4 Strassen (rank 49) at top-left corner.
    Uses recursive 2x2 Strassen structure.
    """
    terms_2x2 = _strassen_2x2_terms()
    
    # 4x4 Strassen: 7 top-level blocks, each is 2x2 Strassen (rank 7)
    # Top-level block (s, t) where s in 0..6, t in 0..6
    # For each top-level term s, we have 7 sub-terms t
    # The top-level structure:
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    # P2 = (A10+A11)B00 -> C10 += P2, C11 -= P2
    # P3 = A00(B01-B11) -> C01 += P3, C11 += P3
    # P4 = A11(B10-B00) -> C00 += P4, C10 += P4
    # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
    # P6 = (A10-A00)(B00+B01) -> C11 += P6
    # P7 = (A01-A11)(B10+B11) -> C00 += P7
    
    # For each top-level term s, the A/B/C blocks are 2x2 blocks of the 4x4 matrices
    # A blocks: A00=(0,0), A01=(0,2), A10=(2,0), A11=(2,2)
    # B blocks: B00=(0,0), B01=(0,2), B10=(2,0), B11=(2,2)
    # C blocks: C00=(0,0), C01=(0,2), C10=(2,0), C11=(2,2)
    
    # Define the 7 top-level terms as (A_block_specs, B_block_specs, C_block_specs)
    # Each spec is a list of ((block_row, block_col), sign)
    top_terms = [
        # s=0: (A00+A11)(B00+B11) -> C00, C11
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # s=1: (A10+A11)B00 -> C10, -C11
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1), ((1,1),-1)]),
        # s=2: A00(B01-B11) -> C01, C11
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
        # s=3: A11(B10-B00) -> C00, C10
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
        # s=4: (A00+A01)B11 -> -C00, C01
        ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
        # s=5: (A10-A00)(B00+B01) -> C11
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
        # s=6: (A01-A11)(B10+B11) -> C00
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]),
    ]
    
    for s, (a_specs, b_specs, c_specs) in enumerate(top_terms):
        for t in range(7):
            col = col_offset + s * 7 + t
            u_vec, v_vec, w_vec = terms_2x2[t]
            
            # A part
            for (bi, bj), sign_a in a_specs:
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        if u_vec[local_idx] != 0:
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            U[gi * m + gj][col] = u_vec[local_idx] * sign_a
            
            # B part
            for (bj, bk), sign_b in b_specs:
                for lj in range(2):
                    for lk in range(2):
                        local_idx = lj * 2 + lk
                        if v_vec[local_idx] != 0:
                            gj = bj * 2 + lj
                            gk = bk * 2 + lk
                            V[gj * p + gk][col] = v_vec[local_idx] * sign_b
            
            # C part
            for (ci, ck), sign_c in c_specs:
                for lk in range(2):
                    for li in range(2):
                        local_idx = lk * 2 + li
                        if w_vec[local_idx] != 0:
                            gk = ck * 2 + lk
                            gi = ci * 2 + li
                            W[gk * n + gi][col] = w_vec[local_idx] * sign_c

def _embed_4x1x6_schoolbook(U, V, W, n, m, p, col_offset):
    """Embed 4x1x6 schoolbook block: A is 4x1 (col 4), B is 1x6 (row 4), C is 4x6.
    Rank = 4*6 = 24.
    """
    for i in range(4):
        for k in range(6):
            col = col_offset + i * 6 + k
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1

def _embed_4x4x2_strassen(U, V, W, n, m, p, col_offset):
    """Embed 4x4x2 decomposition using 4 blocks of 2x2x2 Strassen.
    A is 4x4 (cols 0-3), B is 4x2 (rows 0-3, cols 4-5), C is 4x2.
    4 blocks: (i0,j0,k0) in {(0,0,4),(0,2,4),(2,0,4),(2,2,4)}
    Rank = 4*7 = 28.
    """
    blocks = [(0,0,4), (0,2,4), (2,0,4), (2,2,4)]
    for idx, (i0, j0, k0) in enumerate(blocks):
        _embed_2x2x2_into_block(U, V, W, i0, j0, k0, n, m, p, col_offset + idx * 7)

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank: 49 (4x4x4 Strassen) + 24 (4x1x6 schoolbook) + 28 (4x4x2 Strassen)
    total_rank = 49 + 24 + 28
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: 4x4x4 Strassen at (i:0-3, j:0-3, k:0-3)
    _embed_4x4x4_strassen(U, V, W, n, m, p, 0)
    
    # Block 2: 4x1x6 schoolbook at (i:0-3, j:4, k:0-5)
    _embed_4x1x6_schoolbook(U, V, W, n, m, p, 49)
    
    # Block 3: 4x4x2 Strassen at (i:0-3, j:0-3, k:4-5)
    _embed_4x4x2_strassen(U, V, W, n, m, p, 73)
    
    # Verify with a quick check
    if remaining() > 10:
        import random
        ok = True
        for _ in range(3):
            A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
            C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0
                    for c in range(total_rank):
                        for j in range(m):
                            acc += U[i * m + j][c] * W[k * n + i][c] * V[j * p + k][c]
                    if acc != C[i][k]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            # Fallback to schoolbook
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
