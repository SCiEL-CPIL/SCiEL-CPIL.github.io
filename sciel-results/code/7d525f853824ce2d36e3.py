import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for M in (U, V, W):
        for row in M:
            if len(row) != rank:
                return False
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    one = Fraction(1, 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                urow = Uf[i * m + j]
                vrow = Vf[j * p + k]
                wrow = Wf[k * n + i]
                for r in range(rank):
                    s += urow[r] * vrow[r] * wrow[r]
                if s != one:
                    return False
    return True

def _build_rank88():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: i=0, schoolbook 1x4x8
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 via Strassen 2x2x2 blocks
    sU = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]

    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (j0 + lj)][col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(j0 + lj) * p + (k0 + lk)][col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(k0 + lk) * n + (1 + li)][col] += sW[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}

def _build_rank80():
    """Try a 2x4x4 Strassen decomposition for 2 rows, 4 cols, 8 cols (split into 2x4 blocks)."""
    n, m, p = DIMENSIONS
    # Use 2x4x4 Strassen (rank 16 per 2x4x4 block) * 2 blocks = 32 for 2x4x8
    # Plus 1x4x8 schoolbook = 32
    # Total = 64? No, 2x4x4 has rank 16, so 2 blocks = 32, plus 32 = 64
    # But we need to verify this works
    # Actually let's just try rank 88 which we know works
    return None

def _attempt_rank80():
    """Try to find a rank-80 decomposition by combining different block strategies."""
    n, m, p = DIMENSIONS
    # Strategy: Use 3x4x8 = 3*(4x8)
    # Try: row 0 schoolbook (32), rows 1-2 with 2x4x8
    # 2x4x8: split into 2x4x4 blocks (two of them), each rank 16 = 32
    # Total = 32 + 32 = 64. Let's try this.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Row 0: schoolbook 1x4x8
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Rows 1-2: 2x4x8 via two 2x4x4 Strassen blocks
    # Strassen 2x2x2 rank-7 tensor
    sU = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]

    # For 2x4x4: we have 2 rows, 4 cols, 4 cols
    # Split cols into 2x2 blocks: 2 col-blocks x 2 col-blocks = 4 Strassen blocks
    # Each Strassen block is rank 7, so 4*7 = 28 for 2x4x4
    # Wait, 2x4x4 means A is 2x4, B is 4x4.
    # We can view A as 2x2 blocks (one block), B as 2x2 blocks (two blocks).
    # So we have 1 row-block x 2 col-blocks = 2 Strassen 2x2x2 blocks = rank 14 per 2x4x4
    # Hmm, that doesn't seem right. Let me reconsider.
    
    # Actually for 2x4x4:
    # A is 2x4 (one 2x2 block in rows, two 2x2 blocks in cols)
    # B is 4x4 (two 2x2 blocks in rows, two 2x2 blocks in cols)
    # C is 2x4 (one 2x2 block in rows, two 2x2 blocks in cols)
    # We need: for each row-block (1) x each B-row-block (2) x each col-block (2):
    #   1 Strassen 2x2x2 block
    # So total = 1 * 2 * 2 = 4 Strassen blocks = rank 28 per 2x4x4
    # For 2x4x8: two 2x4x4 blocks = rank 56
    # Total = 32 + 56 = 88. Same as before!

    # Let me try a different split for rows 1-2.
    # What if we don't split cols? 2x4x8 directly.
    # A: 2x4, B: 4x8, C: 2x8
    # Split A cols into 2x2 (2 blocks), B cols into 2x2 (4 blocks), B rows into 2x2 (2 blocks)
    # C rows: 1 block, C cols: 4 blocks
    # Strassen blocks: for each A-col-block (2) x B-row-block (2) x B-col-block (4) = 16 blocks
    # Wait, that's not right either.
    
    # Let me just return the rank-88 solution since I can't easily find rank 80.
    return _build_rank88()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best = _build_rank88()

    if not _verify(best["U"], best["V"], best["W"]):
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try to find a better decomposition if time allows
    if time.time() < deadline:
        candidate = _attempt_rank80()
        if candidate and _verify(candidate["U"], candidate["V"], candidate["W"]):
            if len(candidate["U"][0]) < len(best["U"][0]):
                best = candidate

    return best
