"""Tensor decomposition for 4x4 x 4x6 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time as _time

def _build_strassen_4x4x6():
    """Build rank-minimal-ish decomposition via Strassen on 2x2 blocks.

    We decompose C = A B where A is 4x4, B is 4x6.
    Split A into 2x2 blocks A11,A12,A21,A22 (each 2x2).
    Split B into top 2 rows and bottom 2 rows, each 2x6.
    Use Strassen for the 2x2 block product to get 2x6 blocks.
    """
    # We'll build U,V,W as lists of lists.
    # Use a symbolic-free integer construction.
    # Strassen 2x2x2 has rank 7. For 2x2 x 2x6, treat B as two 2x3 or use
    # Strassen per column-pair. Simplest robust: use rank-7 Strassen for each
    # 2x2x2 sub-block, and there are 3 column-pairs in the 6 columns -> 7*3=21
    # per 2x2 (i-block) x 2 (j-block) ... let's just do schoolbook-free via
    # Strassen on the full 4x4x4 for columns 0..3 and schoolbook for 4..5.
    # To keep it simple and CORRECT, build via explicit Strassen 4x4x4 (rank 28)
    # for the 4x4 part, then add 4x4x2 schoolbook (rank 8) => 36? But measured
    # best is 84. We'll just produce a VALID decomposition; rank lower is better.
    # Provide a correct rank-28+? Let's do full Strassen 4x4x4 = 28 for cols0-3,
    # and Strassen-like for cols 4-5 as 4x4x2.
    n, m, p = 4, 4, 6
    U = []
    V = []
    W = []

    def add_row_u(idx, vec):
        while len(U) <= idx:
            U.append([0] * (len(vec)))
        # ensure length
        while len(U[idx]) < len(vec):
            U[idx].append(0)
        for c, v in enumerate(vec):
            U[idx][c] += v

    def add_row_v(idx, vec):
        while len(V) <= idx:
            V.append([0] * (len(vec)))
        while len(V[idx]) < len(vec):
            V[idx].append(0)
        for c, v in enumerate(vec):
            V[idx][c] += v

    def add_row_w(idx, vec):
        while len(W) <= idx:
            W.append([0] * (len(vec)))
        while len(W[idx]) < len(vec):
            W[idx].append(0)
        for c, v in enumerate(vec):
            W[idx][c] += v

    # We'll allocate columns dynamically.
    cols = 0
    def new_col():
        nonlocal cols
        c = cols
        cols += 1
        # extend all rows
        for r in U:
            r.append(0)
        for r in V:
            r.append(0)
        for r in W:
            r.append(0)
        return c

    # Strassen 2x2x2 rank-7 patterns on indices (a,b) for A-block, (c,d) for B-block
    # For a 2x2 A block at rows (r0,r1) cols (c0,c1), and 2x2 B block at rows (c0',c1') cols (k0,k1):
    # M1 = (A00+A11)(B01+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B10-B11)
    # M4 = A11 (B00-B10)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # We'll iterate over all 2x2 A-blocks (2x2 grid) and all 2x2 B-blocks (2x3 grid over 6 cols).
    # A-blocks: ai in 0..1, aj in 0..1 -> rows 2*ai,2*ai+1 ; cols 2*aj,2*aj+1
    # B-blocks: bi in 0..1 (rows 2*bi,2*bi+1), bk in 0..2 (cols 2*bk,2*bk+1)
    # For each pair (A-block, B-block) we add 7 columns.
    # Total columns = 4 A-blocks * 6 B-blocks * 7 = 168. That's rank 168 (valid but high).
    # We can reduce by noting Strassen shares within a fixed (ai,bi) over bk.
    # Keep it simple: full construction, correct.

    for ai in range(2):
        for aj in range(2):
            a00 = (2*ai, 2*aj)
            a01 = (2*ai, 2*aj+1)
            a10 = (2*ai+1, 2*aj)
            a11 = (2*ai+1, 2*aj+1)
            for bi in range(2):
                for bk in range(3):
                    b00 = (2*bi, 2*bk)
                    b01 = (2*bi, 2*bk+1)
                    b10 = (2*bi+1, 2*bk)
                    b11 = (2*bi+1, 2*bk+1)
                    # output C indices: rows from A-block (2*ai,2*ai+1), cols from B-block (2*bk,2*bk+1)
                    c00 = (2*ai, 2*bk)
                    c01 = (2*ai, 2*bk+1)
                    c10 = (2*ai+1, 2*bk)
                    c11 = (2*ai+1, 2*bk+1)

                    # 7 Strassen products
                    # M1 = (A00+A11)(B01+B11)
                    c = new_col()
                    U[a00[0]*m + a00[1]][c] += 1
                    U[a11[0]*m + a11[1]][c] += 1
                    V[b01[0]*p + b01[1]][c] += 1
                    V[b11[0]*p + b11[1]][c] += 1
                    W[c00[1]*n + c00[0]][c] += 1
                    W[c11[1]*n + c11[0]][c] += 1
                    # M2 = (A10+A11) B00
                    c = new_col()
                    U[a10[0]*m + a10[1]][c] += 1
                    U[a11[0]*m + a11[1]][c] += 1
                    V[b00[0]*p + b00[1]][c] += 1
                    W[c10[1]*n + c10[0]][c] += 1
                    # M3 = A00 (B10-B11)
                    c = new_col()
                    U[a00[0]*m + a00[1]][c] += 1
                    V[b10[0]*p + b10[1]][c] += 1
                    V[b11[0]*p + b11[1]][c] += -1
                    W[c01[1]*n + c01[0]][c] += 1
                    # M4 = A11 (B00-B10)
                    c = new_col()
                    U[a11[0]*m + a11[1]][c] += 1
                    V[b00[0]*p + b00[1]][c] += 1
                    V[b10[0]*p + b10[1]][c] += -1
                    W[c00[1]*n + c00[0]][c] += 1
                    W[c10[1]*n + c10[0]][c] += 1
                    # M5 = (A00+A01) B11
                    c = new_col()
                    U[a00[0]*m + a00[1]][c] += 1
                    U[a01[0]*m + a01[1]][c] += 1
                    V[b11[0]*p + b11[1]][c] += 1
                    W[c00[1]*n + c00[0]][c] += -1
                    W[c01[1]*n + c01[0]][c] += 1
                    # M6 = (A10-A00)(B00+B01)
                    c = new_col()
                    U[a10[0]*m + a10[1]][c] += 1
                    U[a00[0]*m + a00[1]][c] += -1
                    V[b00[0]*p + b00[1]][c] += 1
                    V[b01[0]*p + b01[1]][c] += 1
                    W[c11[1]*n + c11[0]][c] += 1
                    # M7 = (A01-A11)(B10+B11)
                    c = new_col()
                    U[a01[0]*m + a01[1]][c] += 1
                    U[a11[0]*m + a11[1]][c] += -1
                    V[b10[0]*p + b10[1]][c] += 1
                    V[b11[0]*p + b11[1]][c] += 1
                    W[c00[1]*n + c00[0]][c] += 1
                    W[c11[1]*n + c11[0]][c] += 1

    # Ensure all rows have length = cols
    for r in U:
        while len(r) < cols:
            r.append(0)
    for r in V:
        while len(r) < cols:
            r.append(0)
    for r in W:
        while len(r) < cols:
            r.append(0)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, sample_seeds=8):
    import random
    rank = len(U[0])
    for seed in range(sample_seeds):
        rng = random.Random(seed)
        A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # expected C = A B
        Cexp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # computed via tensor: C[i,k] = sum_c U[i*m? ...]
        # U shape (n*m, rank): U[i*m + j, c]
        # V shape (m*p, rank): V[j*p + k, c]
        # W shape (p*n, rank): W[k*n + i, c]
        # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] over j and c
        Ccomp = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                Ccomp[i][k] = s
        if Cexp != Ccomp:
            return False
    return True


def construct():
    start = _time.time()
    deadline = start + 140.0  # safety margin under 150s
    n, m, p = DIMENSIONS

    best = None
    # Build the Strassen-based decomposition.
    try:
        res = _build_strassen_4x4x6()
        # quick verification
        if _verify(res["U"], res["V"], res["W"], n, m, p):
            best = res
    except Exception:
        pass

    # Fallback: schoolbook (always valid)
    if best is None or _time.time() > deadline:
        if best is None:
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
            best = {"U": U, "V": V, "W": W}
        return best

    return best
# EVOLVE-BLOCK-END
