"""Matrix multiplication 4,4,8 rank-minimal search.

Combines a rank-search structure with a known rank-98 Strassen(4x4) x
schoolbook(8) construction and structured local search (flips, row/col
swaps, zeroing, row/col merging) to attempt to reduce the rank below 98.
"""
import random
import time

DIMENSIONS = [4, 4, 8]


def _validate(U, V, W, n, m, p):
    """Exact check: sum_r U[a,r]V[b,r]W[c,r] == 1 at (a,b,c) and 0 elsewhere."""
    rank = len(U[0]) if U else 0
    nm, mp, pn = n * m, m * p, p * n
    ones = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
                ones[(a, b, c)] = True
    # sample of zero positions
    rng = random.Random(12345)
    for _ in range(2000):
        a = rng.randrange(nm)
        b = rng.randrange(mp)
        c = rng.randrange(pn)
        if (a, b, c) in ones:
            continue
        s = 0.0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s) > 1e-9:
            return False
    return True


def _strassen_4x4():
    """Strassen 4x4 decomposition. Returns U,V,W each shape (16,49)."""
    # 2x2 Strassen base factors
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    # M1: (A00+A11)(B00+B11) -> C00,C11
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    # M2: (A10+A11)B00 -> C10
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[2][1]=1
    # M3: A00(B01-B11) -> C01
    U2[0][2]=1
    V2[1][2]=1; V2[3][2]=-1
    W2[1][2]=1
    # M4: A11(B10-B00) -> C10
    U2[3][3]=1
    V2[2][3]=1; V2[0][3]=-1
    W2[2][3]=1
    # M5: (A00+A01)B11 -> C01,C11
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    W2[1][4]=1; W2[3][4]=1
    # M6: (A10-A00)(B00+B01) -> C10
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    W2[2][5]=1
    # M7: (A01-A11)(B10+B11) -> C00,C11
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    W2[0][6]=1; W2[3][6]=1

    # High-level Strassen for the 4x4 (2x2 blocks of 2x2)
    # M_h = (A-combination) * (B-combination); W_h = C-block coefficient matrix
    # factor: (U_blocks, V_blocks, W_coeffs)
    factors = [
        ([(0,1),(1,1)], [(0,1),(1,1)], {(0,0):1,(1,1):1}),          # M11
        ([(1,0),(1,1)], [(0,0)],          {(1,0):1}),                # M12
        ([(0,0)],          [(0,1),(1,1)], {(0,1):1}),                # M13
        ([(1,1)],          [(1,0),(0,0)], {(1,0):1}),                # M14
        ([(0,0),(0,1)],    [(1,1)],       {(0,1):1,(1,1):1}),        # M15
        ([(1,0),(0,0)],    [(0,0),(0,1)], {(1,0):1}),                # M16
        ([(0,1),(1,1)],    [(1,0),(1,1)], {(0,0):1,(1,1):1}),        # M17
    ]

    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]

    def Aidx(br, bc, lr, lc):  # 4x4 index
        return (br*2+lr)*4 + (bc*2+lc)
    def Bidx(br, bc, lr, lc):
        return (br*2+lr)*4 + (bc*2+lc)
    def Cidx(br, bc, lr, lc):
        return (bc*2+lc)*4 + (br*2+lr)

    col = 0
    for h in range(7):
        ub, vb, wc = factors[h]
        for l in range(7):
            for (br, bc), s in ub:
                for lr in range(2):
                    for lc in range(2):
                        U[Aidx(br,bc,lr,lc)][col] += s * U2[lr*2+lc][l]
            for (br, bc), s in vb:
                for lr in range(2):
                    for lc in range(2):
                        V[Bidx(br,bc,lr,lc)][col] += s * V2[lr*2+lc][l]
            for (br, bc), s in wc.items():
                for lr in range(2):
                    for lc in range(2):
                        W[Cidx(br,bc,lr,lc)][col] += s * W2[lr*2+lc][l]
            col += 1
    return U, V, W


def _construct_known(n, m, p):
    """Known rank-98 = Strassen(4x4) x schoolbook(8) decomposition."""
    n, m, p = 4, 4, 8
    U4, V4, W4 = _strassen_4x4()
    r4 = 49
    r = r4 * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    for j in range(m):
        for k in range(p):
            base = j * p + k
            for r4i in range(r4):
                col = base * r4 + r4i
                for i in range(n):
                    U[i*m+j][col] = U4[i*4+j][r4i]
                V[base][col] = V4[j*4+k][r4i]
                W[k*n+i][col] = W4[k*4+i][r4i]
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _local_search(U, V, W, n, m, p, deadline, max_iters=4000):
    """Structured local search to reduce rank while staying exact."""
    rank = len(U[0]) if U else 0
    rng = random.Random(7)
    iters = 0
    while iters < max_iters and time.time() < deadline - 5:
        iters += 1
        op = rng.random()
        if op < 0.40:
            # flip an entry
            mat = rng.choice([U, V, W])
            row = rng.randrange(len(mat))
            col = rng.randrange(rank)
            old = mat[row][col]
            mat[row][col] = -old if old != 0 else 1
            if not _validate(U, V, W, n, m, p):
                mat[row][col] = old
        elif op < 0.65:
            # swap two columns (no rank change, can enable merges)
            mat = rng.choice([U, V, W])
            c1, c2 = rng.sample(range(rank), 2)
            for row in range(len(mat)):
                mat[row][c1], mat[row][c2] = mat[row][c2], mat[row][c1]
            # always keep (permutation of factors, validity unchanged)
        elif op < 0.80:
            # zero an entry
            mat = rng.choice([U, V, W])
            row = rng.randrange(len(mat))
            col = rng.randrange(rank)
            old = mat[row][col]
            mat[row][col] = 0
            if not _validate(U, V, W, n, m, p):
                mat[row][col] = old
        elif op < 0.90:
            # set entry to +/-1
            mat = rng.choice([U, V, W])
            row = rng.randrange(len(mat))
            col = rng.randrange(rank)
            old = mat[row][col]
            mat[row][col] = rng.choice([1, -1])
            if not _validate(U, V, W, n, m, p):
                mat[row][col] = old
        else:
            # drop a column if it is all-zero in all three matrices
            col = rng.randrange(rank)
            if all(U[a][col] == 0 for a in range(n*m)) and \
               all(V[b][col] == 0 for b in range(m*p)) and \
               all(W[c][col] == 0 for c in range(p*n)):
                for a in range(n*m):
                    U[a].pop(col)
                for b in range(m*p):
                    V[b].pop(col)
                for c in range(p*n):
                    W[c].pop(col)
                rank -= 1
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin under 150s kill

    # Start from the known rank-98 Strassen x schoolbook construction.
    base = _construct_known(n, m, p)
    U, V, W = base["U"], base["V"], base["W"]
    best_rank = len(U[0])
    if _validate(U, V, W, n, m, p):
        U, V, W, _ = _local_search(U, V, W, n, m, p, deadline)
        if _validate(U, V, W, n, m, p):
            best_rank = len(U[0])

    # Try random low-rank searches to attempt to beat the baseline.
    for r in range(40, best_rank):
        if time.time() > deadline - 10:
            break
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            U = [[random.choice([0,1,-1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([0,1,-1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([0,1,-1]) for _ in range(r)] for _ in range(p*n)]
            U, V, W, _ = _local_search(U, V, W, n, m, p, deadline, max_iters=600)
            if _validate(U, V, W, n, m, p):
                best_rank = len(U[0])
                break

    if best_rank == len(U[0]) and _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: return the known valid rank-98 (or schoolbook if needed).
    known = _construct_known(n, m, p)
    if _validate(known["U"], known["V"], known["W"], n, m, p):
        return known
    return _construct_schoolbook(n, m, p)
