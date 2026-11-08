"""Combined: Direct rank-minimal search for 4x5x6 matrix multiplication.

Strategy: The current program uses recursive block structures (Strassen + schoolbook)
which gives rank 101. We attempt a direct search for a lower-rank decomposition by
treating it as a constraint satisfaction problem.

Key insight: We can use the known optimal decompositions for sub-problems and try to
find a more efficient combination. For 4x5x6, we can split into:
- 4x4x4: rank 49 (2-level Strassen)
- 4x1x4: rank 16 (schoolbook)  
- 4x4x2: rank 28 (4x Strassen 2x2x2)
- 4x1x2: rank 8 (schoolbook)
Total: 101

But we can try to be smarter about the 4x4x4 part. The theoretical lower bound is
around 47-49 for 4x4x4. Let's see if we can find a better decomposition.

Actually, let's try a different approach: use the fact that 5 = 4+1 and 6 = 4+2,
and see if we can share some structure between the blocks.

For now, let's implement a clean version of the current best (rank 101) and verify it works.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) lists of shape (4, 7).
    Indexing: U row = i*2+j, V row = j*2+k, W row = k*2+i.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # 7 Strassen products with proper output mapping
    # C00 = P1 - P2 + P3 + P4
    # C01 = P5 + P6
    # C10 = P7 + P1
    # C11 = P2 - P4 + P6 - P7
    # Wait, let me use the standard Strassen formula:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)
    # P4 = A11(B01-B11)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    defs = [
        # P1 = (a00+a11)(b00+b11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (a10+a11)(b00)
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
        # P3 = a00(b10-b00)
        ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)]),
        # P4 = a11(b01-b11)
        ([(1,1,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(0,0,1)]),
        # P5 = (a00+a01)(b11)
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
        # P6 = (a10-a00)(b00+b01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7 = (a01-a11)(b10+b11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(7):
        for (i, j, c) in defs[t][0]:
            U[i * 2 + j][t] += c
        for (j, k, c) in defs[t][1]:
            V[j * 2 + k][t] += c
        for (k, i, c) in defs[t][2]:
            W[k * 2 + i][t] += c
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    4x4x4 via 2-level Strassen: rank 49.
    """
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    U2, V2, W2 = _strassen_2x2()
    
    # Outer Strassen: 7 blocks, each using inner Strassen (7 products)
    outer = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
        ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(0,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            ue, ve, we = outer[s]
            for (bi, bj, c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        a_loc = i * 2 + j
                        cc = U2[a_loc][t]
                        if cc:
                            U[gi * 4 + gj][col] += cc * c
            for (bj, bk, c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj * 2 + j
                        gk = bk * 2 + k
                        b_loc = j * 2 + k
                        cc = V2[b_loc][t]
                        if cc:
                            V[gj * 4 + gk][col] += cc * c
            for (bk, bi, c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk * 2 + k
                        gi = bi * 2 + i
                        c_loc = k * 2 + i
                        cc = W2[c_loc][t]
                        if cc:
                            W[gk * 4 + gi][col] += cc * c
    return U, V, W


def _build_4x4x2_strassen():
    """
    4x4x2 via 4 blocks of 2x2x2 Strassen: rank 28.
    """
    U = [[0] * 28 for _ in range(16)]
    V = [[0] * 28 for _ in range(8)]
    W = [[0] * 28 for _ in range(8)]
    U2, V2, W2 = _strassen_2x2()
    
    # 4 blocks: (ar,ac,br,cr,ck)
    blocks = [(0,0,0,0,0), (0,2,2,0,0), (2,0,0,2,0), (2,2,2,2,0)]
    
    for bi, (ar, ac, br, cr, ck) in enumerate(blocks):
        base = bi * 7
        for t in range(7):
            col = base + t
            for i in range(2):
                for j in range(2):
                    a_loc = i * 2 + j
                    cc = U2[a_loc][t]
                    if cc:
                        U[(ar + i) * 4 + (ac + j)][col] += cc
            for j in range(2):
                for k in range(2):
                    b_loc = j * 2 + k
                    cc = V2[b_loc][t]
                    if cc:
                        V[(br + j) * 2 + (ck + k)][col] += cc
            for k in range(2):
                for i in range(2):
                    c_loc = k * 2 + i
                    cc = W2[c_loc][t]
                    if cc:
                        W[(ck + k) * 4 + (cr + i)][col] += cc
    return U, V, W


def _build_outer(n, m, p):
    """
    (n x m) * (m x p) -> (n x p), schoolbook rank n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _combine(parts, n, m, p):
    """
    Combine multiple decompositions by direct sum.
    parts: list of (U, V, W, (i0, j0, ii, jj, k0, kk))
    where (ii, jj, kk) is the sub-problem dimensions.
    """
    total = sum(len(d[0][0]) for d in parts)
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    off = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        nA, mA, nB = ii, jj, kk
        for r in range(nA * mA):
            i = r // mA
            j = r % mA
            for c in range(len(Us[0])):
                U[(i0 + i) * m + (j0 + j)][off + c] = Us[r][c]
        for r in range(mA * nB):
            j = r // nB
            k = r % nB
            for c in range(len(Vs[0])):
                V[(j0 + j) * p + (k0 + k)][off + c] = Vs[r][c]
        for r in range(nB * nA):
            k = r // nA
            i = r % nA
            for c in range(len(Ws[0])):
                W[(k0 + k) * n + (i0 + i)][off + c] = Ws[r][c]
        off += len(Us[0])
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify exact decomposition using random integer matrices.
    """
    rng = random.Random(987654321)
    rank = len(U[0])
    for _ in range(5):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for t in range(rank):
                    au = sum(U[i * m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j * p + k][t] * B[j][k] for j in range(m))
                    val += W[k * n + i][t] * au * bu
                if abs(val - C[i][k]) > tol:
                    return False
    # Also check the basis tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    try:
        # Build the additive decomposition
        # 4x5x6 = (4x4)*(4x4) + (4x1)*(1x4) + (4x4)*(4x2) + (4x1)*(1x2)
        Ua, Va, Wa = _build_4x4x4_strassen()  # rank 49
        Ub, Vb, Wb = _build_outer(4, 1, 4)    # rank 16
        Uc, Vc, Wc = _build_4x4x2_strassen()  # rank 28
        Ud, Vd, Wd = _build_outer(4, 1, 2)    # rank 8
        
        parts = [
            (Ua, Va, Wa, (0, 0, 4, 4, 0, 4)),   # 4x4 * 4x4 -> 4x4
            (Ub, Vb, Wb, (0, 4, 4, 1, 0, 4)),   # 4x1 * 1x4 -> 4x4
            (Uc, Vc, Wc, (0, 4, 4, 4, 4, 2)),   # 4x4 * 4x2 -> 4x2
            (Ud, Vd, Wd, (0, 4, 4, 1, 4, 2)),   # 4x1 * 1x2 -> 4x2
        ]
        
        U, V, W = _combine(parts, n, m, p)
        rank = len(U[0])
        
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
        else:
            # Verification failed, fall back to schoolbook
            pass
            
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
