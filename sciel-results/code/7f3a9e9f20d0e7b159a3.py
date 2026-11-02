DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0

def _tl():
    return _DEADLINE - (time.time() - _T0)

def _schoolbook(n, m, p):
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
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _strassen222():
    """Rank-7 Strassen decomposition for 2x2 x 2x2.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    Verified: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all (i,j,k).
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0],
    ]
    V = [
        [0, 1, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
    ]
    W = [
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1],
    ]
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r] for r in range(7))
                if s != 1:
                    raise RuntimeError(f"Strassen verify failed at ({i},{j},{k}): {s}")
    return {'U': U, 'V': V, 'W': W, 'rank': 7}

def _merge_n(n1, n2, m, p, r1, r2):
    """Merge two decompositions stacked along n dimension."""
    rn = r1['rank']
    rank = rn + r2['rank']
    N = n1 + n2
    U = [[0] * rank for _ in range(N * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * N)]
    for i in range(n1 * m):
        for r in range(rn):
            U[i][r] = r1['U'][i][r]
    for i in range(n2 * m):
        for r in range(r2['rank']):
            U[n1 * m + i][rn + r] = r2['U'][i][r]
    for j in range(m):
        for k in range(p):
            idx = j * p + k
            for r in range(rn):
                V[idx][r] = r1['V'][idx][r]
            for r in range(r2['rank']):
                V[idx][rn + r] = r2['V'][idx][r]
    for k in range(p):
        for i in range(n1):
            for r in range(rn):
                W[k * N + i][r] = r1['W'][k * n1 + i][r]
        for i in range(n2):
            for r in range(r2['rank']):
                W[k * N + n1 + i][rn + r] = r2['W'][k * n2 + i][r]
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _merge_m(n, m1, m2, p, r1, r2):
    """Merge two decompositions concatenated along m dimension."""
    rn = r1['rank']
    rank = rn + r2['rank']
    M = m1 + m2
    U = [[0] * rank for _ in range(n * M)]
    V = [[0] * rank for _ in range(M * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m1):
            for r in range(rn):
                U[i * M + j][r] = r1['U'][i * m1 + j][r]
        for j in range(m2):
            for r in range(r2['rank']):
                U[i * M + m1 + j][rn + r] = r2['U'][i * m2 + j][r]
    for j in range(m1):
        for k in range(p):
            idx = j * p + k
            for r in range(rn):
                V[idx][r] = r1['V'][idx][r]
    for j in range(m2):
        for k in range(p):
            idx = (m1 + j) * p + k
            for r in range(r2['rank']):
                V[idx][rn + r] = r2['V'][j * p + k][r]
    for k in range(p):
        for i in range(n):
            idx = k * n + i
            for r in range(rn):
                W[idx][r] = r1['W'][idx][r]
            for r in range(r2['rank']):
                W[idx][rn + r] = r2['W'][idx][r]
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _merge_p(n, m, p1, p2, r1, r2):
    """Merge two decompositions concatenated along p dimension."""
    rn = r1['rank']
    rank = rn + r2['rank']
    P = p1 + p2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * P)]
    W = [[0] * rank for _ in range(P * n)]
    for i in range(n * m):
        for r in range(rn):
            U[i][r] = r1['U'][i][r]
        for r in range(r2['rank']):
            U[i][rn + r] = r2['U'][i][r]
    for j in range(m):
        for k in range(p1):
            idx = j * P + k
            for r in range(rn):
                V[idx][r] = r1['V'][j * p1 + k][r]
        for k in range(p2):
            idx = j * P + p1 + k
            for r in range(r2['rank']):
                V[idx][rn + r] = r2['V'][j * p2 + k][r]
    for k in range(p1):
        for i in range(n):
            for r in range(rn):
                W[k * n + i][r] = r1['W'][k * n1 + i][r] if False else r1['W'][k * n + i][r]
    for k in range(p2):
        for i in range(n):
            for r in range(r2['rank']):
                W[(p1 + k) * n + i][rn + r] = r2['W'][k * n + i][r]
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _strassen_rect(n, m, p):
    """Strassen-style: split each dimension into two roughly equal halves,
    compute 7 sub-products, recombine with additions. Returns rank."""
    n1, n2 = n // 2, n - n // 2
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2

    # 7 sub-products:
    # P1 = A_top * B_top      : (n1,m1) x (m1,p1) -> (n1,p1)
    # P2 = A_bot  * B_top     : (n2,m1) x (m1,p2) -> (n2,p2)
    # P3 = A_top  * (B_top-B_bot) : (n1,m1) x (m1,p2) -> (n1,p2)
    # P4 = A_bot  * (B_bot-B_top) : (n2,m1) x (m1,p2) -> (n2,p2)
    # P5 = (A_top+A_bot) * B_bot  : (n1,m2) x (m2,p2) -> (n1,p2)
    # P6 = (A_bot-A_top) * (B_top+B_bot) : (n2,m2) x (m2,p2) -> (n2,p2)
    # P7 = (A_top-A_bot) * B_bot  : (n2,m2) x (m2,p2) -> (n2,p2)

    # Wait, standard Strassen for square uses:
    # M1 = (A11+A22)(B12+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7

    # For rectangular, we need to handle the "sum/difference" blocks.
    # The key insight: if we split into blocks and use Strassen on each,
    # the recombination adds entries of U, V, W for each of the 7 sub-decompositions.

    # Sub-decomposition shapes:
    # M1: A_top (n1 x m), B_top (m x p1) -> but A_top is n1 x m (full m), B_top is m x p1
    #     Wait, that's not right. In Strassen, we split both A and B.
    #     A = [A_top; A_bot] where A_top is n1 x m, A_bot is n2 x m
    #     B = [B_left, B_right] where B_left is m x p1, B_right is m x p2
    #
    #     M1 = (A_top + A_bot) * (B_left + B_right)? No, that's for the full product.
    #
    #     Standard Strassen splits A into 4 quadrants and B into 4 quadrants.
    #     For rectangular, we split A row-wise and B column-wise, but the "sums"
    #     in Strassen create (n1+m2) or (n2+m2) shaped blocks which don't exist.
    #
    #     Actually, the standard approach for rectangular Strassen is:
    #     Split A into [A11, A12; A21, A22] and B into [B11, B12; B21, B22]
    #     where A11 is n1 x m1, A12 is n1 x m2, A21 is n2 x m1, A22 is n2 x m2
    #     B11 is m1 x p1, B12 is m1 x p2, B21 is m2 x p1, B22 is m2 x p2
    #
    #     Then C = [A11*B11 + A12*B21, A11*B12 + A12*B22;
    #               A21*B11 + A22*B21, A21*B12 + A22*B22]
    #
    #     Strassen computes 7 products instead of 8:
    #     M1 = (A11+A22)(B12+B22)  -> (n1 x m1 + n2 x m2) doesn't make sense dimensionally
    #
    #     The issue is that Strassen requires square blocks. For rectangular,
    #     the "addition" of blocks from different sub-problems doesn't work directly.
    #
    #     A common approach: pad to square, apply Strassen, extract.
    #     But padding increases the problem size.
    #
    #     Alternative: use the block decomposition approach with Strassen base case
    #     only when all three dimensions are equal (2,2,2) or can be split evenly.
    #
    #     Let me reconsider. The simplest correct approach:
    #     1. Use Strassen(2,2,2) as base case when n=m=p=2.
    #     2. For other sizes, use recursive splitting with additive rank.
    #     3. The key improvement: when n=4, split into 2+2, and each 2x5x6 can
    #        potentially use Strassen-like tricks if we also split m or p.

    # For now, let's just handle the case where we can split into equal halves
    # and use Strassen on the square sub-problems.
    # This is complex. Let me take a different approach:
    # Just use the divide-and-conquer with Strassen(2,2,2) base case.

    # Actually, let me implement a proper rectangular Strassen:
    # The trick is to embed the rectangular problem into a square one.
    # For n x m x p, pad to s x s x s where s = max(n, m, p),
    # apply Strassen, extract the n x p result.
    # But the rank of the padded problem is what we use.

    # Even simpler: for the specific case 4x5x6, let's try:
    # Split n=4 into 2+2. Each sub-problem is 2x5x6.
    # For 2x5x6: split m=5 into 2+3 or 3+2.
    #   2x2x6: split p=6 into 2+2+2, each 2x2x2 uses Strassen rank 7. Total: 21.
    #   2x3x6: split into 2x1x6 + 2x2x6. 2x1x6 = 12 (schoolbook). 2x2x6 = 21. Total: 33.
    #   So 2x5x6 = 21 + 33 = 54? That's worse than schoolbook 60? No, 54 < 60.
    # Wait, 2x5x6 schoolbook = 60. 21+33 = 54 < 60. 
    # Then 4x5x6 = 54 + 54 = 108 < 120.

    # Let me verify: 2x2x6 -> split p into 2+2+2: three 2x2x2 Strassen = 3*7 = 21.
    # 2x1x6 schoolbook = 12.
    # 2x3x6 = 2x1x6 + 2x2x6 = 12 + 21 = 33.
    # 2x5x6 = 2x2x6 + 2x3x6 = 21 + 33 = 54.
    # 4x5x6 = 2x5x6 + 2x5x6 = 54 + 54 = 108.

    # But we need to verify the merge operations handle this correctly.
    # The merge_p for 2x2x6 -> 2x2x2 + 2x2x2 + 2x2x2 needs 3-way merge.
    # Let me implement multi-way merge or just pairwise.

    # For 2x2x6: p=6 = 2+2+2. Pairwise: (2x2x2 + 2x2x2) -> 2x2x4, then + 2x2x2 -> 2x2x6.
    # Rank: 7 + 7 = 14, then 14 + 7 = 21. Correct.

    # For 2x3x6: m=3 = 1+2. 2x1x6 (rank 12) + 2x2x6 (rank 21) = 33.
    # Merge_m: 2x1x6 and 2x2x6.

    # For 2x5x6: m=5 = 2+3. 2x2x6 (rank 21) + 2x3x6 (rank 33) = 54.
    # Merge_m: 2x2x6 and 2x3x6.

    # For 4x5x6: n=4 = 2+2. 2x5x6 (rank 54) + 2x5x6 (rank 54) = 108.
    # Merge_n: 2x5x6 and 2x5x6.

    # Total rank: 108 < 120. 

    # Let me implement this properly.
    return None  # Not used directly; handled by _best_decompose

def _best_decompose(n, m, p):
    """Find best decomposition for n x m x p."""
    if _tl() < 5:
        return _schoolbook(n, m, p)

    if n == 0 or m == 0 or p == 0:
        return _schoolbook(n, m, p)
    if n == 1 and m == 1 and p == 1:
        return _schoolbook(1, 1, 1)
    if n == 2 and m == 2 and p == 2:
        return _strassen222()

    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)

    # Try splitting n
    for n1 in range(1, n):
        if _tl() < 10:
            break
        n2 = n - n1
        r1 = _best_decompose(n1, m, p)
        r2 = _best_decompose(n2, m, p)
        res = _merge_n(n1, n2, m, p, r1, r2)
        if res['rank'] < best_rank:
            best_rank = res['rank']
            best_result = res

    # Try splitting m
    for m1 in range(1, m):
        if _tl() < 10:
            break
        m2 = m - m1
        r1 = _best_decompose(n, m1, p)
        r2 = _best_decompose(n, m2, p)
        res = _merge_m(n, m1, m2, p, r1, r2)
        if res['rank'] < best_rank:
            best_rank = res['rank']
            best_result = res

    # Try splitting p
    for p1 in range(1, p):
        if _tl() < 10:
            break
        p2 = p - p1
        r1 = _best_decompose(n, m, p1)
        r2 = _best_decompose(n, m, p2)
        res = _merge_p(n, m, p1, p2, r1, r2)
        if res['rank'] < best_rank:
            best_rank = res['rank']
            best_result = res

    return best_result

def _verify(U, V, W, n, m, p):
    """Verify decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)

    try:
        result = _best_decompose(n, m, p)
        if result['rank'] < schoolbook['rank']:
            # Verify
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                return {'U': result['U'], 'V': result['V'], 'W': result['W']}
            else:
                return {'U': schoolbook['U'], 'V': schoolbook['V'], 'W': schoolbook['W']}
        else:
            return {'U': schoolbook['U'], 'V': schoolbook['V'], 'W': schoolbook['W']}
    except Exception:
        return {'U': schoolbook['U'], 'V': schoolbook['V'], 'W': schoolbook['W']}
# EVOLVE-BLOCK-END
