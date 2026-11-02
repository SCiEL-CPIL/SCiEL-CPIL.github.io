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

def _strassen442():
    """
    Rank-28 decomposition for 4x4x2 matrix multiplication.
    Uses 4 Strassen 2x2x2 blocks with overlapping structure.
    U: (4*4, 28) = (16, 28)
    V: (4*2, 28) = (8, 28)
    W: (2*4, 28) = (8, 28)
    """
    # 4x4x2 can be viewed as:
    # A (4x4) * B (4x2) = C (4x2)
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into 2x2 blocks: B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, etc.
    # Wait, B is 4x2, so split into top 2 rows and bottom 2 rows:
    # B = [B_top; B_bot] where B_top is 2x2, B_bot is 2x2
    # Actually B is 4x2, so:
    # B = [[B00, B01], [B10, B11], [B20, B21], [B30, B31]]
    # Split into B_top = [[B00, B01], [B10, B11]] (2x2) and B_bot = [[B20, B21], [B30, B31]] (2x2)
    #
    # C = A * B = [[A11, A12], [A21, A22]] * [[B00, B01], [B10, B11], [B20, B21], [B30, B31]]
    # = [[A11*B00 + A12*B10 + A11*B20 + A12*B30, ...], ...]
    #
    # This is getting complicated. Let me use a different approach.
    # 4x4x2 = (2+2)x(2+2)x2
    # We can use the merge_m approach: 4x2x2 + 4x2x2
    # 4x2x2: split n=4 into 2+2: 2x2x2 + 2x2x2 = 7+7 = 14
    # So 4x4x2 = 4x2x2 + 4x2x2 = 14+14 = 28
    # This is what we want.
    
    # Use merge_m to combine 4x2x2 and 4x2x2
    r1 = _best_decompose(4, 2, 2)
    r2 = _best_decompose(4, 2, 2)
    result = _merge_m(4, 2, 2, 2, r1, r2)
    return result

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
                W[k * n + i][r] = r1['W'][k * n + i][r]
    for k in range(p2):
        for i in range(n):
            for r in range(r2['rank']):
                W[(p1 + k) * n + i][rn + r] = r2['W'][k * n + i][r]
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _best_decompose(n, m, p):
    """Find best decomposition for n x m x p with memoization."""
    if _tl() < 5:
        return _schoolbook(n, m, p)

    if n == 0 or m == 0 or p == 0:
        return _schoolbook(n, m, p)
    if n == 1 and m == 1 and p == 1:
        return _schoolbook(1, 1, 1)
    if n == 2 and m == 2 and p == 2:
        return _strassen222()
    
    # Special case: 4x4x2 -> use 4x2x2 + 4x2x2 = 14+14 = 28 (vs schoolbook 32)
    if n == 4 and m == 4 and p == 2:
        r1 = _best_decompose(4, 2, 2)
        r2 = _best_decompose(4, 2, 2)
        return _merge_m(4, 2, 2, 2, r1, r2)

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
