# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # seconds; evaluator kills at 150s
_MEMO = {}

def _time_left():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p."""
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
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _strassen_2x2x2():
    """
    Correct Strassen decomposition for 2×2 × 2×2.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # p1 = (A00+A11)*(B01+B11)
    U[0][0] = 1; U[3][0] = 1
    V[1][0] = 1; V[3][0] = 1
    W[0*2+0][0] = 1   # C00
    W[1*2+0][0] = 1   # C01
    W[0*2+1][0] = -1  # C10
    W[1*2+1][0] = 1   # C11
    
    # p2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0*2+1][1] = 1   # C10
    W[1*2+1][1] = 1   # C11
    
    # p3 = A00*(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0*2+1][2] = 1   # C10
    W[1*2+1][2] = 1   # C11
    
    # p4 = A11*(B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0*2+0][3] = 1   # C00
    W[0*2+1][3] = 1   # C10
    
    # p5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0*2+0][4] = -1  # C00
    W[1*2+0][4] = 1   # C01
    
    # p6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0*2+0][5] = 1   # C00
    W[1*2+1][5] = 1   # C11
    
    # p7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0*2+0][6] = 1   # C00
    W[1*2+1][6] = 1   # C11
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_n(n1, n2, m, p, r1, r2):
    """Combine decompositions split on first dimension (n)."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range((n1+n2) * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * (n1+n2))]
    
    # r1 (top block)
    for i in range(n1 * m):
        for r in range(r1['rank']):
            U[i][r] = r1['U'][i][r]
    for j in range(m):
        for k in range(p):
            for r in range(r1['rank']):
                V[j*p+k][r] = r1['V'][j*p+k][r]
    for k in range(p):
        for i in range(n1):
            for r in range(r1['rank']):
                W[k*(n1+n2)+i][r] = r1['W'][k*n1+i][r]
    
    # r2 (bottom block)
    off = r1['rank']
    for i in range(n2 * m):
        for r in range(r2['rank']):
            U[n1*m + i][off+r] = r2['U'][i][r]
    for j in range(m):
        for k in range(p):
            for r in range(r2['rank']):
                V[j*p+k][off+r] = r2['V'][j*p+k][r]
    for k in range(p):
        for i in range(n2):
            for r in range(r2['rank']):
                W[k*(n1+n2)+n1+i][off+r] = r2['W'][k*n2+i][r]
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_m(n, m1, m2, p, r1, r2):
    """Combine decompositions split on middle dimension (m)."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * (m1+m2))]
    V = [[0] * rank for _ in range((m1+m2) * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r1 (left block)
    for i in range(n * m1):
        for r in range(r1['rank']):
            U[i][r] = r1['U'][i][r]
    for j in range(m1):
        for k in range(p):
            for r in range(r1['rank']):
                V[j*p+k][r] = r1['V'][j*p+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r1['rank']):
                W[k*n+i][r] = r1['W'][k*n+i][r]
    
    # r2 (right block)
    off = r1['rank']
    for i in range(n * m2):
        for r in range(r2['rank']):
            U[n*m1 + i][off+r] = r2['U'][i][r]
    for j in range(m2):
        for k in range(p):
            for r in range(r2['rank']):
                V[(m1+j)*p+k][off+r] = r2['V'][j*p+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r2['rank']):
                W[k*n+i][off+r] = r2['W'][k*n+i][r]
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_p(n, m, p1, p2, r1, r2):
    """Combine decompositions split on third dimension (p)."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * (p1+p2))]
    W = [[0] * rank for _ in range((p1+p2) * n)]
    
    # r1 (left block)
    for i in range(n * m):
        for r in range(r1['rank']):
            U[i][r] = r1['U'][i][r]
    for j in range(m):
        for k in range(p1):
            for r in range(r1['rank']):
                V[j*(p1+p2)+k][r] = r1['V'][j*p1+k][r]
    for k in range(p1):
        for i in range(n):
            for r in range(r1['rank']):
                W[k*n+i][r] = r1['W'][k*n+i][r]
    
    # r2 (right block)
    off = r1['rank']
    for i in range(n * m):
        for r in range(r2['rank']):
            U[i][off+r] = r2['U'][i][r]
    for j in range(m):
        for k in range(p2):
            for r in range(r2['rank']):
                V[j*(p1+p2)+p1+k][off+r] = r2['V'][j*p2+k][r]
    for k in range(p2):
        for i in range(n):
            for r in range(r2['rank']):
                W[(p1+k)*n+i][off+r] = r2['W'][k*n+i][r]
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _best_decompose(n, m, p):
    """Find the best known decomposition for n×m × m×p using divide and conquer."""
    if time.time() > _DEADLINE - 5:
        return _schoolbook(n, m, p)
        
    key = (n, m, p)
    if key in _MEMO:
        return _MEMO[key]

    # Base cases
    if n == 0 or m == 0 or p == 0:
        res = _schoolbook(n, m, p)
        _MEMO[key] = res
        return res
        
    if n == 1 and m == 1 and p == 1:
        res = _schoolbook(1, 1, 1)
        _MEMO[key] = res
        return res
        
    if n == 2 and m == 2 and p == 2:
        res = _strassen_2x2x2()
        _MEMO[key] = res
        return res
        
    # If any dimension is 1, schoolbook is optimal (rank = n*m*p)
    # Actually, we can still try to split, but it won't improve over schoolbook
    # because R(1, m, p) = m*p.
    if n == 1 or m == 1 or p == 1:
        res = _schoolbook(n, m, p)
        _MEMO[key] = res
        return res

    # Try all possible splits and pick the one with minimum rank
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        if time.time() > _DEADLINE - 10:
            break
        r1 = _best_decompose(n1, m, p)
        r2 = _best_decompose(n2, m, p)
        res = _combine_n(n1, n2, m, p, r1, r2)
        if res['rank'] < best_rank:
            best_rank = res['rank']
            best_result = res
            
    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        if time.time() > _DEADLINE - 10:
            break
        r1 = _best_decompose(n, m1, p)
        r2 = _best_decompose(n, m2, p)
        res = _combine_m(n, m1, m2, p, r1, r2)
        if res['rank'] < best_rank:
            best_rank = res['rank']
            best_result = res
            
    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        if time.time() > _DEADLINE - 10:
            break
        r1 = _best_decompose(n, m, p1)
        r2 = _best_decompose(n, m, p2)
        res = _combine_p(n, m, p1, p2, r1, r2)
        if res['rank'] < best_rank:
            best_rank = res['rank']
            best_result = res
            
    _MEMO[key] = best_result
    return best_result

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try divide and conquer
    try:
        result = _best_decompose(n, m, p)
        # Verify result
        if result['rank'] < schoolbook['rank']:
            return {'U': result['U'], 'V': result['V'], 'W': result['W']}
        else:
            return {'U': schoolbook['U'], 'V': schoolbook['V'], 'W': schoolbook['W']}
    except Exception:
        return {'U': schoolbook['U'], 'V': schoolbook['V'], 'W': schoolbook['W']}
# EVOLVE-BLOCK-END
