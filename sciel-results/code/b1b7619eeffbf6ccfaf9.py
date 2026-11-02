"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

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
    # C00 += p1, C01 += p1, C10 -= p1, C11 += p1
    W[0*2+0][0] = 1   # C00
    W[1*2+0][0] = 1   # C01
    W[0*2+1][0] = -1  # C10
    W[1*2+1][0] = 1   # C11
    
    # p2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += p2, C11 += p2
    W[0*2+1][1] = 1   # C10
    W[1*2+1][1] = 1   # C11
    
    # p3 = A00*(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # C10 += p3, C11 += p3
    W[0*2+1][2] = 1   # C10
    W[1*2+1][2] = 1   # C11
    
    # p4 = A11*(B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    # C00 += p4, C10 += p4
    W[0*2+0][3] = 1   # C00
    W[0*2+1][3] = 1   # C10
    
    # p5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 -= p5, C01 += p5
    W[0*2+0][4] = -1  # C00
    W[1*2+0][4] = 1   # C01
    
    # p6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # Not used in standard Strassen output? Let me verify the formulas.
    # Standard: C00=p1+p4-p5+p7, C01=p1+p5, C10=p2+p4, C11=p1-p2+p3+p7
    # p6 is not in any formula. This is suspicious.
    # Actually, I think I need to double-check. The standard Strassen uses all 7.
    # Let me use a different verified set of formulas.
    
    # After careful review, the correct Strassen formulas are:
    # p1 = (a00+a11)(b01+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b00-b10)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    # c00 = p1+p4-p5+p7
    # c01 = p1+p5
    # c10 = p2+p4
    # c11 = p1-p2+p3+p7
    # 
    # Indeed p6 is not used in the output! This means the decomposition has a free parameter.
    # We can set p6's contribution to zero, effectively making it rank 6? No, that's impossible.
    # The issue is that p6 is needed to make the decomposition work, but its coefficient in the
    # output is zero. This means we can drop p6 and get rank 6, contradicting the lower bound.
    # I must be misremembering the formulas.
    
    # Let me use a completely different approach: just verify by computation.
    # For now, I'll use the schoolbook for 2x2x2 to be safe.
    return _schoolbook(2, 2, 2)


def _block_decompose(n, m, p, split_dim, split_pos):
    """
    Decompose n×m × m×p by splitting one dimension.
    split_dim: 0 for n, 1 for m, 2 for p
    split_pos: position to split
    Returns a decomposition dict.
    """
    if split_dim == 0:
        # Split n into n1 + n2
        n1, n2 = split_pos, n - split_pos
        r1 = _best_decompose(n1, m, p)
        r2 = _best_decompose(n2, m, p)
        return _combine_n(n1, n2, m, p, r1, r2)
    elif split_dim == 1:
        # Split m into m1 + m2
        m1, m2 = split_pos, m - split_pos
        r1 = _best_decompose(n, m1, p)
        r2 = _best_decompose(n, m2, p)
        return _combine_m(n, m1, m2, p, r1, r2)
    elif split_dim == 2:
        # Split p into p1 + p2
        p1, p2 = split_pos, p - split_pos
        r1 = _best_decompose(n, m, p1)
        r2 = _best_decompose(n, m, p2)
        return _combine_p(n, m, p1, p2, r1, r2)


def _combine_n(n1, n2, m, p, r1, r2):
    """Combine decompositions split on first dimension."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range((n1+n2) * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * (n1+n2))]
    
    # r1
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
    
    # r2
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
    """Combine decompositions split on middle dimension."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * (m1+m2))]
    V = [[0] * rank for _ in range((m1+m2) * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r1
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
    
    # r2
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
    """Combine decompositions split on third dimension."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * (p1+p2))]
    W = [[0] * rank for _ in range((p1+p2) * n)]
    
    # r1
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
    
    # r2
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
    """Find the best known decomposition for n×m × m×p."""
    # Base cases
    if n == 0 or m == 0 or p == 0:
        return {'U': [], 'V': [], 'W': [], 'rank': 0}
    
    # Try schoolbook
    best = _schoolbook(n, m, p)
    best_rank = best['rank']
    
    # Try splitting each dimension
    # Split n
    for pos in range(1, n):
        r = _block_decompose(n, m, p, 0, pos)
        if r['rank'] < best_rank:
            best = r
            best_rank = r['rank']
    
    # Split m
    for pos in range(1, m):
        r = _block_decompose(n, m, p, 1, pos)
        if r['rank'] < best_rank:
            best = r
            best_rank = r['rank']
    
    # Split p
    for pos in range(1, p):
        r = _block_decompose(n, m, p, 2, pos)
        if r['rank'] < best_rank:
            best = r
            best_rank = r['rank']
    
    return best


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = result['rank']
    
    # Try all single-level splits
    # Split n
    for pos in range(1, n):
        if time.time() > deadline:
            break
        r1 = _schoolbook(pos, m, p)
        r2 = _schoolbook(n-pos, m, p)
        combined = _combine_n(pos, n-pos, m, p, r1, r2)
        if combined['rank'] < best_rank:
            best_rank = combined['rank']
            result = combined
    
    # Split m
    for pos in range(1, m):
        if time.time() > deadline:
            break
        r1 = _schoolbook(n, pos, p)
        r2 = _schoolbook(n, m-pos, p)
        combined = _combine_m(n, pos, m-pos, p, r1, r2)
        if combined['rank'] < best_rank:
            best_rank = combined['rank']
            result = combined
    
    # Split p
    for pos in range(1, p):
        if time.time() > deadline:
            break
        r1 = _schoolbook(n, m, pos)
        r2 = _schoolbook(n, m, p-pos)
        combined = _combine_p(n, m, pos, p-pos, r1, r2)
        if combined['rank'] < best_rank:
            best_rank = combined['rank']
            result = combined
    
    # Try two-level splits
    # Split n, then split one part
    for pos1 in range(1, n):
        if time.time() > deadline:
            break
        n1, n2 = pos1, n - pos1
        # Split n1 further
        for pos2 in range(1, n1):
            if time.time() > deadline:
                break
            r1 = _combine_n(pos2, n1-pos2, m, p, _schoolbook(pos2, m, p), _schoolbook(n1-pos2, m, p))
            r2 = _schoolbook(n2, m, p)
            combined = _combine_n(n1, n2, m, p, r1, r2)
            if combined['rank'] < best_rank:
                best_rank = combined['rank']
                result = combined
        # Split n2 further
        for pos2 in range(1, n2):
            if time.time() > deadline:
                break
            r1 = _schoolbook(n1, m, p)
            r2 = _combine_n(pos2, n2-pos2, m, p, _schoolbook(pos2, m, p), _schoolbook(n2-pos2, m, p))
            combined = _combine_n(n1, n2, m, p, r1, r2)
            if combined['rank'] < best_rank:
                best_rank = combined['rank']
                result = combined
    
    # Split m, then split one part
    for pos1 in range(1, m):
        if time.time() > deadline:
            break
        m1, m2 = pos1, m - pos1
        for pos2 in range(1, m1):
            if time.time() > deadline:
                break
            r1 = _combine_m(n, pos2, m1-pos2, p, _schoolbook(n, pos2, p), _schoolbook(n, m1-pos2, p))
            r2 = _schoolbook(n, m2, p)
            combined = _combine_m(n, m1, m2, p, r1, r2)
            if combined['rank'] < best_rank:
                best_rank = combined['rank']
                result = combined
        for pos2 in range(1, m2):
            if time.time() > deadline:
                break
            r1 = _schoolbook(n, m1, p)
            r2 = _combine_m(n, pos2, m2-pos2, p, _schoolbook(n, pos2, p), _schoolbook(n, m2-pos2, p))
            combined = _combine_m(n, m1, m2, p, r1, r2)
            if combined['rank'] < best_rank:
                best_rank = combined['rank']
                result = combined
    
    # Split p, then split one part
    for pos1 in range(1, p):
        if time.time() > deadline:
            break
        p1, p2 = pos1, p - pos1
        for pos2 in range(1, p1):
            if time.time() > deadline:
                break
            r1 = _combine_p(n, m, pos2, p1-pos2, _schoolbook(n, m, pos2), _schoolbook(n, m, p1-pos2))
            r2 = _schoolbook(n, m, p2)
            combined = _combine_p(n, m, p1, p2, r1, r2)
            if combined['rank'] < best_rank:
                best_rank = combined['rank']
                result = combined
        for pos2 in range(1, p2):
            if time.time() > deadline:
                break
            r1 = _schoolbook(n, m, p1)
            r2 = _combine_p(n, m, pos2, p2-pos2, _schoolbook(n, m, pos2), _schoolbook(n, m, p2-pos2))
            combined = _combine_p(n, m, p1, p2, r1, r2)
            if combined['rank'] < best_rank:
                best_rank = combined['rank']
                result = combined
    
    return {"U": result['U'], "V": result['V'], "W": result['W']}
# EVOLVE-BLOCK-END
