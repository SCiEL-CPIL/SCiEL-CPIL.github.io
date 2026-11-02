"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

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

def _verify_decomp(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    rank = len(U[0])
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T is the identity tensor for matmul: T[i, j, k] = 1 if the indices correspond to a valid multiplication path?
    # Actually, the matrix multiplication tensor T_{ijk} is defined such that (AB)_{ik} = sum_j A_{ij} B_{jk} = sum_{j,r} U_{ij,r} V_{jr} W_{rk}
    # Wait, the standard decomposition is:
    # (AB)_{ik} = sum_{j=1}^m A_{ij} B_{jk}
    # The tensor T has entries T_{ijk} = 1 if we are summing over j.
    # Specifically, T_{ijk} = 1 for all i, j, k.
    # And the decomposition requires: sum_{r=1}^R U_{ij,r} V_{jk,r} W_{ki,r} = T_{ijk} = 1 for all i,j,k.
    # Note the index order in W is k*n+i, which corresponds to W_{ki,r}.
    
    # Let's check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # We can do this by iterating or using einsum.
    # Since n,m,p are small, we can just iterate.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _combine_m(n, m1, m2, p, r1, r2):
    """
    Combine two decompositions for (n x m1) * (m1 x p) and (n x m2) * (m2 x p)
    to form a decomposition for (n x (m1+m2)) * ((m1+m2) x p).
    """
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * (m1 + m2))]
    V = [[0] * rank for _ in range((m1 + m2) * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: from r1 (uses first m1 columns of A, first m1 rows of B)
    for i in range(n):
        for j in range(m1):
            for r in range(r1['rank']):
                U[i*(m1+m2)+j][r] = r1['U'][i*m1+j][r]
    for j in range(m1):
        for k in range(p):
            for r in range(r1['rank']):
                V[j*p+k][r] = r1['V'][j*p+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r1['rank']):
                W[k*n+i][r] = r1['W'][k*n+i][r]
                
    # Part 2: from r2 (uses next m2 columns of A, next m2 rows of B)
    off = r1['rank']
    for i in range(n):
        for j in range(m2):
            for r in range(r2['rank']):
                U[i*(m1+m2)+(m1+j)][off+r] = r2['U'][i*m2+j][r]
    for j in range(m2):
        for k in range(p):
            for r in range(r2['rank']):
                V[(m1+j)*p+k][off+r] = r2['V'][j*p+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r2['rank']):
                W[k*n+i][off+r] = r2['W'][k*n+i][r]
                
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_n(n1, n2, m, p, r1, r2):
    """
    Combine two decompositions for (n1 x m) * (m x p) and (n2 x m) * (m x p)
    to form a decomposition for ((n1+n2) x m) * (m x p).
    """
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range((n1 + n2) * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * (n1 + n2))]
    
    # Part 1: from r1 (top n1 rows)
    for i in range(n1):
        for j in range(m):
            for r in range(r1['rank']):
                U[i*m+j][r] = r1['U'][i*m+j][r]
    for j in range(m):
        for k in range(p):
            for r in range(r1['rank']):
                V[j*p+k][r] = r1['V'][j*p+k][r]
    for k in range(p):
        for i in range(n1):
            for r in range(r1['rank']):
                W[k*(n1+n2)+i][r] = r1['W'][k*n1+i][r]
                
    # Part 2: from r2 (bottom n2 rows)
    off = r1['rank']
    for i in range(n2):
        for j in range(m):
            for r in range(r2['rank']):
                U[(n1+i)*m+j][off+r] = r2['U'][i*m+j][r]
    for j in range(m):
        for k in range(p):
            for r in range(r2['rank']):
                V[j*p+k][off+r] = r2['V'][j*p+k][r]
    for k in range(p):
        for i in range(n2):
            for r in range(r2['rank']):
                W[k*(n1+n2)+(n1+i)][off+r] = r2['W'][k*n2+i][off+r]
                
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_p(n, m, p1, p2, r1, r2):
    """
    Combine two decompositions for (n x m) * (m x p1) and (n x m) * (m x p2)
    to form a decomposition for (n x m) * (m x (p1+p2)).
    """
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * (p1 + p2))]
    W = [[0] * rank for _ in range((p1 + p2) * n)]
    
    # Part 1: from r1 (first p1 columns of C)
    for i in range(n):
        for j in range(m):
            for r in range(r1['rank']):
                U[i*m+j][r] = r1['U'][i*m+j][r]
    for j in range(m):
        for k in range(p1):
            for r in range(r1['rank']):
                V[j*(p1+p2)+k][r] = r1['V'][j*p1+k][r]
    for k in range(p1):
        for i in range(n):
            for r in range(r1['rank']):
                W[k*n+i][r] = r1['W'][k*n+i][r]
                
    # Part 2: from r2 (next p2 columns of C)
    off = r1['rank']
    for i in range(n):
        for j in range(m):
            for r in range(r2['rank']):
                U[i*m+j][off+r] = r2['U'][i*m+j][r]
    for j in range(m):
        for k in range(p2):
            for r in range(r2['rank']):
                V[j*(p1+p2)+(p1+k)][off+r] = r2['V'][j*p2+k][r]
    for k in range(p2):
        for i in range(n):
            for r in range(r2['rank']):
                W[(p1+k)*n+i][off+r] = r2['W'][k*n+i][off+r]
                
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _best_decompose(n, m, p):
    """
    Find the best known decomposition for n x m * m x p.
    Uses memoization and tries all block decompositions.
    """
    if (n, m, p) in _memo:
        return _memo[(n, m, p)]
        
    best = _schoolbook(n, m, p)
    
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r1 = _best_decompose(n, m1, p)
        r2 = _best_decompose(n, m2, p)
        combined = _combine_m(n, m1, m2, p, r1, r2)
        if combined['rank'] < best['rank']:
            best = combined
            
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1 = _best_decompose(n1, m, p)
        r2 = _best_decompose(n2, m, p)
        combined = _combine_n(n1, n2, m, p, r1, r2)
        if combined['rank'] < best['rank']:
            best = combined
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1 = _best_decompose(n, m, p1)
        r2 = _best_decompose(n, m, p2)
        combined = _combine_p(n, m, p1, p2, r1, r2)
        if combined['rank'] < best['rank']:
            best = combined
            
    _memo[(n, m, p)] = best
    return best

_memo = {}

def construct():
    """
    Construct a low-rank decomposition for 4x5 * 5x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Clear memo to ensure fresh state if needed, though _best_decompose handles it
    _memo.clear()
    
    # Try to find the best block decomposition
    # This might take a bit, but for small dimensions it should be fast.
    # However, we have a 120s limit.
    
    best = None
    try:
        best = _best_decompose(n, m, p)
    except Exception:
        best = _schoolbook(n, m, p)
        
    # Verify the result
    if not _verify_decomp(best['U'], best['V'], best['W'], n, m, p):
        # Fallback to schoolbook if verification fails
        best = _schoolbook(n, m, p)
        
    return {"U": best['U'], "V": best['V'], "W": best['W']}
# EVOLVE-BLOCK-END
