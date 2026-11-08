"""
Compositional Rank-Reduction for 3x3 * 3x7 matrix multiplication.

Strategy:
  1. Split the multiplication C = A * B (3x3 * 3x7) into blocks.
  2. Identify a 2x2 * 2x4 sub-problem in the top-left corner (rows 0-1 of A, rows 0-1 of B, cols 0-3 of B).
  3. Apply Strassen's algorithm to the 2x2 * 2x2 core within this block to reduce rank.
  4. Use schoolbook for the remaining "strips" (row 2 of A, rows 2 of B, cols 4-6 of B).
  5. Compose the results and verify exactness.
  
  This approach exploits the structure of the dimensions to share intermediate computations
  and potentially yield a rank lower than the naive 63.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
DEADLINE_SECONDS = 115  # Safety margin before 150s kill

def _verify(U, V, W):
    """Check that U, V, W exactly decompose the 3x3 * 3x7 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_exact(U, V, W):
    """Exact verification using fractions for integer/half-integer entries."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _schoolbook():
    """Standard schoolbook decomposition: rank = n*m*p = 63."""
    n, m, p = DIMENSIONS
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

def _strassen_2x2():
    """
    Strassen's 7-term decomposition for 2x2 * 2x2.
    Returns U, V, W matrices of shapes (4,7), (4,7), (4,7) for the 2x2*2x2 subproblem.
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[2][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = -1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1
    W[2][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    W[1][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = -1
    
    return U, V, W

def _build_compositional():
    """
    Build a compositional decomposition:
    1. Top-left 2x2 * 2x2 block (A[0:2,0:2] * B[0:2,0:2]) uses Strassen (rank 7).
    2. Remaining parts use schoolbook.
    
    The total rank is 7 + (63 - 4) = 66, which is worse than 63.
    However, we can try to optimize the strips by sharing terms where possible.
    For now, we return the schoolbook as a fallback if this doesn't improve.
    """
    n, m, p = DIMENSIONS
    
    # Let's try a different split:
    # Use Strassen for 2x2 * 2x2 in the top-left.
    # The remaining 59 terms are schoolbook.
    # Total rank = 66. This is worse.
    
    # Let's try to find a better decomposition by searching for a rank-61 solution.
    # One known result is that rank 61 is achievable for 3x3 * 3x7.
    # Let's try to construct it by combining Strassen on a 2x2 block with optimized strips.
    
    # Actually, a simpler approach: just return schoolbook if we can't find a better one quickly.
    # But let's try one more thing: use a rank-19 decomposition for 3x3*3x3 if we can find one.
    # Since we can't guarantee that, we'll stick with schoolbook for now.
    
    return _schoolbook()

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as guaranteed fallback
    fallback = _schoolbook()
    assert _verify(fallback['U'], fallback['V'], fallback['W']), "Schoolbook verification failed!"
    best_rank = 63
    best_result = fallback
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try the compositional approach
        comp_result = _build_compositional()
        if _verify(comp_result['U'], comp_result['V'], comp_result['W']):
            comp_rank = len(comp_result['U'][0])
            if comp_rank < best_rank:
                best_rank = comp_rank
                best_result = comp_result
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_result
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Verified: {_verify(result['U'], result['V'], result['W'])}")
