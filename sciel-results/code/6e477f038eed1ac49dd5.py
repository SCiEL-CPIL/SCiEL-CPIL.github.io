"""Matrix multiplication 3x3 by 3x4: rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using Fraction arithmetic."""
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

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r] for r in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit
    
    # Attempt to construct a rank-34 decomposition using Strassen on 2x2 blocks
    # Partition A(3x3) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Partition B(3x4) = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    # C(3x3) = [C11 C12; C21 C22] where C11 is 2x2, C12 is 2x1, C21 is 1x2, C22 is 1x1
    #
    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x1*1x2)
    # C12 = A11*B12 + A12*B22  (2x1 = 2x2*2x1 + 2x1*1x1)
    # C21 = A21*B11 + A22*B21  (1x2 = 1x2*2x2 + 1x1*1x2)
    # C22 = A21*B12 + A22*B22  (1x1 = 1x2*2x1 + 1x1*1x1)
    #
    # Using Strassen for the 2x2*2x2 block in C11:
    # M1 = (A11[0,0]+A11[1,1])*(B11[0,0]+B11[1,1])  -- but this is for full 2x2*2x2
    # Actually, let me be more careful.
    #
    # For 2x2 * 2x2 = 2x2, Strassen uses 7 multiplications:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For the remaining blocks, use schoolbook:
    # A12*B21: 2x1 * 1x2 = 2x2, schoolbook rank = 4
    # A11*B12: 2x2 * 2x1 = 2x1, schoolbook rank = 4
    # A12*B22: 2x1 * 1x1 = 2x1, schoolbook rank = 2
    # A21*B11: 1x2 * 2x2 = 1x2, schoolbook rank = 4
    # A22*B21: 1x1 * 1x2 = 1x2, schoolbook rank = 2
    # A21*B12: 1x2 * 2x1 = 1x1, schoolbook rank = 2
    # A22*B22: 1x1 * 1x1 = 1x1, schoolbook rank = 1
    #
    # Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26... that doesn't seem right for rank 34.
    #
    # Let me reconsider. The schoolbook rank is 36. If we can save 2, we get 34.
    # 
    # Actually, let me just try a direct construction. The key insight from the knowledge
    # is that using Strassen on two 2x2x2 subproblems saves 6 terms. But Strassen uses 7
    # vs schoolbook 4, so it uses MORE terms. That can't be right.
    #
    # Wait - I think the "saves" refers to the overall structure. Let me just try to
    # construct a valid rank-34 decomposition directly and verify it.
    #
    # For now, return schoolbook rank 36 as a safe baseline.
    # A proper rank-reduction would require more careful analysis.
    
    U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
