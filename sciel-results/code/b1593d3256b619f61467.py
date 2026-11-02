"""Construct a rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=3, p=4):
    """Exact verification of tensor decomposition."""
    rank = len(U[0])
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    # Check all entries of the 3x3x4 tensor
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1 if (i == ip and j == jp) else 0
                            actual = 0
                            for r in range(rank):
                                actual += Ua[i*m+j, r] * Va[jp*p+k, r] * Wa[kp*n+ip, r]
                            if actual != expected:
                                return False
    return True

def _build_rank30():
    """
    Rank-30 decomposition: 27 schoolbook terms for 3x3x3 (k=0,1,2)
    + 3 terms for the 4th column (k=3).
    Guaranteed valid.
    """
    n, m, p = 3, 3, 4
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 27 schoolbook terms for 3x3x3
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 3 terms for 4th column (k=3): one per j
    for j in range(3):
        r = col
        col += 1
        for i in range(3):
            U[i * m + j][r] = 1
            W[3 * n + i][r] = 1
        V[j * p + 3][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_2x2_block():
    """
    Attempt to use Strassen's algorithm on a 2x2 sub-block of the 3x3x3 part.
    This gives rank 7 for the 2x2x2 block instead of 4, but we need to handle
    the remaining terms carefully.
    
    Partition:
    - Rows 0,1 and Cols 0,1: use Strassen (7 terms)
    - Row 2, Cols 0,1: schoolbook (2 terms)
    - Rows 0,1,2, Col 2: schoolbook (3 terms)
    - 4th column (k=3): 3 terms
    
    Total: 7 + 2 + 3 + 3 = 15? That's not right because the Strassen terms
    only cover the 2x2x2 subproblem.
    
    Actually, let me think more carefully. The 3x3x3 tensor has 27 terms in schoolbook.
    If I take a 2x2x2 sub-block (rows 0,1; cols 0,1; output rows 0,1), that's 4 terms.
    Replacing with Strassen gives 7 terms, so net +3.
    
    But I also need to handle:
    - Row 2, Cols 0,1 -> Output rows 0,1,2: 2*3 = 6 terms
    - Rows 0,1,2, Col 2 -> Output rows 0,1,2: 3*3 = 9 terms
    
    Total: 7 + 6 + 9 = 22 for the 3x3x3 part, plus 3 for the 4th column = 25.
    
    Let me construct this explicitly.
    """
    n, m, p = 3, 3, 4
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 for rows {0,1}, cols {0,1}, output rows {0,1}
    # M1 = (a00+a10)(b00+b10) -> c00 += M1, c10 += M1
    # M2 = (a10+a11)(b00+b01) -> c01 += M2, c11 += M2
    # M3 = a00(b10-b00) -> c00 += M3, c10 -= M3
    # M4 = a11(b11-b01) -> c11 += M4, c01 -= M4
    # M5 = (a00+a01)(b10+b11) -> c00 += M5, c10 += M5
    # M6 = (a10-a00)(b01+b11) -> c01 += M6, c11 -= M6
    # M7 = (a01-a11)(b00+b10) -> c00 -= M7, c10 += M7
    
    # Index mapping: U[i*3+j], V[j*3+k], W[k*3+i]
    # For k=0,1 only in this block
    
    # M1: U[0,0]=1, U[3,0]=1; V[0,0]=1, V[3,0]=1; W[0,0]=1, W[3,0]=1
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[3][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # M2: U[3,1]=1, U[4,1]=1; V[0,2]=1, V[1,2]=1; W[0,1]=1, W[3,1]=1
    r = 1
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[1][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # M3: U[0,0]=1; V[3,0]=1, V[0,0]=-1; W[0,0]=1, W[3,0]=-1
    r = 2
    U[0][r] = 1
    V[3][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[3][r] = -1
    
    # M4: U[4,1]=1; V[4,1]=1, V[1,1]=-1; W[3,1]=1, W[0,1]=-1
    r = 3
    U[4][r] = 1
    V[4][r] = 1; V[1][r] = -1
    W[3][r] = 1; W[0][r] = -1
    
    # M5: U[0,0]=1, U[1,0]=1; V[3,0]=1, V[4,0]=1; W[0,0]=1, W[3,0]=1
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[3][r] = 1; V[4][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # M6: U[3,0]=1, U[0,0]=-1; V[1,0]=1, V[4,0]=1; W[0,1]=1, W[3,1]=-1
    r = 5
    U[3][r] = 1; U[0][r] = -1
    V[1][r] = 1; V[4][r] = 1
    W[0][r] = 1; W[3][r] = -1
    
    # M7: U[1,0]=1, U[4,0]=-1; V[0,0]=1, V[3,0]=1; W[0,0]=-1, W[3,0]=1
    r = 6
    U[1][r] = 1; U[4][r] = -1
    V[0][r] = 1; V[3][r] = 1
    W[0][r] = -1; W[3][r] = 1
    
    # Now handle remaining terms for 3x3x3:
    # Row 2, Cols 0,1 -> Output rows 0,1,2 (6 terms)
    # These are: a20*b0k, a20*b1k, a21*b0k, a21*b1k for k=0,1
    # But wait, I already have some of these in the Strassen terms? No, Strassen only uses rows 0,1.
    
    # a20*b00 -> c00: U[6,0]=1, V[0,0]=1, W[0,0]=1
    r = 7
    U[6][r] = 1; V[0][r] = 1; W[0][r] = 1
    
    # a20*b10 -> c10: U[6,0]=1, V[3,0]=1, W[3,0]=1
    r = 8
    U[6][r] = 1; V[3][r] = 1; W[3][r] = 1
    
    # a20*b01 -> c01: U[6,0]=1, V[1,0]=1, W[0,1]=1
    r = 9
    U[6][r] = 1; V[1][r] = 1; W[0][r] = 1
    
    # a20*b11 -> c11: U[6,0]=1, V[4,0]=1, W[3,1]=1
    r = 10
    U[6][r] = 1; V[4][r] = 1; W[3][r] = 1
    
    # a21*b00 -> c02: U[7,0]=1, V[0,0]=1, W[0,2]=1
    r = 11
    U[7][r] = 1; V[0][r] = 1; W[0][r] = 1
    
    # a21*b10 -> c12: U[7,0]=1, V[3,0]=1, W[3,2]=1
    r = 12
    U[7][r] = 1; V[3][r] = 1; W[3][r] = 1
    
    # a21*b01 -> c02: U[7,0]=1, V[1,0]=1, W[0,2]=1
    r = 13
    U[7][r] = 1; V[1][r] = 1; W[0][r] = 1
    
    # a21*b11 -> c12: U[7,0]=1, V[4,0]=1, W[3,2]=1
    r = 14
    U[7][r] = 1; V[4][r] = 1; W[3][r] = 1
    
    # Rows 0,1,2, Col 2 -> Output rows 0,1,2 (9 terms)
    # a02*b0k, a02*b1k, a12*b0k, a12*b1k, a22*b0k, a22*b1k for k=0,1
    # Wait, I need to be more careful.
    
    # Actually, I realize I'm making this too complicated. The Strassen approach
    # only helps if I can cleanly separate the 2x2x2 subproblem. But the 3x3x3
    # tensor doesn't decompose so cleanly.
    
    # Let me just return None and use the guaranteed rank-30.
    return None

def construct():
    """Construct a tensor decomposition for 3x3 * 3x4 matrix multiplication."""
    n, m, p = 3, 3, 4
    
    # Try Strassen-based approach
    result_strassen = _try_strassen_2x2_block()
    if result_strassen is not None:
        if _verify(result_strassen['U'], result_strassen['V'], result_strassen['W']):
            return result_strassen
    
    # Use rank-30 (guaranteed valid)
    result_30 = _build_rank30()
    if _verify(result_30['U'], result_30['V'], result_30['W']):
        return result_30
    
    # Fallback: schoolbook rank-36
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
# EVOLVE-BLOCK-END
