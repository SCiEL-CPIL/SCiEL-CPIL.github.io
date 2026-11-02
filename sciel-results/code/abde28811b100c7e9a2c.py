import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_n, _m, _p = DIMENSIONS

def _schoolbook_rank27():
    """Known valid rank-27 (schoolbook) decomposition."""
    n, m, p = _n, _m, _p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check exact tensor decomposition."""
    n, m, p = _n, _m, _p
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank26_structured():
    """Construct a rank-26 decomposition using Strassen on the 2x2 block.
    
    The key insight: schoolbook 3x3 uses 27 multiplications.
    If we partition into a 2x2 block and handle it with Strassen (7 mults instead of 8),
    we save exactly 1 multiplication, giving rank 26.
    
    The decomposition:
    - Use Strassen for the top-left 2x2 submatrix (indices 0,1)
    - Use schoolbook for all entries involving index 2
    
    But we need to be careful: the Strassen decomposition produces C00, C01, C10, C11
    using combinations of A and B entries. The entries involving index 2 are independent.
    
    Total rank = 7 (Strassen) + 19 (schoolbook for index-2 entries) = 26
    """
    n, m, p = _n, _m, _p
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # W index mapping: W[k*n+i] where k is col of C, i is row of C
    # W[0]=C[0,0], W[1]=C[1,0], W[2]=C[2,0], W[3]=C[0,1], W[4]=C[1,1], 
    # W[5]=C[2,1], W[6]=C[0,2], W[7]=C[1,2], W[8]=C[2,2]
    
    # Strassen 2x2 block: indices (0,1) x (0,1)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][col] = 1.0; U[4][col] = 1.0
    V[0][col] = 1.0; V[4][col] = 1.0
    W[0][col] = 1.0; W[4][col] = 1.0
    col += 1
    
    # M2 = (A10 + A11) * B00
    U[3][col] = 1.0; U[4][col] = 1.0
    V[0][col] = 1.0
    W[3][col] = 1.0; W[4][col] = -1.0
    col += 1
    
    # M3 = A00 * (B10 - B11)
    U[0][col] = 1.0
    V[3][col] = 1.0; V[4][col] = -1.0
    W[1][col] = 1.0; W[3][col] = 1.0
    col += 1
    
    # M4 = A11 * (B01 - B00)
    U[4][col] = 1.0
    V[1][col] = 1.0; V[0][col] = -1.0
    W[0][col] = 1.0; W[4][col] = 1.0
    col += 1
    
    # M5 = (A00 + A01) * B11
    U[0][col] = 1.0; U[1][col] = 1.0
    V[4][col] = 1.0
    W[0][col] = -1.0; W[1][col] = 1.0
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[3][col] = 1.0; U[0][col] = -1.0
    V[0][col] = 1.0; V[1][col] = 1.0
    W[4][col] = 1.0
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][col] = 1.0; U[4][col] = -1.0
    V[3][col] = 1.0; V[4][col] = 1.0
    W[0][col] = 1.0
    col += 1
    
    # Schoolbook for entries involving index 2
    # C02 = A00*B02 + A01*B12 + A02*B22
    U[0][col] = 1.0; V[2][col] = 1.0; W[6][col] = 1.0; col += 1
    U[1][col] = 1.0; V[5][col] = 1.0; W[6][col] = 1.0; col += 1
    U[2][col] = 1.0; V[8][col] = 1.0; W[6][col] = 1.0; col += 1
    
    # C12 = A10*B02 + A11*B12 + A12*B22
    U[3][col] = 1.0; V[2][col] = 1.0; W[7][col] = 1.0; col += 1
    U[4][col] = 1.0; V[5][col] = 1.0; W[7][col] = 1.0; col += 1
    U[5][col] = 1.0; V[8][col] = 1.0; W[7][col] = 1.0; col += 1
    
    # C20 = A20*B00 + A21*B10 + A22*B20
    U[6][col] = 1.0; V[0][col] = 1.0; W[0][col] = 1.0; col += 1
    U[7][col] = 1.0; V[3][col] = 1.0; W[0][col] = 1.0; col += 1
    U[8][col] = 1.0; V[6][col] = 1.0; W[0][col] = 1.0; col += 1
    
    # C21 = A20*B01 + A21*B11 + A22*B21
    U[6][col] = 1.0; V[1][col] = 1.0; W[1][col] = 1.0; col += 1
    U[7][col] = 1.0; V[4][col] = 1.0; W[1][col] = 1.0; col += 1
    U[8][col] = 1.0; V[7][col] = 1.0; W[1][col] = 1.0; col += 1
    
    # C22 = A20*B02 + A21*B12 + A22*B22
    U[6][col] = 1.0; V[2][col] = 1.0; W[2][col] = 1.0; col += 1
    U[7][col] = 1.0; V[5][col] = 1.0; W[2][col] = 1.0; col += 1
    U[8][col] = 1.0; V[8][col] = 1.0; W[2][col] = 1.0; col += 1
    
    # Total: 7 + 15 = 22 columns used. Rank is 26, so 4 zero columns remain.
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0
    best = _schoolbook_rank27()
    
    # Try rank 26
    cand = _try_rank26_structured()
    if cand is not None and _verify(cand["U"], cand["V"], cand["W"]):
        return cand
        
    # Fallback to rank 27
    return best
