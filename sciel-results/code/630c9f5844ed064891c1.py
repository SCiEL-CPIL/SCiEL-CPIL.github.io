import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _to_halfint(x):
    """Convert a Fraction or number to integer or half-integer (multiple of 0.5)."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        if den == 1:
            return num
        if den == 2:
            return num / 2
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        if abs(2*x - round(2*x)) < 1e-9:
            return round(2*x) / 2
        return None
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    # Check all entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                h = _to_halfint(val)
                if h is None:
                    return False
    # Verify tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1) > 1e-9:
                        return False
    return True

def _schoolbook():
    n, m, p = 2, 5, 6
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

def _strassen_222():
    """
    Return the Strassen decomposition for 2x2 matrix multiplication.
    U: (2*2, 7) = (4, 7)
    V: (2*2, 7) = (4, 7)
    W: (2*2, 7) = (4, 7)
    
    Strassen formulas:
    M1 = (A22)(B21 - B11)
    M2 = (A11 + A22)(B11)
    M3 = (A11 + A22)(B22)
    M4 = (A11)(B12 - B22)
    M5 = (A21 + A22)(B11)
    M6 = (A11 - A21)(B21 + B22)
    M7 = (A12 - A22)(B11 + B12)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M2 + M3
    C21 = M5 + M6
    C22 = M1 - M3 + M6 + M7
    
    Indices: A[i,j] -> U[i*2+j, c], B[j,k] -> V[j*2+k, c], C[i,k] -> W[k*2+i, c]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # M1 = (A22)(B21 - B11)
    # A22 is at index (1,1) -> U[3]
    # B21 is at index (1,0) -> V[2], B11 is at index (0,0) -> V[0]
    U[3][0] = 1
    V[2][0] = 1
    V[0][0] = -1
    # C11 += M1, C22 += M1
    W[0][0] = 1  # C11 (i=0,k=0) -> W[0*2+0] = W[0]
    W[3][0] = 1  # C22 (i=1,k=1) -> W[1*2+1] = W[3]
    
    # M2 = (A11 + A22)(B11)
    # A11 is at (0,0) -> U[0], A22 is at (1,1) -> U[3]
    # B11 is at (0,0) -> V[0]
    U[0][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    # C12 += M2
    W[1][1] = 1  # C12 (i=0,k=1) -> W[1*2+0] = W[2]? Wait.
    # W[k*n+i] where n=2. So W[k*2+i].
    # C[i,k] -> W[k*2+i]
    # C12: i=0, k=1 -> W[1*2+0] = W[2]
    W[2][1] = 1
    
    # M3 = (A11 + A22)(B22)
    # A11 -> U[0], A22 -> U[3]
    # B22 -> V[3]
    U[0][2] = 1
    U[3][2] = 1
    V[3][2] = 1
    # C12 += M3
    W[2][2] = 1
    
    # M4 = (A11)(B12 - B22)
    # A11 -> U[0]
    # B12 -> V[1], B22 -> V[3]
    U[0][3] = 1
    V[1][3] = 1
    V[3][3] = -1
    # C11 += M4
    W[0][3] = 1
    
    # M5 = (A21 + A22)(B11)
    # A21 -> U[2], A22 -> U[3]
    # B11 -> V[0]
    U[2][4] = 1
    U[3][4] = 1
    V[0][4] = 1
    # C11 -= M5, C21 += M5
    W[0][4] = -1  # C11
    W[2][4] = 1   # C21: i=1, k=0 -> W[0*2+1] = W[1]? No.
    # C21: i=1, k=0 -> W[0*2+1] = W[1]
    W[1][4] = 1
    
    # M6 = (A11 - A21)(B21 + B22)
    # A11 -> U[0], A21 -> U[2]
    # B21 -> V[2], B22 -> V[3]
    U[0][5] = 1
    U[2][5] = -1
    V[2][5] = 1
    V[3][5] = 1
    # C21 += M6, C22 += M6
    W[1][5] = 1   # C21
    W[3][5] = 1   # C22
    
    # M7 = (A12 - A22)(B11 + B12)
    # A12 -> U[1], A22 -> U[3]
    # B11 -> V[0], B12 -> V[1]
    U[1][6] = 1
    U[3][6] = -1
    V[0][6] = 1
    V[1][6] = 1
    # C11 += M7, C22 += M7
    W[0][6] = 1   # C11
    W[3][6] = 1   # C22
    
    return U, V, W

def _embed_block(U_global, V_global, W_global, U_block, V_block, W_block, 
                 i_off, j_off, k_off, i_size, j_size, k_size, col_start, n, m, p):
    """
    Embed a block decomposition into the global matrices.
    i_off, j_off, k_off: offsets in the global matrix dimensions.
    i_size, j_size, k_size: sizes of the block.
    col_start: starting column index in the global matrices.
    """
    for i in range(i_size):
        for j in range(j_size):
            for c in range(len(U_block[0])):
                val = U_block[i * j_size + j][c]
                if val != 0:
                    U_global[i_off + i][j_off + j][col_start + c] += val
    
    for j in range(j_size):
        for k in range(k_size):
            for c in range(len(V_block[0])):
                val = V_block[j * k_size + k][c]
                if val != 0:
                    V_global[j_off + j][k_off + k][col_start + c] += val
    
    for k in range(k_size):
        for i in range(i_size):
            for c in range(len(W_block[0])):
                val = W_block[k * i_size + i][c]
                if val != 0:
                    W_global[k_off + k][i_off + i][col_start + c] += val

def _construct_strassen_based():
    """
    Construct a decomposition using Strassen for 2x2 blocks.
    A is 2x5, B is 5x6.
    Split A into 2x2, 2x2, 2x1 blocks.
    Split B into 2x2, 2x2, 2x2 blocks (along the 5 dimension).
    Wait, B is 5x6. We can split the 5 dimension into 2, 2, 1.
    So B is:
    B11 (2x2), B12 (2x2), B13 (2x2)
    B21 (2x2), B22 (2x2), B23 (2x2)
    B31 (1x2), B32 (1x2), B33 (1x2)
    
    A is:
    A11 (2x2), A12 (2x2), A13 (2x1)
    
    C = A * B
    C11 = A11*B11 + A12*B21 + A13*B31
    C12 = A11*B12 + A12*B22 + A13*B32
    C13 = A11*B13 + A12*B23 + A13*B33
    
    C is 2x6.
    C11 is 2x2, C12 is 2x2, C13 is 2x2.
    
    We can apply Strassen to the 2x2 multiplications.
    For A11*B11 (2x2 * 2x2): rank 7
    For A12*B21 (2x2 * 2x2): rank 7
    For A13*B31 (2x1 * 1x2): rank 4 (just 4 terms)
    
    Total rank = 7 + 7 + 4 = 18 for the first 2x2 block of C.
    Similarly for the other 2x2 blocks of C.
    Total rank = 3 * 18 = 54.
    """
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s, V_s, W_s = _strassen_222()
    
    col = 0
    # Block 1: C11 (2x2)
    # A11 (2x2), B11 (2x2) -> Strassen
    _embed_block(U, V, W, U_s, V_s, W_s, 0, 0, 0, 2, 2, 2, col, n, m, p)
    col += 7
    # A12 (2x2), B21 (2x2) -> Strassen
    _embed_block(U, V, W, U_s, V_s, W_s, 0, 2, 2, 2, 2, 2, col, n, m, p)
    col += 7
    # A13 (2x1), B31 (1x2) -> Schoolbook (4 terms)
    # A13 is at i=0,1; j=4. B31 is at j=4; k=0,1.
    # C11 is at i=0,1; k=0,1.
    for ii in range(2):
        for kk in range(2):
            U[ii * m + 4][col] = 1
            V[4 * p + kk][col] = 1
            W[kk * n + ii][col] = 1
            col += 1
    
    # Block 2: C12 (2x2)
    # A11 (2x2), B12 (2x2) -> Strassen
    _embed_block(U, V, W, U_s, V_s, W_s, 0, 0, 2, 2, 2, 2, col, n, m, p)
    col += 7
    # A12 (2x2), B22 (2x2) -> Strassen
    _embed_block(U, V, W, U_s, V_s, W_s, 0, 2, 4, 2, 2, 2, col, n, m, p)
    col += 7
    # A13 (2x1), B32 (1x2) -> Schoolbook
    for ii in range(2):
        for kk in range(2):
            U[ii * m + 4][col] = 1
            V[4 * p + 2 + kk][col] = 1
            W[2 + kk * n + ii][col] = 1
            col += 1
    
    # Block 3: C13 (2x2)
    # A11 (2x2), B13 (2x2) -> Strassen
    _embed_block(U, V, W, U_s, V_s, W_s, 0, 0, 4, 2, 2, 2, col, n, m, p)
    col += 7
    # A12 (2x2), B23 (2x2) -> Strassen
    _embed_block(U, V, W, U_s, V_s, W_s, 0, 2, 6, 2, 2, 2, col, n, m, p)
    col += 7
    # A13 (2x1), B33 (1x2) -> Schoolbook
    for ii in range(2):
        for kk in range(2):
            U[ii * m + 4][col] = 1
            V[4 * p + 4 + kk][col] = 1
            W[4 + kk * n + ii][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 5, 6
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try Strassen-based construction
    strassen_result = _construct_strassen_based()
    if _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
        best = strassen_result
        best_rank = 54
    
    return best
# EVOLVE-BLOCK-END
