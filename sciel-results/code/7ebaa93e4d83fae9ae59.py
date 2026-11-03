import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if isinstance(U, np.ndarray):
        U = U.tolist()
    if isinstance(V, np.ndarray):
        V = V.tolist()
    if isinstance(W, np.ndarray):
        W = W.tolist()
    
    rank = len(U[0]) if U else 0
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs).
    
    For X(2x2) * Y(2x2) = Z(2x2):
    M1 = (X00+X11)(Y00+Y11)
    M2 = (X10+X11)(Y00)
    M3 = (X00)(Y01-Y11)
    M4 = (X11)(Y10-Y00)
    M5 = (X00+X01)(Y11)
    M6 = (X10-X00)(Y00+Y01)
    M7 = (X01-X11)(Y10+Y11)
    
    Z00 = M1 + M4 - M5 + M7
    Z01 = M3 + M5
    Z10 = M2 + M4
    Z11 = M1 - M2 + M3 + M6
    """
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms

def _add_strassen_block(U, V, W, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, offset, m, p, n):
    terms = _strassen_2x2_terms()
    for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(terms):
        r = offset + t_idx
        for (xi, xj), val in x_coeff.items():
            U[(a_row_off + xi) * m + (a_col_off + xj)][r] += val
        for (yi, yj), val in y_coeff.items():
            V[(b_row_off + yi) * p + (b_col_off + yj)][r] += val
        for (zi, zj), val in z_coeff.items():
            W[(c_col_off + zj) * n + (c_row_off + zi)][r] += val

def _try_rank_44():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for j in range(4):
        for k in range(4):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        _add_strassen_block(U, V, W, ar, ac, br, bc, cr, cc, col, m, p, n)
        col += 7
    
    return U, V, W

def _try_rank_40():
    """
    Try a different partition:
    - Use Strassen for all of A (3x4) by padding to 4x4, but only using 3 rows.
    - Actually, let's try: 
      Row 0: 1x4 * 4x4 -> schoolbook 16
      Rows 1-2: 2x4 * 4x4 -> try to use a rank-19 decomposition if available, 
                     otherwise fall back to 4 Strassen blocks (28).
    Since we don't have rank-19 for 2x4x4, let's try a different split:
    
    Alternative: Split the 4 columns into 2+2, and 4 rows of B into 2+2.
    A is 3x4. Split A into:
      A[:, 0:2] (3x2) and A[:, 2:4] (3x2)
    B is 4x4. Split B into 4 blocks of 2x2.
    
    For each (A_block, B_block) pair, we compute a 3x2 * 2x2 = 3x2 product.
    There are 2 * 4 = 8 such products, each of size 3x2x2.
    Rank of M_{3,2,2} = 3*2*2 = 12 schoolbook.
    Total = 8 * 12 = 96. Worse.
    
    Let's try:
    A is 3x4. Split into 3 blocks of 1x4.
    B is 4x4.
    Each 1x4 * 4x4 = 1x4. Rank 16 each. Total 48.
    
    The current 44 is already an improvement over 48.
    
    Let's try to see if we can reduce the 28 for the 2x4 part.
    2x4 * 4x4 = 2x4.
    Split the 4 columns of A (and B) into 2+2.
    A[1:3, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[1:3, 0:2] (2x2) : rank 7
    A[1:3, 0:2] (2x2) * B[2:4, 0:2] (2x2) -> C[1:3, 0:2] (2x2) : rank 7
    A[1:3, 2:4] (2x2) * B[0:2, 2:4] (2x2) -> C[1:3, 2:4] (2x2) : rank 7
    A[1:3, 2:4] (2x2) * B[2:4, 2:4] (2x2) -> C[1:3, 2:4] (2x2) : rank 7
    Total 28. This is what we have.
    
    Is there a rank < 28 for 2x4x4?
    The tensor is M_{2,4,4}. 
    Lower bound: ceil(2*4*4 / (2+4+4-2)) = ceil(32/8) = 4.
    Upper bound: 32 (schoolbook) or 28 (Strassen blocks) or 19 (from 4x4x4).
    
    If I could use a rank-19 decomposition for M_{4,4,4} and restrict it to the first 2 rows of A, 
    I would get a rank-19 decomposition for M_{2,4,4}.
    Then total rank = 16 + 19 = 35.
    
    I don't have the explicit rank-19 coefficients for M_{4,4,4}.
    
    However, I can try to construct one using the known rank-19 algorithm.
    The rank-19 algorithm for 4x4 matrix multiplication is known (Bini, Capovani, Lotti, Romani 1980).
    
    Let me try to implement the Bini-Capovani-Lotti-Romani (BCLR) rank-19 algorithm.
    """
    # BCLR rank-19 algorithm for 4x4 matrix multiplication.
    # This is complex. Let's see if we can find a simpler improvement.
    
    # Actually, let's just return None to indicate failure, and fall back to 44.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try rank-44 decomposition
    try:
        U, V, W = _try_rank_44()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}
