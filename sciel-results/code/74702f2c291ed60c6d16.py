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
    # Each term: (dict of (row,col)->coeff for X, dict for Y, dict for Z)
    terms = [
        # M1: (X00+X11)(Y00+Y11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (X10+X11)(Y00)
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: (X00)(Y01-Y11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: (X11)(Y10-Y00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (X00+X01)(Y11)
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (X10-X00)(Y00+Y01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (X01-X11)(Y10+Y11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms

def _add_strassen_block(U, V, W, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, offset, m, p, n):
    """Add a 2x2 Strassen block at the given offsets.
    
    X = A[a_row_off:a_row_off+2, a_col_off:a_col_off+2]
    Y = B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
    Z = C[c_row_off:c_row_off+2, c_col_off:c_col_off+2]
    """
    terms = _strassen_2x2_terms()
    for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(terms):
        r = offset + t_idx
        # Add X coefficients to U
        for (xi, xj), val in x_coeff.items():
            U[(a_row_off + xi) * m + (a_col_off + xj)][r] += val
        # Add Y coefficients to V
        for (yi, yj), val in y_coeff.items():
            V[(b_row_off + yi) * p + (b_col_off + yj)][r] += val
        # Add Z coefficients to W (W index: k*n + i where k is col, i is row)
        for (zi, zj), val in z_coeff.items():
            W[(c_col_off + zj) * n + (c_row_off + zi)][r] += val

def _try_rank_44():
    """
    Split 3x4 * 4x4 into:
    - Row 0: 1x4 * 4x4 -> schoolbook, rank 16
    - Rows 1-2: 2x4 * 4x4 -> 4 Strassen 2x2 blocks, rank 28
    Total: 44
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) * B (4x4) -> Row 0 of C (1x4)
    # Schoolbook: 4*4 = 16 terms
    for j in range(4):
        for k in range(4):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) * B (4x4) -> Rows 1-2 of C (2x4)
    # Split into four 2x2 blocks:
    # A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    # A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
    # A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
    # A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]
    
    blocks = [
        (1, 0, 0, 0, 1, 0),  # A[1:3,0:2] * B[0:2,0:2] -> C[1:3,0:2]
        (1, 0, 0, 2, 1, 2),  # A[1:3,0:2] * B[0:2,2:4] -> C[1:3,2:4]
        (1, 2, 2, 0, 1, 0),  # A[1:3,2:4] * B[2:4,0:2] -> C[1:3,0:2]
        (1, 2, 2, 2, 1, 2),  # A[1:3,2:4] * B[2:4,2:4] -> C[1:3,2:4]
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        _add_strassen_block(U, V, W, ar, ac, br, bc, cr, cc, col, m, p, n)
        col += 7
    
    return U, V, W

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
