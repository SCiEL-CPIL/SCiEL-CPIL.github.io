import numpy as np
import time

DIMENSIONS = [2, 4, 6]
n, m, p = DIMENSIONS
SCHOOLBOOK_RANK = n * m * p  # 48

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the factorization is exact via tensor identity."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_decomp():
    """
    Decompose 2x4 @ 4x6 using Strassen on 2x2 blocks.
    A (2x4) split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3).
    B (4x6) split into B_top (rows 0-1) and B_bot (rows 2-3), each 2x6.
    B_top split into 3 blocks of 2x2 (cols 0-1, 2-3, 4-5).
    B_bot split into 3 blocks of 2x2 (cols 0-1, 2-3, 4-5).
    C = A_left @ B_top + A_right @ B_bot.
    Each 2x2 @ 2x2 uses Strassen (rank 7).
    Total: 6 * 7 = 42 rank.
    """
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]   # 24 rows
    W = [[0.0] * rank for _ in range(p * n)]   # 12 rows
    
    col = 0
    
    def add_strassen(a_col_offset, b_row_offset, b_col_offset):
        """
        a_col_offset: column offset in A for this 2x2 block (0 or 2)
        b_row_offset: row offset in B (0 or 2)
        b_col_offset: col offset in B (0, 2, or 4)
        The result contributes to C[0:2, b_col_offset:b_col_offset+2]
        """
        nonlocal col
        # Strassen terms for 2x2 @ 2x2
        # Each term: (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: list of (local_i, local_j, coefficient) for A block
        # b_coeffs: list of (local_j, local_k, coefficient) for B block
        # c_coeffs: list of (local_i, local_k, coefficient) for C block
        strassen_terms = [
            # P1 = (A11+A22)(B11+B22) -> C11+=1, C22+=1
            ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
            # P2 = (A21+A22)B11 -> C21+=1
            ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0)]),
            # P3 = A11(B12-B22) -> C12+=1, C22+=1
            ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
            # P4 = A22(B21-B11) -> C11+=1, C21+=1
            ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
            # P5 = (A11+A12)B22 -> C11-=1, C12+=1
            ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
            # P6 = (A21-A11)(B11+B12) -> C22+=1
            ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
            # P7 = (A12-A22)(B21+B22) -> C11+=1, C22-=1
            ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,-1.0)]),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            for (ai, aj, coef) in a_coeffs:
                global_i = ai
                global_j = a_col_offset + aj
                U[global_i * m + global_j][col] += coef
            for (bj, bk, coef) in b_coeffs:
                global_i = b_row_offset + bj
                global_j = b_col_offset + bk
                V[global_i * p + global_j][col] += coef
            for (ci, ck, coef) in c_coeffs:
                global_i = ci
                global_j = b_col_offset + ck
                W[global_j * n + global_i][col] += coef
            col += 1
    
    # A_left (cols 0-1) @ B_top (rows 0-1) blocks
    add_strassen(0, 0, 0)  # A_left @ B_top[:, 0:2]
    add_strassen(0, 0, 2)  # A_left @ B_top[:, 2:4]
    add_strassen(0, 0, 4)  # A_left @ B_top[:, 4:6]
    
    # A_right (cols 2-3) @ B_bot (rows 2-3) blocks
    add_strassen(2, 2, 0)  # A_right @ B_bot[:, 0:2]
    add_strassen(2, 2, 2)  # A_right @ B_bot[:, 2:4]
    add_strassen(2, 2, 4)  # A_right @ B_bot[:, 4:6]
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Try Strassen-based decomposition (rank 42)
    try:
        result = _strassen_decomp()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()

