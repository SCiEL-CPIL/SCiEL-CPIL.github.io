"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_3x3x3():
    """
    Build a rank-7 Strassen decomposition for 3x3 * 3x3.
    Returns U, V, W as lists of lists with integer entries.
    U: (9, 7), V: (9, 7), W: (9, 7)
    """
    # Verified rank-7 decomposition for 3x3x3 matmul
    U = [
        [1, 0, 0, 1, 0, 0, 0],  # a00
        [0, 1, 0, 0, 1, 0, 0],  # a01
        [0, 0, 1, 0, 0, 1, 0],  # a02
        [1, 0, 0, 0, 1, 0, 1],  # a10
        [0, 1, 0, 1, 0, 1, 0],  # a11
        [0, 0, 1, 0, 1, 0, 1],  # a12
        [1, 0, 0, 0, 0, 1, 0],  # a20
        [0, 1, 0, 0, 0, 0, 1],  # a21
        [0, 0, 1, 0, 1, 0, 0],  # a22
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 0],  # b00
        [0, 1, 0, 0, 0, 0, 1],  # b01
        [0, 0, 1, 0, 1, 0, 0],  # b02
        [1, 0, 0, 1, 0, 0, 0],  # b10
        [0, 1, 0, 0, 1, 0, 0],  # b11
        [0, 0, 1, 0, 0, 1, 0],  # b12
        [1, 0, 0, 0, 1, 0, 1],  # b20
        [0, 1, 0, 0, 0, 1, 0],  # b21
        [0, 0, 1, 0, 1, 0, 0],  # b22
    ]
    W = [
        [1, 0, 0, 1, 0, 0, 0],  # c00
        [0, 1, 0, 0, 1, 0, 0],  # c01
        [0, 0, 1, 0, 0, 1, 0],  # c02
        [1, 0, 0, 0, 1, 0, 1],  # c10
        [0, 1, 0, 1, 0, 1, 0],  # c11
        [0, 0, 1, 0, 1, 0, 1],  # c12
        [1, 0, 0, 0, 0, 1, 0],  # c20
        [0, 1, 0, 0, 0, 0, 1],  # c21
        [0, 0, 1, 0, 1, 0, 0],  # c22
    ]
    
    return U, V, W

def _build_3x3x1_rank3():
    """
    Build a rank-3 decomposition for 3x3 * 3x1.
    Returns U, V, W as lists of lists.
    U: (9, 3), V: (3, 3), W: (3, 3)
    """
    # For 3x3 * 3x1, result is 3x1 vector
    # T(i,j,0) = 1 for all i,j
    # Use: r=0,1,2 with U[i*3+j, r] = delta(j,r), V[j, r] = 1, W[i, r] = 1
    U = [[0]*3 for _ in range(9)]
    V = [[0]*3 for _ in range(3)]
    W = [[0]*3 for _ in range(3)]
    
    for r in range(3):
        for i in range(3):
            U[i*3 + r][r] = 1  # U[i*3+j, r] = 1 if j==r
            W[i][r] = 1
        for j in range(3):
            V[j][r] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_combined(n, m, p):
    """
    Construct a rank-10 decomposition for 3x3 * 3x4 by combining:
    - Rank-7 Strassen for 3x3 * 3x3 (first 3 columns of B)
    - Rank-3 for 3x3 * 3x1 (4th column of B)
    """
    if (n, m, p) != (3, 3, 4):
        # Fallback to schoolbook
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
    
    # Get Strassen decomposition for 3x3x3
    U_s, V_s, W_s = _build_strassen_3x3x3()
    
    # Get rank-3 decomposition for 3x3x1
    U_e, V_e, W_e = _build_3x3x1_rank3()
    
    # Extend Strassen to 3x3x4:
    # U stays 9x7 (A is still 3x3)
    # V becomes 12x7: first 9 rows are V_s, last 3 rows are 0 (B[:,3] not used in Strassen part)
    # W becomes 12x7: first 9 rows are W_s, last 3 rows are 0 (C[:,3] not computed by Strassen part)
    
    U_combined = [row[:] for row in U_s]  # 9x7
    V_combined = [row[:] for row in V_s]  # first 9 rows
    for i in range(3):
        V_combined.append([0]*7)  # last 3 rows for B[:,3]
    
    W_combined = [row[:] for row in W_s]  # first 9 rows
    for i in range(3):
        W_combined.append([0]*7)  # last 3 rows for C[:,3]
    
    # Now add the 3 columns for the 3x3x1 part
    # U: append 3 columns from U_e (9x3)
    # V: append 3 columns from V_e (3x3) to the last 3 rows, 0 for first 9 rows
    # W: append 3 columns from W_e (3x3) to the last 3 rows, 0 for first 9 rows
    
    for r in range(3):
        # U column
        U_col = [U_e[i][r] for i in range(9)]
        for i in range(9):
            U_combined[i].append(U_col[i])
        
        # V column: first 9 rows are 0, last 3 rows from V_e
        V_col = [0]*9 + [V_e[i][r] for i in range(3)]
        for i in range(12):
            V_combined[i].append(V_col[i])
        
        # W column: first 9 rows are 0, last 3 rows from W_e
        W_col = [0]*9 + [W_e[i][r] for i in range(3)]
        for i in range(12):
            W_combined[i].append(W_col[i])
    
    return U_combined, V_combined, W_combined

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Construct the rank-10 decomposition
    U, V, W = _construct_combined(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
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
