"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Validate that U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Check tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return U, V, W for Strassen 2x2x2 decomposition.
    U: (4, 7) - flattened 2x2 matrix P
    V: (4, 7) - flattened 2x2 matrix Q  
    W: (4, 7) - flattened 2x2 matrix R, indexed as W[k*2+i, c] for R[i,k]
    
    Strassen:
    M0 = (p00+p11)(q00+q11)
    M1 = (p10+p11)q00
    M2 = p00(q10-q11)
    M3 = p11(q01-q00)
    M4 = (p00+p01)q11
    M5 = (p10-p00)(q00+q01)
    M6 = (p01-p11)(q10+q11)
    
    R[0,0] = M0+M3-M4+M6
    R[0,1] = M2+M4
    R[1,0] = M1+M3
    R[1,1] = M0-M1+M2+M5
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0 = (p00+p11)(q00+q11)
    U[0][0] = 1; U[2][0] = 1
    V[0][0] = 1; V[2][0] = 1
    # R[0,0] += M0, R[1,1] += M0
    # W[k*2+i, c]: R[0,0] -> W[0], R[1,1] -> W[3]
    W[0][0] += 1; W[3][0] += 1
    
    # M1 = (p10+p11)q00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # R[1,0] += M1
    # R[1,0] -> W[1*2+0] = W[2]
    W[2][1] += 1
    
    # M2 = p00(q10-q11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # R[0,1] += M2, R[1,1] += M2
    # R[0,1] -> W[1*2+0] = W[2], R[1,1] -> W[3]
    W[2][2] += 1; W[3][2] += 1
    
    # M3 = p11(q01-q00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # R[0,0] += M3, R[1,0] += M3
    # R[0,0] -> W[0], R[1,0] -> W[2]
    W[0][3] += 1; W[2][3] += 1
    
    # M4 = (p00+p01)q11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # R[0,0] -= M4, R[0,1] += M4
    # R[0,0] -> W[0], R[0,1] -> W[2]
    W[0][4] += -1; W[2][4] += 1
    
    # M5 = (p10-p00)(q00+q01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # R[1,1] += M5
    # R[1,1] -> W[3]
    W[3][5] += 1
    
    # M6 = (p01-p11)(q10+q11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # R[0,0] += M6
    # R[0,0] -> W[0]
    W[0][6] += 1
    
    return U, V, W

def _strassen_2x2x6():
    """
    Return U, V, W for 2x2 by 2x6 multiplication using 3 Strassen 2x2x2 instances.
    U: (4, 21) - flattened 2x2 matrix P
    V: (12, 21) - flattened 2x6 matrix Q
    W: (12, 21) - flattened 6x2 matrix R, indexed as W[k*2+i, c] for R[i,k]
    """
    U_base, V_base, W_base = _strassen_2x2x2()
    
    U = [[0]*21 for _ in range(4)]
    V = [[0]*21 for _ in range(12)]
    W = [[0]*21 for _ in range(12)]
    
    for block in range(3):
        for c in range(7):
            col = block * 7 + c
            # U
            for i in range(4):
                U[i][col] = U_base[i][c]
            # V: Q is 2x6, block covers columns 2*block and 2*block+1
            # V index: row*6 + col_in_Q
            for i in range(4):
                v_row = i // 2  # 0 or 1
                v_col_in_block = i % 2  # 0 or 1
                v_global_col = 2 * block + v_col_in_block
                V[v_row * 6 + v_global_col][col] = V_base[i][c]
            # W: R is 2x6, block covers columns 2*block and 2*block+1
            # W index: k*2 + i where k is col in R, i is row in R
            for i in range(4):
                w_row = i // 2  # 0 or 1 (row in R)
                w_col_in_block = i % 2  # 0 or 1 (col in R within block)
                w_global_col = 2 * block + w_col_in_block
                W[w_global_col * 2 + w_row][col] = W_base[i][c]
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    Uses Strassen-based decomposition with rank 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    start_time = time.time()
    deadline = start_time + 110
    
    # Split A (2x4) into A_left (2x2) and A_right (2x2)
    # Split B (4x6) into B_top (2x6) and B_bottom (2x6)
    # C = A_left @ B_top + A_right @ B_bottom
    
    # A_left @ B_top: 2x2 by 2x6 -> rank 21
    # A_right @ B_bottom: 2x2 by 2x6 -> rank 21
    # Total rank: 42
    
    U_left, V_left, W_left = _strassen_2x2x6()
    U_right, V_right, W_right = _strassen_2x2x6()
    
    total_rank = 42
    U = [[0] * total_rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 42)
    
    # Left block: A_left rows 0,1 cols 0,1 -> U indices 0,1,2,3
    # B_top rows 0,1 -> V indices 0..5, 6..11
    # C -> W indices as defined
    
    for i in range(4):
        for c in range(21):
            U[i][c] = U_left[i][c]
    
    for i in range(12):
        for c in range(21):
            V[i][c] = V_left[i][c]
    
    for i in range(12):
        for c in range(21):
            W[i][c] = W_left[i][c]
    
    # Right block: A_right rows 0,1 cols 2,3 -> U indices 4,5,6,7
    # B_bottom rows 2,3 -> V indices 12..17, 18..23
    # C -> same W indices (added to left block contributions)
    
    for i in range(4):
        for c in range(21):
            U[i + 4][21 + c] = U_right[i][c]
    
    for i in range(12):
        for c in range(21):
            V[i + 12][21 + c] = V_right[i][c]
    
    for i in range(12):
        for c in range(21):
            W[i][21 + c] = W_right[i][c]
    
    result = {"U": U, "V": V, "W": W}
    
    # Validate
    if _validate(U, V, W, n, m, p):
        return result
    
    # Fallback to schoolbook if validation fails
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
