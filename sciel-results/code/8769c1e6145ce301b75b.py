"""Matrix multiplication 4x4x6 via Strassen-Winograd block decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    for c in [0]:
        U[0][c] = 1; U[3][c] = 1
        V[0][c] = 1; V[3][c] = 1
        W[0][c] = 1; W[3][c] = 1
    
    # M2
    for c in [1]:
        U[2][c] = 1; U[3][c] = 1
        V[0][c] = 1
        W[1][c] = 1; W[3][c] = -1
    
    # M3
    for c in [2]:
        U[0][c] = 1
        V[2][c] = 1; V[3][c] = -1
        W[1][c] = 1; W[3][c] = 1
    
    # M4
    for c in [3]:
        U[3][c] = 1
        V[1][c] = 1; V[0][c] = -1
        W[0][c] = 1; W[1][c] = 1
    
    # M5
    for c in [4]:
        U[0][c] = 1; U[1][c] = 1
        V[3][c] = 1
        W[0][c] = -1; W[1][c] = 1
    
    # M6
    for c in [5]:
        U[2][c] = 1; U[0][c] = -1
        V[0][c] = 1; V[1][c] = 1
        W[3][c] = 1
    
    # M7
    for c in [6]:
        U[1][c] = 1; U[3][c] = -1
        V[2][c] = 1; V[3][c] = 1
        W[0][c] = 1
        
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    rank = 84
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_strassen_block(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        sU, sV, sW = _strassen_2x2()
        
        i1, i2 = c_row_offset, c_row_offset + 1
        j1, j2 = b_col_offset, b_col_offset + 1
        k1, k2 = c_col_offset, c_col_offset + 1
        
        U_map = [i1*m + j1, i1*m + j2, i2*m + j1, i2*m + j2]
        V_map = [j1*p + k1, j1*p + k2, j2*p + k1, j2*p + k2]
        W_map = [k1*n + i1, k1*n + i2, k2*n + i1, k2*n + i2]
        
        for c in range(7):
            for li in range(4):
                if sU[li][c] != 0:
                    U[U_map[li]][col] += sU[li][c]
            for lv in range(4):
                if sV[lv][c] != 0:
                    V[V_map[lv]][col] += sV[lv][c]
            for lw in range(4):
                if sW[lw][c] != 0:
                    W[W_map[lw]][col] += sW[lw][c]
            col += 1
    
    # Part 1: 4x4x4
    for i_block in range(2):
        for j_block in range(2):
            for k_block in range(2):
                add_strassen_block(i_block * 2, j_block * 2, i_block * 2, k_block * 2)
                
    # Part 2: 4x4x2
    for i_block in range(2):
        for j_block in range(2):
            add_strassen_block(i_block * 2, j_block * 2, i_block * 2, 4)
            
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W, n, m, p):
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
