"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _laderman_3x3():
    """Return the rank-23 Laderman decomposition for 3x3x3 matrix multiplication."""
    # Indices:
    # A, B, C are 3x3 matrices flattened to 9 vectors.
    # U, V, W are 9x23 matrices.
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # k=0
    U[0][0] = 1; U[4][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[4][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[8][0] = 1
    
    # k=1
    U[3][1] = 1; U[4][1] = 1
    V[3][1] = 1; V[4][1] = 1
    W[4][1] = 1
    
    # k=2
    U[6][2] = 1; U[7][2] = 1
    V[6][2] = 1; V[7][2] = 1
    W[8][2] = 1
    
    # k=3
    U[1][3] = 1; U[4][3] = 1
    V[1][3] = 1; V[4][3] = 1
    W[3][3] = 1
    
    # k=4
    U[2][4] = 1; U[5][4] = 1
    V[2][4] = 1; V[5][4] = 1
    W[6][4] = 1
    
    # k=5
    U[5][5] = 1; U[8][5] = 1
    V[5][5] = 1; V[8][5] = 1
    W[7][5] = 1
    
    # k=6
    U[7][6] = 1; U[8][6] = 1
    V[7][6] = 1; V[8][6] = 1
    W[5][6] = 1
    
    # k=7
    U[0][7] = 1; U[8][7] = 1
    V[8][7] = 1; V[0][7] = -1
    W[2][7] = 1; W[6][7] = 1
    
    # k=8
    U[3][8] = 1; U[8][8] = 1
    V[8][8] = 1; V[3][8] = -1
    W[5][8] = 1; W[7][8] = 1
    
    # k=9
    U[6][9] = 1; U[4][9] = 1
    V[4][9] = 1; V[6][9] = -1
    W[8][9] = 1; W[7][9] = -1
    
    # k=10
    U[1][10] = 1; U[8][10] = 1
    V[8][10] = 1; V[1][10] = -1
    W[2][10] = 1; W[6][10] = -1
    
    # k=11
    U[4][11] = 1; U[8][11] = 1
    V[8][11] = 1; V[4][11] = -1
    W[5][11] = 1; W[7][11] = -1
    
    # k=12
    U[2][12] = 1; U[4][12] = 1
    V[4][12] = 1; V[2][12] = -1
    W[3][12] = 1; W[6][12] = -1
    
    # k=13
    U[5][13] = 1; U[2][13] = 1
    V[2][13] = 1; V[5][13] = -1
    W[6][13] = 1; W[7][13] = -1
    
    # k=14
    U[7][14] = 1; U[1][14] = 1
    V[1][14] = 1; V[7][14] = -1
    W[3][14] = 1; W[5][14] = -1
    
    # k=15
    U[3][15] = 1; U[1][15] = 1
    V[1][15] = 1; V[3][15] = -1
    W[3][15] = 1; W[0][15] = -1
    
    # k=16
    U[6][16] = 1; U[2][16] = 1
    V[2][16] = 1; V[6][16] = -1
    W[6][16] = 1; W[2][16] = -1
    
    # k=17
    U[3][17] = 1; U[1][17] = 1
    V[1][17] = 1; V[3][17] = 1
    W[0][17] = 1; W[3][17] = -1
    
    # k=18
    U[6][18] = 1; U[2][18] = 1
    V[2][18] = 1; V[6][18] = 1
    W[2][18] = 1; W[6][18] = -1
    
    # k=19
    U[3][19] = 1; U[1][19] = 1
    V[1][19] = 1; V[3][19] = -1
    W[0][19] = -1; W[3][19] = 1
    
    # k=20
    U[6][20] = 1; U[2][20] = 1
    V[2][20] = 1; V[6][20] = -1
    W[2][20] = -1; W[6][20] = 1
    
    # k=21
    U[3][21] = 1; U[1][21] = 1
    V[1][21] = 1; V[3][21] = 1
    W[0][21] = -1; W[3][21] = 1
    
    # k=22
    U[6][22] = 1; U[2][22] = 1
    V[2][22] = 1; V[6][22] = 1
    W[2][22] = -1; W[6][22] = 1
    
    return U, V, W

def _block_decomposition_6x6x6_laderman():
    """
    Decompose 6x6x6 using 3x3 blocks with Laderman rank-23.
    Total rank = 8 * 23 = 184.
    """
    n, m, p = 6, 6, 6
    rank = 184
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _laderman_3x3()
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c in range(23):
                    for li in range(3):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            U[u_row][col] = sU[li*3+lj][c]
                    
                    for lj in range(3):
                        for lk in range(3):
                            v_row = (bj*3+lj)*6 + (bk*3+lk)
                            V[v_row][col] = sV[lj*3+lk][c]
                    
                    for lk in range(3):
                        for li in range(3):
                            w_row = (bk*3+lk)*6 + (bi*3+li)
                            W[w_row][col] = sW[lk*3+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 using 3x3 blocks of 2x2 matrices.
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # Try Laderman 3x3 blocks (Rank 184)
    best = _block_decomposition_6x6x6_laderman()
    
    U_arr = np.array(best["U"], dtype=np.float64)
    V_arr = np.array(best["V"], dtype=np.float64)
    W_arr = np.array(best["W"], dtype=np.float64)
    
    valid = True
    # Verify all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U_arr[0])):
                    s += U_arr[i*m+j][c] * V_arr[j*p+k][c] * W_arr[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return best
    
    # Fallback to Strassen 2x2 blocks (Rank 189)
    best = _block_decomposition_6x6x6()
    U_arr = np.array(best["U"], dtype=np.float64)
    V_arr = np.array(best["V"], dtype=np.float64)
    W_arr = np.array(best["W"], dtype=np.float64)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U_arr[0])):
                    s += U_arr[i*m+j][c] * V_arr[j*p+k][c] * W_arr[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return best
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
