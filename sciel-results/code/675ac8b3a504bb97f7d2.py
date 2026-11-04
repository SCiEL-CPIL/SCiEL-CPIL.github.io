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
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_3x3():
    """Return the rank-27 schoolbook decomposition for 3x3x3 matrix multiplication.
    U: (9, 27), V: (9, 27), W: (9, 27)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 2x2 blocks of 3x3 matrices.
    The 6x6 matrix is divided into a 2x2 grid of 3x3 blocks.
    Matrix multiplication C = A*B where A, B, C are 6x6.
    In block form: C_block[i,k] = sum_j A_block[i,j] * B_block[j,k]
    where each block is 3x3.
    
    We use Strassen (rank 7) for the 2x2 block multiplication.
    Each of the 7 Strassen products involves multiplying 3x3 matrices.
    We use Schoolbook (rank 27) for the 3x3 matrix multiplication.
    Total rank = 7 * 27 = 189.
    
    U: (36, 189), V: (36, 189), W: (36, 189)
    U row index: (block_i, local_i, block_j, local_j) -> (block_i*3+local_i)*6 + (block_j*3+local_j)
    i = block_i * 3 + local_i, j = block_j * 3 + local_j
    U row = i*6 + j = (block_i*3+local_i)*6 + (block_j*3+local_j)
    
    V[j*p+k, c]: j = block_j*3+local_j, k = block_k*3+local_k
    V row = j*6 + k = (block_j*3+local_j)*6 + (block_k*3+local_k)
    
    W[k*n+i, c]: k = block_k*3+local_k, i = block_i*3+local_i
    W row = k*6 + i = (block_k*3+local_k)*6 + (block_i*3+local_i)
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Strassen 2x2x2 decomposition
    sU, sV, sW = _strassen_2x2()
    
    # Schoolbook 3x3x3 decomposition
    bU, bV, bW = _schoolbook_3x3()
    
    # For each Strassen term c in {0..6}, and each Schoolbook term d in {0..26}
    col = 0
    for c in range(7):
        for d in range(27):
            # U: for each block_i, local_i, block_j, local_j
            # U row = (block_i*3+li)*6 + (block_j*3+lj)
            for bi in range(2):
                for li in range(3):
                    for bj in range(2):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            # sU index: bi*2+bj
                            # bU index: li*3+lj
                            val = sU[bi*2+bj][c] * bU[li*3+lj][d]
                            if val != 0:
                                U[u_row][col] = val
            
            # V: for each block_j, local_j, block_k, local_k
            # V row = (block_j*3+lj)*6 + (block_k*3+lk)
            for bj in range(2):
                for lj in range(3):
                    for bk in range(2):
                        for lk in range(3):
                            v_row = (bj*3+lj)*6 + (bk*3+lk)
                            # sV index: bj*2+bk
                            # bV index: lj*3+lk
                            val = sV[bj*2+bk][c] * bV[lj*3+lk][d]
                            if val != 0:
                                V[v_row][col] = val
            
            # W: for each block_k, local_k, block_i, local_i
            # W row = (block_k*3+lk)*6 + (block_i*3+li)
            for bk in range(2):
                for lk in range(3):
                    for bi in range(2):
                        for li in range(3):
                            w_row = (bk*3+lk)*6 + (bi*3+li)
                            # sW index: bk*2+bi
                            # bW index: lk*3+li
                            val = sW[bk*2+bi][c] * bW[lk*3+li][d]
                            if val != 0:
                                W[w_row][col] = val
            
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
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 189)
    best = _block_decomposition_6x6x6()
    best_rank = 189
    
    # Verify it
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback: schoolbook
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
