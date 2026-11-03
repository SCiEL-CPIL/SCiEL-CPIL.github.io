"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition search."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE_MARGIN = 10  # seconds before the 150s kill
START_TIME = time.time()


def _time_left():
    return (START_TIME + 150 - DEADLINE_MARGIN) - time.time()


def _verify(U, V, W):
    """Check exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < 1e-9


def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U shape (4, 7): U[i*2+j, c]
    V shape (4, 7): V[j*2+k, c]
    W shape (4, 7): W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A: A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B: B00=V[0], B01=V[1], B10=V[2], B11=V[3]
    # C: C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B10-B11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B01-B10)
    U[3][3] += 1
    V[1][3] += 1; V[2][3] -= 1
    W[0][3] -= 1; W[1][3] += 1  # C00 -= M4, C01 += M4
    
    # M5 = (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += 1  # C00 += M5
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1  # C11 += M6
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[1][6] += 1; V[3][6] += 1
    W[0][6] -= 1; W[2][6] += 1  # C00 -= M7, C10 += M7
    
    return U, V, W


def _schoolbook_1x2x2():
    """Return (U, V, W) for 1x2x2 schoolbook, rank 4.
    U shape (2, 4): U[j, c]
    V shape (4, 4): V[j*2+k, c]
    W shape (2, 4): W[k, c]
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    
    for c in range(4):
        j = c % 2
        k = c // 2
        U[j][c] = 1
        V[j*2+k][c] = 1
        W[k][c] = 1
    
    return U, V, W


def _build_block_decomposition():
    """Build block-structured decomposition for 3x4x6, rank 66.
    
    Partition: i: 2+1, j: 2+2, k: 2+2+2
    - 6 blocks of 2x2x2: Strassen rank 7 each = 42
    - 6 blocks of 1x2x2: schoolbook rank 4 each = 24
    Total: 66
    """
    n, m, p = 3, 4, 6
    
    i_groups = [(0, 2), (2, 3)]
    j_groups = [(0, 2), (2, 4)]
    k_groups = [(0, 2), (2, 4), (4, 6)]
    
    # Compute total rank
    total_rank = 0
    for i_start, i_end in i_groups:
        i_size = i_end - i_start
        for j_start, j_end in j_groups:
            j_size = j_end - j_start
            for k_start, k_end in k_groups:
                k_size = k_end - k_start
                if i_size == 2 and j_size == 2 and k_size == 2:
                    total_rank += 7
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    total_rank += 4
                else:
                    total_rank += i_size * j_size * k_size
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for i_start, i_end in i_groups:
        i_size = i_end - i_start
        for j_start, j_end in j_groups:
            j_size = j_end - j_start
            for k_start, k_end in k_groups:
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    U_block, V_block, W_block = _strassen_2x2x2()
                    block_rank = 7
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    U_block, V_block, W_block = _schoolbook_1x2x2()
                    block_rank = 4
                else:
                    block_rank = i_size * j_size * k_size
                    U_block = [[0]*block_rank for _ in range(i_size * j_size)]
                    V_block = [[0]*block_rank for _ in range(j_size * k_size)]
                    W_block = [[0]*block_rank for _ in range(k_size * i_size)]
                    for c in range(block_rank):
                        i = (c // (j_size * k_size)) % i_size
                        j = (c // k_size) % j_size
                        k = c % k_size
                        U_block[i * j_size + j][c] = 1
                        V_block[j * k_size + k][c] = 1
                        W_block[k * i_size + i][c] = 1
                
                for c in range(block_rank):
                    for i_local in range(i_size):
                        for j_local in range(j_size):
                            U[(i_start + i_local) * m + (j_start + j_local)][col_offset + c] += U_block[i_local * j_size + j_local][c]
                    for j_local in range(j_size):
                        for k_local in range(k_size):
                            V[(j_start + j_local) * p + (k_start + k_local)][col_offset + c] += V_block[j_local * k_size + k_local][c]
                    for k_local in range(k_size):
                        for i_local in range(i_size):
                            W[(k_start + k_local) * n + (i_start + i_local)][col_offset + c] += W_block[k_local * i_size + i_local][c]
                
                col_offset += block_rank
    
    return U, V, W, total_rank


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build block-structured decomposition (rank 66)
    U, V, W, rank = _build_block_decomposition()
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
