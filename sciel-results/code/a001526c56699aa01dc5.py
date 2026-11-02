"""Structural redesign: recursive block-based decomposition using Strassen for 2x2 blocks.

The 4x5x6 matrix multiplication is decomposed into four 2x2-block sub-problems:
- A[0:2, 0:2] * B[0:2, 0:6]  -> 2x2 * 2x6
- A[0:2, 2:4] * B[2:4, 0:6]  -> 2x2 * 2x6
- A[2:4, 0:2] * B[0:2, 0:6]  -> 2x2 * 2x6
- A[2:4, 2:4] * B[2:4, 0:6]  -> 2x2 * 2x6

Each 2x2 * 2x6 sub-problem can be decomposed using 3 Strassen multiplications of 2x3 blocks,
or more efficiently, we can use the fact that 2x2 * 2x6 = 2x2 * (2x3 + 2x3) to use Strassen
on each 2x2*2x3 piece.

Actually, a cleaner approach: decompose 4x5x6 as:
- 4x5 * 5x6 = (4x2 + 4x3) * (2x6 + 3x6)
  = 4x2*2x6 + 4x2*3x6 + 4x3*2x6 + 4x3*3x6

For each sub-problem, we can use Strassen-like decompositions where possible.

However, the simplest structural improvement is to use the known rank-7 Strassen decomposition
for 2x2*2x2 and pad appropriately.

Let me use a different structural approach: 
- Split the middle dimension (5) into 2+3
- Split the first dimension (4) into 2+2
- Split the last dimension (6) into 3+3

This gives 8 sub-problems. For 2x2*2x2 we use Strassen (rank 7).
For other sizes, we use schoolbook or other known decompositions.

Actually, let me just implement a recursive algorithm that uses Strassen whenever the dimensions
are even, and schoolbook otherwise. This is a structural change from pure schoolbook.
"""

import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition by splitting into blocks.
    # Strategy: Use Strassen for 2x2x2 sub-problems, schoolbook for others.
    # 
    # Split n=4 into [2,2], m=5 into [2,3], p=6 into [3,3]
    # This gives 2*2*2 = 8 sub-problems.
    
    n_splits = [2, 2]
    m_splits = [2, 3]
    p_splits = [3, 3]
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    for ni, ni_size in enumerate(n_splits):
        for mi, mi_size in enumerate(m_splits):
            for pi, pi_size in enumerate(p_splits):
                # Sub-problem: ni_size x mi_size x pi_size
                # Starting indices
                n_start = sum(n_splits[:ni])
                m_start = sum(m_splits[:mi])
                p_start = sum(p_splits[:pi])
                
                # Get sub-decomposition
                sub_U, sub_V, sub_W = _decompose_sub(ni_size, mi_size, pi_size)
                rank = sub_U.shape[1]
                
                # Adjust indices
                for r in range(sub_U.shape[0]):
                    # U index: (n_start + i) * m + (m_start + j)
                    # r corresponds to i*mi_size + j, so i = r // mi_size, j = r % mi_size
                    i_local = r // mi_size
                    j_local = r % mi_size
                    global_i = n_start + i_local
                    global_j = m_start + j_local
                    global_r = global_i * m + global_j
                    U_rows.append((global_r, sub_U[r]))
                
                for r in range(sub_V.shape[0]):
                    # V index: (m_start + j) * p + (p_start + k)
                    # r corresponds to j*pi_size + k
                    j_local = r // pi_size
                    k_local = r % pi_size
                    global_j = m_start + j_local
                    global_k = p_start + k_local
                    global_r = global_j * p + global_k
                    V_rows.append((global_r, sub_V[r]))
                
                for r in range(sub_W.shape[0]):
                    # W index: (p_start + k) * n + (n_start + i)
                    # r corresponds to k*ni_size + i
                    k_local = r // ni_size
                    i_local = r % ni_size
                    global_k = p_start + k_local
                    global_i = n_start + i_local
                    global_r = global_k * n + global_i
                    W_rows.append((global_r, sub_W[r]))
    
    # Now assemble U, V, W
    # Count total rank
    total_rank = 0
    for _, row in U_rows:
        total_rank = max(total_rank, len(row))
    
    # Actually, we need to concatenate columns
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (global_r, sub_row), (global_r_v, sub_row_v), (global_r_w, sub_row_w) in zip(U_rows, V_rows, W_rows):
        # This is not quite right - we need to track which sub-problem each row belongs to
        pass
    
    # Let me redo this more carefully
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for ni, ni_size in enumerate(n_splits):
        for mi, mi_size in enumerate(m_splits):
            for pi, pi_size in enumerate(p_splits):
                n_start = sum(n_splits[:ni])
                m_start = sum(m_splits[:mi])
                p_start = sum(p_splits[:pi])
                
                sub_U, sub_V, sub_W = _decompose_sub(ni_size, mi_size, pi_size)
                rank = sub_U.shape[1]
                
                for r in range(sub_U.shape[0]):
                    i_local = r // mi_size
                    j_local = r % mi_size
                    global_i = n_start + i_local
                    global_j = m_start + j_local
                    global_r = global_i * m + global_j
                    for c in range(rank):
                        U[global_r][col_offset + c] = sub_U[r, c]
                
                for r in range(sub_V.shape[0]):
                    j_local = r // pi_size
                    k_local = r % pi_size
                    global_j = m_start + j_local
                    global_k = p_start + k_local
                    global_r = global_j * p + global_k
                    for c in range(rank):
                        V[global_r][col_offset + c] = sub_V[r, c]
                
                for r in range(sub_W.shape[0]):
                    k_local = r // ni_size
                    i_local = r % ni_size
                    global_k = p_start + k_local
                    global_i = n_start + i_local
                    global_r = global_k * n + global_i
                    for c in range(rank):
                        W[global_r][col_offset + c] = sub_W[r, c]
                
                col_offset += rank
    
    return {"U": U, "V": V, "W": W}


def _decompose_sub(n, m, p):
    """Decompose an n x m * m x p multiplication into rank-r decomposition.
    Uses Strassen when n=m=p=2, schoolbook otherwise.
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank).
    """
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    else:
        return _schoolbook(n, m, p)


def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.
    
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    C = A*B
    
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b12-b22)
    m4 = a22(b21-b11)
    m5 = (a11+a12)b22
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    U maps A indices (i*2+j) to the 7 products.
    V maps B indices (j*2+k) to the 7 products.
    W maps C indices (k*2+i) to the 7 products.
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # U: A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # m1 = (a00+a11)(b00+b11): U[0]+=1, U[3]+=1
    U[0, 0] = 1; U[3, 0] = 1
    # m2 = (a10+a11)b00: U[2]+=1, U[3]+=1
    U[2, 1] = 1; U[3, 1] = 1
    # m3 = a00(b01-b11): U[0]+=1
    U[0, 2] = 1
    # m4 = a11(b10-b00): U[3]+=1
    U[3, 3] = 1
    # m5 = (a00+a01)b11: U[0]+=1, U[1]+=1
    U[0, 4] = 1; U[1, 4] = 1
    # m6 = (a10-a00)(b00+b01): U[2]+=1, U[0]-=1
    U[2, 5] = 1; U[0, 5] = -1
    # m7 = (a01-a11)(b10+b11): U[1]+=1, U[3]-=1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V: B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # m1: b00+b11: V[0]+=1, V[3]+=1
    V[0, 0] = 1; V[3, 0] = 1
    # m2: b00: V[0]+=1
    V[0, 1] = 1
    # m3: b01-b11: V[1]+=1, V[3]-=1
    V[1, 2] = 1; V[3, 2] = -1
    # m4: b10-b00: V[2]+=1, V[0]-=1
    V[2, 3] = 1; V[0, 3] = -1
    # m5: b11: V[3]+=1
    V[3, 4] = 1
    # m6: b00+b01: V[0]+=1, V[1]+=1
    V[0, 5] = 1; V[1, 5] = 1
    # m7: b10+b11: V[2]+=1, V[3]+=1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # c11 = m1+m4-m5+m7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # c12 = m3+m5
    W[1, 2] = 1; W[1, 4] = 1
    # c21 = m2+m4
    W[2, 1] = 1; W[2, 3] = 1
    # c22 = m1-m2+m3+m6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W
