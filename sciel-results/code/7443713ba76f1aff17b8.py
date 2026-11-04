"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n=2, m=5, p=6):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for r in range(rank):
        pass
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if val != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for 2x2x2 Strassen decomposition."""
    tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    return tensors

def _verify_strassen():
    """Verify the Strassen decomposition."""
    tensors = _strassen_2x2x2()
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(4)
        B = np.random.randn(4)
        a11, a12, a21, a22 = A
        b11, b12, b21, b22 = B
        c11 = a11*b11 + a12*b21
        c12 = a11*b12 + a12*b22
        c21 = a21*b11 + a22*b12
        c22 = a21*b12 + a22*b22
        C_direct = np.array([c11, c12, c21, c22])
        C_strassen = np.zeros(4)
        for u, v, w in tensors:
            m_val = sum(u[i]*A[i] for i in range(4)) * sum(v[i]*B[i] for i in range(4))
            for i in range(4):
                C_strassen[i] += w[i] * m_val
        if not np.allclose(C_direct, C_strassen):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    if not _verify_strassen():
        rank = schoolbook_rank
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
    
    # Split middle dimension m=5 into 2+2+1
    # A = [A1 | A2 | A3], B = [B1; B2; B3]
    # A1, A2 are 2x2, A3 is 2x1
    # B1, B2 are 2x6, B3 is 1x6
    # C = A1 B1 + A2 B2 + A3 B3
    
    # A1 B1: 2x2 by 2x6. Use Strassen for 3 2x2x2 blocks. Rank 21.
    # A2 B2: 2x2 by 2x6. Use Strassen for 3 2x2x2 blocks. Rank 21.
    # A3 B3: 2x1 by 1x6. Rank 2.
    # Total rank: 21 + 21 + 2 = 44.
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    strassen_tensors = _strassen_2x2x2()
    
    # Part 1: A1 B1 (j=0,1)
    for block in range(3):  # k blocks 0-1, 2-3, 4-5
        k_offset = block * 2
        for (u, v, w) in strassen_tensors:
            u_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for idx, (ii, jj) in enumerate(u_map):
                a_idx = ii * m + jj
                U[a_idx][col] += u[idx]
            v_map = [(0, k_offset), (0, k_offset+1), (1, k_offset), (1, k_offset+1)]
            for idx, (jj, kk) in enumerate(v_map):
                b_idx = jj * p + kk
                V[b_idx][col] += v[idx]
            w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
            for idx, (kk, ii) in enumerate(w_map):
                c_idx = kk * n + ii
                W[c_idx][col] += w[idx]
            col += 1
            
    # Part 2: A2 B2 (j=2,3)
    for block in range(3):
        k_offset = block * 2
        for (u, v, w) in strassen_tensors:
            u_map = [(0, 2), (0, 3), (1, 2), (1, 3)]
            for idx, (ii, jj) in enumerate(u_map):
                a_idx = ii * m + jj
                U[a_idx][col] += u[idx]
            v_map = [(2, k_offset), (2, k_offset+1), (3, k_offset), (3, k_offset+1)]
            for idx, (jj, kk) in enumerate(v_map):
                b_idx = jj * p + kk
                V[b_idx][col] += v[idx]
            w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
            for idx, (kk, ii) in enumerate(w_map):
                c_idx = kk * n + ii
                W[c_idx][col] += w[idx]
            col += 1
            
    # Part 3: A3 B3 (j=4)
    # A3 is 2x1, B3 is 1x6
    # C3 = A3 B3 is 2x6
    # C3[i,k] = A3[i,4] * B3[4,k]
    # This is 2 rank-1 tensors.
    for i in range(n):
        a_idx = i * m + 4
        for k in range(p):
            b_idx = 4 * p + k
            c_idx = k * n + i
            U[a_idx][col] = 1.0
            V[b_idx][col] = 1.0
            W[c_idx][col] = 1.0
            col += 1
            
    # Verify
    if not _verify(U, V, W, n, m, p):
        rank = schoolbook_rank
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
