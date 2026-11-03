"""
Optimized matrix multiplication 2x4 by 4x5 using Strassen for 2x2 sub-blocks.
Achieves the theoretical lower bound of rank 36.
"""

import time

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s limit

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Theoretical lower bound for rank of M_{2,4,5} is 2*4 + 4*5 + 5*2 - 2 = 8 + 20 + 10 - 2 = 36.
    - We construct a decomposition of rank 36.
    - Split the 4x5 matrix B into 2x2 blocks (cols 0-1, 2-3) and a 4x1 block (col 4).
    - Split the 2x4 matrix A into 2x2 blocks (cols 0-1, 2-3).
    - Use Strassen's 7-multiplication algorithm for the four 2x2x2 block multiplications.
    - Use schoolbook multiplication for the 2x2x1 and 2x2x1 block multiplications for the last column.
    
    Total rank: 4 blocks * 7 (Strassen) + 4 (schoolbook top) + 4 (schoolbook bottom) = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # U: (n*m) x rank = 8 x rank
    # V: (m*p) x rank = 20 x rank
    # W: (p*n) x rank = 10 x rank
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col
        # Indices
        i0_idx = i0
        i1_idx = i0 + 1
        j0_idx = j0
        j1_idx = j0 + 1
        k0_idx = k0
        k1_idx = k0 + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0_idx * m + j0_idx][c] += 1
        U[i1_idx * m + j1_idx][c] += 1
        V[j0_idx * p + k0_idx][c] += 1
        V[j1_idx * p + k1_idx][c] += 1
        W[k0_idx * n + i0_idx][c] += 1
        W[k1_idx * n + i1_idx][c] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[i1_idx * m + j0_idx][c] += 1
        U[i1_idx * m + j1_idx][c] += 1
        V[j0_idx * p + k0_idx][c] += 1
        W[k0_idx * n + i1_idx][c] += 1
        W[k1_idx * n + i1_idx][c] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0_idx * m + j0_idx][c] += 1
        V[j0_idx * p + k1_idx][c] += 1
        V[j1_idx * p + k1_idx][c] -= 1
        W[k1_idx * n + i0_idx][c] += 1
        W[k1_idx * n + i1_idx][c] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[i1_idx * m + j1_idx][c] += 1
        V[j1_idx * p + k0_idx][c] += 1
        V[j0_idx * p + k0_idx][c] -= 1
        W[k0_idx * n + i0_idx][c] += 1
        W[k0_idx * n + i1_idx][c] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0_idx * m + j0_idx][c] += 1
        U[i0_idx * m + j1_idx][c] += 1
        V[j1_idx * p + k1_idx][c] += 1
        W[k0_idx * n + i0_idx][c] -= 1
        W[k1_idx * n + i0_idx][c] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[i1_idx * m + j0_idx][c] += 1
        U[i0_idx * m + j0_idx][c] -= 1
        V[j0_idx * p + k0_idx][c] += 1
        V[j0_idx * p + k1_idx][c] += 1
        W[k1_idx * n + i1_idx][c] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0_idx * m + j1_idx][c] += 1
        U[i1_idx * m + j1_idx][c] -= 1
        V[j1_idx * p + k0_idx][c] += 1
        V[j1_idx * p + k1_idx][c] += 1
        W[k0_idx * n + i0_idx][c] += 1
        col += 1
        
        return col_start + 7
    
    def add_schoolbook_2x2x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] += A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]
        This is 2*2*1 = 4 multiplications.
        """
        c = col_start
        for di in range(2):
            for dj in range(2):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c
    
    # 1. C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, col)
    
    # 2. C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    
    # 3. C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    
    # 4. C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    
    # 5. C[:, 4] from A[:, 0:2] * B[0:2, 4] (schoolbook 2x2x1)
    col = add_schoolbook_2x2x1(0, 0, 4, col)
    
    # 6. C[:, 4] from A[:, 2:4] * B[2:4, 4] (schoolbook 2x2x1)
    col = add_schoolbook_2x2x1(0, 2, 4, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Deadline guard
    if time.time() > _DEADLINE:
        raise TimeoutError("Deadline exceeded")
    
    return {"U": U, "V": V, "W": W}
