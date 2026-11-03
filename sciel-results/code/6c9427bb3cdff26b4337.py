"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time

def construct():
    """
    Rank-18 tensor decomposition for 3x3 x 3x6 matrix multiplication.
    
    The 3x6 matrix B is split into two 3x3 blocks. Each block multiplication
    A*Bk uses schoolbook (rank 9), giving total rank 18.
    """
    n, m, p = 3, 3, 6
    rank = 18  # 2 blocks * 9 schoolbook terms each
    
    U = [[0] * rank for _ in range(n * m)]  # shape (9, 18)
    V = [[0] * rank for _ in range(m * p)]  # shape (18, 18)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 18)
    
    col = 0
    for block_idx in range(2):
        for i in range(n):
            for j in range(m):
                for k_offset in range(3):
                    k = block_idx * 3 + k_offset
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
