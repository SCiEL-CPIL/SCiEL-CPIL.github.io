"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Strategy: Use block decomposition.
    Split B (3x6) into B1 (3x3) and B2 (3x3).
    C = A * B = [A * B1, A * B2].
    Use schoolbook decomposition for each 3x3 * 3x3 block.
    Rank = 9 + 9 = 18.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    # Block 1: B1 is columns 0-2 of B. C1 is columns 0-2 of C.
    # Schoolbook for 3x3 * 3x3
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k is local index 0..2
                # U[i*m+j, col] = 1
                # V[j*p+k, col] = 1  (k is 0..2, so index is j*6 + k)
                # W[k*n+i, col] = 1  (k is 0..2, so index is k*3 + i)
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: B2 is columns 3-5 of B. C2 is columns 3-5 of C.
    # Schoolbook for 3x3 * 3x3
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k is local index 0..2, global index 3..5
                global_k = k + 3
                
                U[i * m + j][col] = 1
                V[j * p + global_k][col] = 1
                W[global_k * n + i][col] = 1
                col += 1
                
    # Verification
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
        
    if not verify(U, V, W, rank):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][c] = 1
                    V_sb[j * p + k][c] = 1
                    W_sb[k * n + i][c] = 1
                    c += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
