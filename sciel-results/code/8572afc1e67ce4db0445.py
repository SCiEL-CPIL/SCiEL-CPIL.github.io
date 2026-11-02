"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Rank-42 decomposition via Kronecker product of Strassen (2x2x2, rank 7)
    and vector scaling (1x1x6, rank 6).
    
    The 2x2 * 2x12 multiplication is decomposed as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Partition the 12 columns into 6 pairs. For each pair (k, k+1), we have
    a 2x2 * 2x2 subproblem. Using Strassen's 7-product algorithm for each
    gives rank 7*6 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j for A[i][j]
      V[j*p+k, r]: row = j*p+k for B[j][k]  
      W[k*n+i, r]: row = k*n+i for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For Kronecker product with 6 independent column pairs:
    # Total rank = 7 * 6 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # Strassen factor matrices for 2x2x2
    # U_strassen[ai*m+aj, s] for s in 0..6
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 1],   # A00: appears in M1, M3, M5, M7
        [0, 0, 0, 0, 1, -1, -1],  # A01: appears in M5, M6, M7
        [0, 1, 0, 0, 0, 1, 0],   # A10: appears in M2, M6
        [1, 1, 0, 1, 0, 0, -1],  # A11: appears in M1, M2, M4, M7
    ]
    
    # V_strassen[bj*p+bk, s] for s in 0..6
    V_strassen = [
        [1, 1, -1, -1, 0, 1, 0],  # B00: M1, M2, M3, M4, M6
        [0, 0, 0, 1, 0, 1, 1],   # B01: M4, M6, M7
        [0, 0, 1, 0, 0, 0, 0],   # B10: M3
        [1, 0, -1, 0, 1, 0, 1],  # B11: M1, M3, M5, M7
    ]
    
    # W_strassen[ck*n+ci, s] for s in 0..6
    # C[i,k] where i is row, k is col in 2x2 output
    # W index = k*n + i
    W_strassen = [
        [1, 0, 1, -1, 0, 0, 1],  # C[0,0]: M1, M3, M4, M7
        [0, 0, 0, 1, 1, 0, 0],   # C[0,1]: M4, M5
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]: M2, M4
        [1, -1, 1, 0, 0, 1, 0],  # C[1,1]: M1, M2, M3, M6
    ]
    
    num_pairs = p // 2  # 6
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        base_col = 2 * t
        for s in range(7):
            # U: A factors (same for all pairs)
            for ai in range(n):
                for aj in range(m):
                    U[ai * m + aj][col] = U_strassen[ai * m + aj][s]
            
            # V: B factors (shifted by base_col)
            for bj in range(m):
                for bk in range(2):
                    V[bj * p + base_col + bk][col] = V_strassen[bj * 2 + bk][s]
            
            # W: C factors (shifted by base_col, trace convention: k*n + i)
            for ck in range(2):
                for ci in range(n):
                    W[(base_col + ck) * n + ci][col] = W_strassen[ck * n + ci][s]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
