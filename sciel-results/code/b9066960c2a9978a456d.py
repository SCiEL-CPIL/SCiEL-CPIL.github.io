"""Rewritten program: rank-minimal tensor decomposition for 3x3 by 3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3 by 3x7 matrix multiplication.
    
    Strategy: Split the 3x7 matrix B into three parts:
      - B1: columns 0,1,2 (3x3) -> schoolbook rank 9
      - B2: columns 3,4,5 (3x3) -> schoolbook rank 9  
      - B3: column 6 (3x1) -> schoolbook rank 9
    
    Total rank = 9 + 9 + 9 = 27
    
    U has shape (n*m, rank) = (9, rank)
    V has shape (m*p, rank) = (21, rank)
    W has shape (p*n, rank) = (21, rank)
    W uses convention W[k*n+i, r] where k=output row, i=output col.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    def build_block_UVW(start_col, num_cols, col_offset, rank, U, V, W):
        """Add schoolbook terms for a block of columns starting at start_col."""
        for i in range(n):
            for j in range(m):
                for k_local in range(num_cols):
                    k_global = start_col + k_local
                    U[i * m + j][col_offset] = 1
                    V[j * p + k_global][col_offset] = 1
                    W[k_global * n + i][col_offset] = 1
                    col_offset += 1
        return col_offset
    
    # Attempt 1: Block schoolbook (rank 27)
    rank = 27
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 21 rows
    W = [[0] * rank for _ in range(p * n)]   # 21 rows
    
    col_offset = 0
    # Block 1: columns 0,1,2
    col_offset = build_block_UVW(0, 3, col_offset, rank, U, V, W)
    # Block 2: columns 3,4,5
    col_offset = build_block_UVW(3, 3, col_offset, rank, U, V, W)
    # Block 3: column 6
    col_offset = build_block_UVW(6, 1, col_offset, rank, U, V, W)
    
    assert col_offset == 27, f"Expected 27, got {col_offset}"
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: full schoolbook (rank 63)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
