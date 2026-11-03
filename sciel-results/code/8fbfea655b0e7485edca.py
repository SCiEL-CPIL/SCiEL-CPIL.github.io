import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We decompose into 3 blocks of 2x3x3
    # Each block uses a rank 7 decomposition of 3x3x3 restricted to 2x3x3
    # Total rank = 3 * 7 = 21
    
    # Rank 7 decomposition for 3x3x3 matrix multiplication
    # This is a known decomposition.
    # U: 9x7, V: 9x7, W: 9x7
    # Indices for U: i*3+j, V: j*3+k, W: k*3+i
    
    # I will use a specific rank 7 decomposition.
    # One such decomposition is:
    # M1 = (A11+A22+A33)(B11+B22+B33)
    # ... this is getting complicated.
    
    # Let's use a simpler approach: Rank 9 for 2x3x3 (schoolbook)
    # Total rank 27.
    
    rank_block = 9
    total_rank = 3 * rank_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(3):
        offset = block * rank_block
        # For each block, we compute A (2x3) * B_block (3x3)
        # Schoolbook for 2x3x3 has rank 9
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    col = offset + (i * m + j) * 3 + k
                    # Wait, the column index in the block decomposition
                    # We need to map (i,j,k) to a column in 0..8
                    col_in_block = (i * m + j) * 3 + k
                    col = offset + col_in_block
                    
                    U[i * m + j][col] = 1
                    V[j * p + (block * 3 + k)][col] = 1
                    W[(block * 3 + k) * n + i][col] = 1
                    
    return {"U": U, "V": V, "W": W}
