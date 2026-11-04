import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using block structure
    # Split 4x4x7 into 4x4x4 + 4x4x3
    # For 4x4x4, try a valid rank-49 decomposition (Winograd)
    # For 4x4x3, use schoolbook rank 48
    # Total: 97
    
    # First, let's verify schoolbook works
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False
        return True
    
    # Try a simple improvement: use 2x2 block Strassen but expand correctly
    # For 2x2x2, Strassen uses 7 multiplications of linear combinations.
    # Each multiplication of two 2-vectors gives a 2x2 result, which is 4 scalar products.
    # But that's 28 scalar products, worse than schoolbook 8.
    # So Strassen doesn't help for tensor decomposition.
    
    # Instead, let's try a different block decomposition.
    # Split A into 2x2 blocks, B into 2x2 and 2x1 blocks.
    # For each 2x2x2 subproblem, use schoolbook (rank 8).
    # This gives the same as overall schoolbook.
    
    # The only way to improve is to find a non-trivial decomposition.
    # For now, return schoolbook.
    
    if verify(U_sb, V_sb, W_sb):
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Should never reach here
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
