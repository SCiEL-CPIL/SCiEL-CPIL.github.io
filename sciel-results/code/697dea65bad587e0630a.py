"""Kronecker-style decomposition of 3x5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    
    # Strategy: Use block decomposition with Strassen for 2x2 blocks
    # Split m=5 into [2,3] and p=6 into [2,4]
    # This gives 4 blocks: (3x2x2), (3x3x2), (3x2x4), (3x3x4)
    
    # For each block, use the best known decomposition:
    # 3x2x2: Use Strassen on the 2x2 part, rank 7 (but we need 3 rows)
    # Actually, for 3x2x2, we can use schoolbook: rank 12
    # But we can do better: view as 3 separate 1x2x2, each rank 2, total rank 6
    # Wait, no. Let me think more carefully.
    
    # For 3x2x2: C = A * B where A is 3x2, B is 2x2, C is 3x2
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is 3*2*2 = 12 multiplications schoolbook
    # But we can use Strassen-like methods
    
    # Actually, let me just use schoolbook for correctness first, then optimize
    
    # Let me try a different approach: use the fact that we can combine blocks
    
    # For now, use schoolbook for all blocks:
    # Block (3x2x2): rank 12
    # Block (3x3x2): rank 18
    # Block (3x2x4): rank 24
    # Block (3x3x4): rank 36
    # Total: 90
    
    # Let me try to use Strassen for the 2x2 sub-blocks to reduce rank
    # For 3x2x2: we can view this as 3 separate 1x2x2 multiplications
    # Each 1x2x2 has rank 2 (schoolbook), so 3*2 = 6
    # But wait, that's not right either. Let me be more careful.
    
    # Actually, for 3x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    # This is 6 separate 1x2x2 multiplications, each of rank 2
    # So total rank for 3x2x2 is 12 (schoolbook)
    # But we can do better with Strassen-like methods
    
    # Let me just use schoolbook to ensure correctness
    rank = n * m * p  # 90
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
# EVOLVE-BLOCK-END
