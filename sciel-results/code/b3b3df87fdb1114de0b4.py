"""Strassen-based rank decomposition for 3x4 @ 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build a decomposition of M_{3,4,8}
    # Strategy: decompose the 3 rows into 1+2 blocks
    # 1x4 @ 4x8: rank 32 (schoolbook)
    # 2x4 @ 4x8: use Strassen on 2x2 blocks of the 4x8, giving rank 2*4*8/2 = 64? 
    # Actually let's think more carefully.
    
    # M_{2,4,8}: We can view this as 2x2 blocks where each block is 1x2 times 2x2...
    # Let's use the fact that M_{2,4,8} can be decomposed using Strassen on the 4x4 subproblem.
    # Actually, let me just use a direct approach.
    
    # For M_{n,m,p}, the schoolbook rank is n*m*p.
    # For M_{2,4,8}: 
    #   Split m=4 into two 2s: B = [B1 | B2] where B1, B2 are 2x8
    #   A = [A1; A2] where A1, A2 are 1x2
    #   AB = A1*B1 + A2*B2
    #   A1*B1 is 1x2 @ 2x8, rank 16
    #   A2*B2 is 1x2 @ 2x8, rank 16
    #   Total: 32. That's the same as schoolbook (2*4*8=64)? No, 32 < 64!
    #   Wait, that's not right. A is 2x4, so A1 is 1x4, A2 is 1x4.
    #   B is 4x8. B1 is first 2 cols (4x2), B2 is last 2 cols (4x2).
    #   A1*B1: 1x4 @ 4x2 = 1x2, rank 8
    #   A2*B2: 1x4 @ 4x2 = 1x2, rank 8
    #   But we need the full product, not just these.
    
    # Let me reconsider. M_{2,4,8} schoolbook rank = 64.
    # Can we do better?
    # Use Strassen: M_{2,4,8} >= M_{2,2,2} tensor something?
    
    # Let me just use a simple approach: pad 3 to 4, use Strassen M_{4,4,8} if possible.
    # M_{4,4,8}: rank <= ? 
    # M_{4,4,4} has rank 49 (Coppersmith). M_{4,4,8} = M_{4,4,4} + M_{4,4,4} shifted = 98?
    # Or M_{4,4,8} via Strassen on the 4x4 part: 4*4*8 with Strassen on 4x4 gives...
    
    # Actually, let's just implement a valid decomposition. The schoolbook gives 96.
    # Let me try to get below 96 using the 1+2 split.
    
    # M_{1,4,8}: rank 32 (each entry is one product)
    # M_{2,4,8}: 
    #   Write A as 2x4, B as 4x8.
    #   Split columns of B into 4 groups of 2: B = [B_0 | B_1 | B_2 | B_3], each 4x2
    #   Split rows of A... no, A has 2 rows.
    #   Product C = AB, C is 2x8.
    #   C[i, 2j] and C[i, 2j+1] depend on row i of A and columns 2j, 2j+1 of B.
    #   For fixed i and j: C[i, 2j:2j+2] = A[i,:] * B[2j:2j+2, :]
    #   This is 1x4 @ 4x2 = 1x2, which needs 8 products.
    #   Total: 2*4*8 = 64. Same as schoolbook.
    
    # Let me try another split. For M_{2,4,8}, use the identity:
    # M_{2,4,8} <= M_{2,2,2} * M_{1,2,4} in some tensor product sense?
    # Not straightforward.
    
    # I'll just implement the schoolbook for now, which gives rank 96.
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
