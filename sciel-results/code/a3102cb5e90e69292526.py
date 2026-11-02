"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Strategy: split the middle dimension m=3 into m1=1 and m2=2
    # Part 1: n x m1 x p = 2 x 1 x 10, rank = n*m1*p = 2*1*10 = 20
    # Part 2: n x m2 x p = 2 x 2 x 10, rank = n*m2*p = 2*2*10 = 40
    # Total rank = 20 + 40 = 60 (same as trivial, but let's try to do better)
    
    # Actually, let's think more carefully. The trivial rank is n*m*p = 60.
    # Can we do better?
    
    # For 2x2x1, the minimal rank is 7 (Strassen-like), but that's for 2x2x1.
    # For 2x2x10, we can use the structure.
    
    # Let's try a different approach: use the fact that 2x3 = (2x1) + (2x2)
    # But the rank of 2x3x10 might be less than 60.
    
    # Actually, the rank of matrix multiplication tensor M_{n,m,p} is at most n*m*p.
    # For small dimensions, we can sometimes do better.
    
    # Let's try rank 50: split into 2x1x10 (rank 20) and 2x2x10 (rank 30)
    # Can 2x2x10 be done in rank 30? That would require rank 3 per k, but 2x2x1 needs rank 4.
    # So 2x2x10 needs at least 40.
    
    # What about a different split? 2x3x10 as a whole.
    # The rank of M_{2,3,10}... let's just use the trivial 60 and see if we can shave.
    
    # Actually, let me try rank 55 or so.
    # Alternative: for each pair (i,j), we need p=10 terms. But maybe we can share.
    
    # Let's just go with the trivial decomposition of rank 60, which is guaranteed correct.
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
