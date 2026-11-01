"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # For 2x2x5 matrix multiplication, we seek a rank-minimal decomposition.
    # The trivial schoolbook rank is 2*2*5 = 20.
    # We can do better by exploiting structure.
    
    # Let's try to find a decomposition with rank < 20.
    # One approach: use the fact that matrix multiplication is a bilinear map.
    # We can try to use known results or construct one manually.
    
    # For 2x2x2, the minimal rank is 7 (Strassen).
    # For 2x2x5, we can potentially combine blocks.
    
    # Let's try a direct construction.
    # The tensor T is in R^{(2*2)} x R^{(2*5)} x R^{(5*2)} = R^4 x R^10 x R^10
    # T[i,j,k,l] = 1 if i==l and j==k (in appropriate indexing)
    
    # Actually, let's think about this differently.
    # We want U (4 x r), V (10 x r), W (10 x r) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,l,r] = delta_{i,l} * delta_{j,k}
    
    # Let's try rank 16, 17, 18, 19 and see if we can find a solution.
    # For simplicity, let's start with a known good bound.
    
    # Actually, let's just return the schoolbook for now but with a slightly better structure
    # if we can find one. Otherwise, fall back to 20.
    
    # Let me try to construct a rank-19 solution by merging some terms.
    
    # Alternative: use the fact that we can share some structure.
    # For the 2x2 part, Strassen gives rank 7 for 2x2x2.
    # For the 5th column of B, we need additional terms.
    
    # Let's try a systematic approach:
    # We'll use the schoolbook but try to see if we can reduce it.
    
    # Actually, the minimal rank for 2x2xn is known to be at least 2n+1 for n>=3 (I think).
    # For n=5, that would be 11. But I'm not sure about the exact value.
    
    # Let's try to construct a solution with rank 16-19.
    
    # For now, let's return the schoolbook decomposition with rank 20.
    # We can try to improve this later.
    
    rank = 20
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
