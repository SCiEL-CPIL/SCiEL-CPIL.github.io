"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    n, m, p = DIMENSIONS
    # Use a decomposition based on treating 3x6 as two 3x3 blocks
    # with shared A factors, using a rank-21 decomposition per 3x3 block
    # Total rank: 42
    
    # For 3x3 matrix multiplication, use a rank-21 decomposition
    # (simpler than optimal 23, but explicit)
    # Then use it twice for the two 3x3 blocks of the 3x6 matrix
    
    # Actually, let's use a direct construction with rank 36
    # We'll use 6 groups of 6 terms, where each group handles 
    # interactions between rows of A and columns of B
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 9 x rank
    V = [[0] * rank for _ in range(m * p)]  # 18 x rank  
    W = [[0] * rank for _ in range(p * n)]  # 18 x rank
    
    # Construction: for each of the 6 output columns k, and for each 
    # "Strassen-like" term, we create a rank-1 component
    # 
    # Simpler: use schoolbook but merge some terms
    # Let's just use a valid rank-48 decomposition as a safe fallback
    # and hope for a better one
    
    # Actually, let's construct a rank-36 decomposition:
    # For each i (row of A), j (col of A / row of B), and a "group" of columns
    # 
    # I'll use the following: 3 (i) * 3 (j) * 4 (groups of 1.5 cols) = 36
    # This is getting complicated. Let me just use schoolbook rank 54
    # but that's the baseline.
    
    # Let me try a different approach: use the fact that we can write
    # the multiplication as a sum of rank-1 tensors where we share
    # the A[i,j] factor across multiple k values.
    
    # For each (i,j), A[i,j] is multiplied by B[j,k] for all k.
    # So for fixed (i,j), we have 6 terms: A[i,j]*B[j,k]*delta(k,i) for k=0..5
    # We can combine these into fewer rank-1 terms if we use
    # linear combinations.
    
    # Let me just implement a verified rank-42 decomposition:
    # Use 7 terms per row of A, for 3 rows = 21, but that doesn't work.
    
    # I'll use a simple but valid approach: rank 48
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Standard schoolbook term
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        if col >= rank:
            break
    
    # If we ran out of columns, we need more rank
    if col < rank:
        # Fill remaining with zeros (they won't contribute)
        pass
    else:
        # We need exactly rank terms, but schoolbook needs 54
        # So this construction is invalid for rank < 54
        # Let me just return schoolbook for now
        pass
    
    # Just return schoolbook for safety
    rank = n * m * p
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
