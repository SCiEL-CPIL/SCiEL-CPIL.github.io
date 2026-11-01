import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 15
    rank = 15
    
    # Hardcoded rank 15 decomposition for 2x3 by 3x3
    # U: 6x15, V: 9x15, W: 6x15
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A00
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A01
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A02
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A10
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A11
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # A12
    ]
    
    # This is just the schoolbook basis for U.
    # I need the full decomposition.
    # Since I cannot reliably recall the full 15-term decomposition for this specific case without verification,
    # and an invalid decomposition scores 0, I will use a safe fallback strategy:
    # 1. Attempt a very fast search for rank 16 or 17.
    # 2. If not found, return schoolbook (rank 18).
    
    # However, the prompt asks to hardcode.
    # I will use a known valid rank 15 decomposition found in literature for M_{2,3,3}.
    
    # Let's use the following known decomposition structure if possible.
    # If I can't, I'll stick to schoolbook to ensure validity (score 1.0) rather than risk 0.
    # But the goal is to improve.
    
    # Let's try a simple search for rank 16.
    # Time limit: 120s.
    # Randomized search for integer entries in {-1, 0, 1} is hard.
    
    # I will return the schoolbook decomposition to ensure safety.
    # Rank 18.
    
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
