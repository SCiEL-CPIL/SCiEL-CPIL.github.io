import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = 3, 3, 7
    # Use the schoolbook decomposition for the 3x3 part (rank 9) and replicate
    # for each of the 7 columns. This gives rank 9*7 = 63.
    # However, we can do better by using the fact that the 3x3 multiplication
    # can be done in rank 7 using Strassen-like tricks, but for exactness and
    # simplicity, let's first try to see if we can reduce the rank per column block.
    
    # Actually, the minimum rank for 3x3x3 matrix multiplication is 7 (Strassen).
    # So we can decompose each 3x3 block in rank 7, giving total rank 7*7=49.
    
    # Let's implement the 7-rank decomposition for 3x3x3.
    # The 7 terms are:
    # (1,1,1), (1,1,2), (1,2,1), (2,1,1), (2,2,2), (1,2,2), (2,1,2) ... no, that's not right.
    
    # Let me use the known 7-rank decomposition for 3x3x3.
    # I'll use the one from the literature.
    
    # For now, let's use the simple 9-rank per block, giving 63 total.
    # This is the same as the current program.
    
    # Let me try a different approach: use the fact that for a fixed i, the row i of C
    # is a linear combination of rows of B. So we can factor differently.
    
    # Actually, let me just implement the schoolbook for now, but with a twist:
    # group by (i, k) and use the m=3 terms for j. This doesn't help.
    
    # Let me try to use a low-rank decomposition for 3x3.
    # The minimum rank is 7. Let me find the 7-rank decomposition.
    
    # I'll use the following 7-rank decomposition for 3x3x3:
    # This is based on the Strassen algorithm for 3x3.
    
    # Actually, let me just use rank 9 per block (63 total) for now, and try to improve.
    
    rank = 63
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
