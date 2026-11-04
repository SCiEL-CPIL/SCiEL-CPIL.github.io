import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115
    
    # Strategy: column-splitting
    # Split m=4 into 2 blocks of 2, p=8 into 4 blocks of 2
    # Each sub-problem is 3x2x2 matrix multiplication with schoolbook rank 12
    # Total rank = 2 * 4 * 12 = 96... wait that's the same
    
    # Actually, for each (mb, pb) pair, the sub-tensor involves:
    # U indices: i in 0..2, j in mb*2..mb*2+1 -> 6 rows
    # V indices: j in mb*2..mb*2+1, k in pb*2..pb*2+1 -> 4 rows
    # W indices: k in pb*2..pb*2+1, i in 0..2 -> 6 rows
    # This is exactly a 3x2x2 multiplication tensor
    
    # Schoolbook for 3x2x2: rank 12
    # Total: 2 * 4 * 12 = 96. No improvement with schoolbook sub-decompositions.
    
    # We need to find a lower-rank decomposition for 3x2x2.
    # Let's try rank 10 for 3x2x2, giving total rank 80.
    # Or rank 9, giving total rank 72.
    
    # For now, let's use a structured approach to try to find a better decomposition.
    # The key insight: we can try to share structure between the 8 sub-problems.
    
    # Alternative: just use schoolbook but verify correctness
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Try to improve: use the fact that we can potentially find a lower-rank
    # decomposition by exploiting the block structure
    
    # For 3x2x2, let's attempt a rank-10 decomposition
    # This requires solving a system of equations, which is complex
    # For now, return the schoolbook result which is correct
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
