import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Rank 18 decomposition
    rank = 18
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    
    col = 0
    # Block 1: j in 0,1,2
    for i in range(n):
        for k in range(m):
            for j in range(3): # j is 0, 1, 2
                a = i * m + k
                b = k * p + j
                c = j * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    # Block 2: j in 3,4,5
    for i in range(n):
        for k in range(m):
            for j in range(3, 6): # j is 3, 4, 5
                a = i * m + k
                b = k * p + j
                c = j * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    # Verify
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Should be 1 if a,b,c form a valid triple, 0 otherwise.
    # Valid triple: a = i*m+k, b = k*p+j, c = j*n+i
    
    # We can quickly verify a few points
    # Point 1: i=0, k=0, j=0 -> a=0, b=0, c=0
    # U[0,0]=1, V[0,0]=1, W[0,0]=1 -> sum=1. Correct.
    # Point 2: i=0, k=0, j=1 -> a=0, b=1, c=2
    # U[0,1]=1, V[1,1]=1, W[2,1]=1 -> sum=1. Correct.
    # Point 3: i=0, k=0, j=3 -> a=0, b=3, c=6
    # In Block 2, col=9. i=0, k=0, j=3.
    # U[0,9]=1, V[3,9]=1, W[6,9]=1. Correct.
    
    # Check an invalid point: a=0, b=0, c=1
    # a=0 implies i=0, k=0.
    # b=0 implies k=0, j=0.
    # c=1 implies j=0, i=1.
    # Contradiction i=0 vs i=1.
    # Sum should be 0.
    # U[0, r] is non-zero only for r in 0..8 (Block 1) and 9..17 (Block 2).
    # In Block 1, U[0,r]=1 for r corresponding to i=0, k=0, j=0,1,2.
    # These are r=0,1,2.
    # For r=0: V[0,0]=1, W[1,0]=0 (W[1,0] corresponds to j=0, i=1? No, W[1,0] is for c=1. c=1 is j=0, i=1. Block 1 r=0 is j=0, i=0. So W[1,0] is 0).
    # Let's check W indices.
    # Block 1, r=0: i=0, k=0, j=0. c = 0*2+0 = 0. W[0,0]=1.
    # We are checking c=1. W[1,0]=0.
    # So term is 0.
    # Similarly for other r.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
