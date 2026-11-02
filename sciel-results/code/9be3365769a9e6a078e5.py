import numpy as np

def construct():
    # Dimensions
    n, m, p = 3, 4, 7
    
    # Total Rank
    rank = 76
    
    # Initialize
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 3x3x3 (Rank 23)
    # A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # U rows: i*4+j for i in 0..2, j in 0..2
    # V rows: j*7+k for j in 0..2, k in 0..2
    # W rows: k*3+i for k in 0..2, i in 0..2
    
    # Part 2: 3x3x3 (Rank 23)
    # A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    
    # Part 3: 3x3x1 (Rank 9)
    # A[0:3, 0:3] * B[0:3, 6:7] -> C[0:3, 6:7]
    
    # Part 4: 3x1x7 (Rank 21)
    # A[0:3, 3:4] * B[3:4, 0:7] -> C[0:3, 0:7]
    
    # Hardcoded 3x3x3 Rank 23 decomposition
    # This is the Bini decomposition
    # ...
