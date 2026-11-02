"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The rank of the tensor for $2 \times 2 \times 10$ matrix multiplication is 35.
    This is derived from the fact that the multiplication can be viewed as 5 independent
    $2 \times 2 \times 2$ multiplications, each requiring rank 7 (Strassen's algorithm).
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    # U: shape (n*m, rank) = (4, 35)
    # V: shape (m*p, rank) = (20, 35)
    # W: shape (p*n, rank) = (20, 35)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Coefficients for A (indices 0:a, 1:b, 2:c, 3:d)
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    # Coefficients for B (indices 0:e, 1:f, 2:g, 3:h)
    strassen_B = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    # Coefficients for C (indices 0:C00, 1:C01, 2:C10, 3:C11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    w_contrib[0][0] = 1; w_contrib[0][3] = 1   # P1
    w_contrib[1][2] = 1; w_contrib[1][3] = -1  # P2
    w_contrib[2][1] = 1; w_contrib[2][3] = 1   # P3
    w_contrib[3][0] = 1; w_contrib[3][2] = 1   # P4
    w_contrib[4][0] = -1; w_contrib[4][1] = 1  # P5
    w_contrib[5][3] = 1                       # P6
    w_contrib[6][0] = 1                       # P7
    
    col = 0
    for b in range(5):
        for r in range(7):
            # U: Same for all blocks
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: Specific to block b
            # B indices: e=2b, f=2b+1, g=10+2b, h=10+2b+1
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            # W: Specific to block b
            # C indices: C00=4b, C01=4b+2, C10=4b+1, C11=4b+3
            # Note: W index is k*n + i.
            # C00: k=2b, i=0 -> 2b*2+0 = 4b
            # C01: k=2b+1, i=0 -> (2b+1)*2+0 = 4b+2
            # C10: k=2b, i=1 -> 2b*2+1 = 4b+1
            # C11: k=2b+1, i=1 -> (2b+1)*2+1 = 4b+3
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
