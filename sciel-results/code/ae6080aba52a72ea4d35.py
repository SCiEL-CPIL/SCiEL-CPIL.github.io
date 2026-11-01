import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen's algorithm on 6 disjoint 2x2 blocks of the second matrix.
    Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2
    rank = num_pairs * 7
    
    # Initialize U, V, W as lists of lists for easy assignment
    # U shape: (n*m, rank) = (4, 42)
    # V shape: (m*p, rank) = (24, 42)
    # W shape: (p*n, rank) = (24, 42)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1: (A00+A11)*(B00+B11)
    # M2: (A10+A11)*B00
    # M3: A00*(B01-B11)
    # M4: A11*(B10-B00)
    # M5: (A00+A01)*B11
    # M6: (A10-A00)*(B00+B01)
    # M7: (A01-A11)*(B10+B11)
    
    # Output combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices in U: A[i,j] -> i*m+j
    # Indices in V: B[j,k] -> j*p+k
    # Indices in W: C[k,i] -> k*n+i
    
    for t in range(num_pairs):
        base = t * 7
        # Local B column indices: 2*t, 2*t+1
        b_col_0 = 2 * t
        b_col_1 = 2 * t + 1
        
        for s in range(7):
            c = base + s
            
            if s == 0:
                # M1
                U[0][c] = 1
                U[3][c] = 1
                V[0 * p + b_col_0][c] = 1
                V[1 * p + b_col_1][c] = 1
                W[b_col_0 * n + 0][c] = 1
                W[b_col_1 * n + 1][c] = 1
            elif s == 1:
                # M2
                U[2][c] = 1
                U[3][c] = 1
                V[0 * p + b_col_0][c] = 1
                W[b_col_0 * n + 1][c] = 1
                W[b_col_1 * n + 1][c] = -1
            elif s == 2:
                # M3
                U[0][c] = 1
                V[0 * p + b_col_1][c] = 1
                V[1 * p + b_col_1][c] = -1
                W[b_col_1 * n + 0][c] = 1
                W[b_col_1 * n + 1][c] = 1
            elif s == 3:
                # M4
                U[3][c] = 1
                V[1 * p + b_col_0][c] = 1
                V[0 * p + b_col_0][c] = -1
                W[b_col_0 * n + 0][c] = 1
                W[b_col_0 * n + 1][c] = 1
            elif s == 4:
                # M5
                U[0][c] = 1
                U[1][c] = 1
                V[1 * p + b_col_1][c] = 1
                W[b_col_0 * n + 0][c] = -1
                W[b_col_1 * n + 0][c] = 1
            elif s == 5:
                # M6
                U[2][c] = 1
                U[0][c] = -1
                V[0 * p + b_col_0][c] = 1
                V[0 * p + b_col_1][c] = 1
                W[b_col_1 * n + 1][c] = 1
            elif s == 6:
                # M7
                U[1][c] = 1
                U[3][c] = -1
                V[1 * p + b_col_0][c] = 1
                V[1 * p + b_col_1][c] = 1
                W[b_col_0 * n + 0][c] = 1
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
