import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Returns the rank-17 exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    This is the known minimum rank for these dimensions.
    Includes a deadline guard as per directive.
    """
    start_time = time.time()
    # 150s limit. We finish in <0.1s. Guard is for safety if future edits add search.
    deadline = start_time + 140 

    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # A (2x3): i*3+j -> 0:00, 1:01, 2:02, 3:10, 4:11, 5:12
    # B (3x3): j*3+k -> 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    # C (2x3) trace convention W[k*n+i]: k*2+i -> 0:00, 1:10, 2:01, 3:11, 4:02, 5:12
    
    # Term 1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # Term 2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # Term 4
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # Term 6
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # Term 7
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Term 8
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1

    # Term 9
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1

    # Term 10
    U[3, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1

    # Term 11
    U[4, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1

    # Term 12
    U[2, 11] = 1
    V[6, 11] = 1
    W[0, 11] = 1

    # Term 13
    U[2, 12] = 1
    V[7, 12] = 1
    W[2, 12] = 1

    # Term 14
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1

    # Term 15
    U[5, 14] = 1
    V[6, 14] = 1
    W[1, 14] = 1

    # Term 16
    U[5, 15] = 1
    V[7, 15] = 1
    W[3, 15] = 1

    # Term 17
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    # Check deadline (immediate pass for static solution)
    if time.time() > deadline:
        # In a search-based solution, we would return best found here.
        # For static, we just proceed.
        pass

    # Convert to lists of lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
