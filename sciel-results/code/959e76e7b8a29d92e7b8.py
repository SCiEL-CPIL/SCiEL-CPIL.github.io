import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find a rank-28 decomposition using two independent 2x3x3 blocks
    if time.time() < deadline - 20:
        result = _try_rank28_decomposition(n, m, p, deadline)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid."""
    r = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank28_decomposition(n, m, p, deadline):
    """Try to construct a rank-28 decomposition using two rank-14 2x3x3 blocks."""
    # For 2x3x3, use a structured decomposition
    # U1: 6x14, V1: 9x14, W1: 6x14
    
    # Construct a rank-14 decomposition for 2x3x3
    # Use Strassen for 2x2x2 subblock + additional terms
    
    # Strassen terms for 2x2x2 (j=0,1; k=0,1):
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b10-b11)
    # m4 = a11*(b01-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b11+b01)
    
    # Additional terms for the remaining products:
    # m8 = a02*b20
    # m9 = a02*b21
    # m10 = a02*b22
    # m11 = a12*b20
    # m12 = a12*b21
    # m13 = a12*b22
    # m14 = a00*b02 + a01*b12 (combined)
    # m15 = a10*b02 + a11*b12 (combined)
    
    # Wait, that's 15 terms, not 14. Let me try to combine more.
    
    # For now, return None to use schoolbook fallback
    return None

# EVOLVE-BLOCK-END
