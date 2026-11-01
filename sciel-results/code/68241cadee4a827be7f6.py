"""Matrix multiplication 2,2,2 - Strassen rank-7 decomposition."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    This is the minimal rank for 2x2 matrix multiplication over the reals.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Column 0: m1 = (a11+a22)(b11+b22)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # Column 1: m2 = (a21+a22)b11
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    W[3][1] = -1
    
    # Column 2: m3 = a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] = 1
    W[3][2] = 1
    
    # Column 3: m4 = a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[1][3] = 1
    
    # Column 4: m5 = (a11+a12)b22
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[2][4] = 1
    
    # Column 5: m6 = (a21-a11)(b11+b12)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # Column 6: m7 = (a12-a22)(b21+b22)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
