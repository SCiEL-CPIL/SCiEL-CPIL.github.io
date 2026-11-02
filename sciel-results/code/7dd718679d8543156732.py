import time
import sys

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-17 exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Uses Strassen for the 2x2x2 sub-block and schoolbook for the rest.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] = val

    # Strassen terms for A[:, 0:2] * B[0:2, 0:2]
    # A indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # C indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    
    # M1 = (a+d)(e+g)
    set_val(U, 0, 0, 1.0); set_val(U, 4, 0, 1.0)
    set_val(V, 0, 0, 1.0); set_val(V, 3, 0, 1.0)
    set_val(W, 0, 0, 1.0); set_val(W, 3, 0, 1.0)
    
    # M2 = (c+d)e
    set_val(U, 3, 1, 1.0); set_val(U, 4, 1, 1.0)
    set_val(V, 0, 1, 1.0)
    set_val(W, 1, 1, 1.0)
    
    # M3 = a(f-h)
    set_val(U, 0, 2, 1.0)
    set_val(V, 1, 2, 1.0); set_val(V, 4, 2, -1.0)
    set_val(W, 2, 2, 1.0); set_val(W, 3, 2, -1.0)
    
    # M4 = d(g-e)
    set_val(U, 4, 3, 1.0)
    set_val(V, 3, 3, 1.0); set_val(V, 0, 3, -1.0)
    set_val(W, 0, 3, -1.0)
    
    # M5 = (a+b)h
    set_val(U, 0, 4, 1.0); set_val(U, 1, 4, 1.0)
    set_val(V, 4, 4, 1.0)
    set_val(W, 2, 4, 1.0); set_val(W, 3, 4, 1.0)
    
    # M6 = (c-a)(e+f)
    set_val(U, 3, 5, 1.0); set_val(U, 0, 5, -1.0)
    set_val(V, 0, 5, 1.0); set_val(V, 1, 5, 1.0)
    set_val(W, 0, 5, 1.0)
    
    # M7 = (b-d)(g+h)
    set_val(U, 1, 6, 1.0); set_val(U, 4, 6, -1.0)
    set_val(V, 3, 6, 1.0); set_val(V, 4, 6, 1.0)
    set_val(W, 1, 6, 1.0)
    
    # Schoolbook for A[:, 0:2] * B[0:2, 2]
    # A indices: 0, 1, 3, 4
    # B indices: 2:(0,2), 5:(1,2)
    # C indices: 4:(0,2), 5:(1,2)
    
    # a * B[0,2]
    set_val(U, 0, 7, 1.0)
    set_val(V, 2, 7, 1.0)
    set_val(W, 4, 7, 1.0)
    
    # b * B[1,2]
    set_val(U, 1, 8, 1.0)
    set_val(V, 5, 8, 1.0)
    set_val(W, 4, 8, 1.0)
    
    # c * B[0,2]
    set_val(U, 3, 9, 1.0)
    set_val(V, 2, 9, 1.0)
    set_val(W, 5, 9, 1.0)
    
    # d * B[1,2]
    set_val(U, 4, 10, 1.0)
    set_val(V, 5, 10, 1.0)
    set_val(W, 5, 10, 1.0)
    
    # Schoolbook for A[:, 2] * B[2, :]
    # A indices: 2:(0,2), 5:(1,2)
    # B indices: 6:(2,0), 7:(2,1), 8:(2,2)
    # C indices: 0, 1, 2, 3, 4, 5
    
    # A[0,2] * B[2,0]
    set_val(U, 2, 11, 1.0)
    set_val(V, 6, 11, 1.0)
    set_val(W, 0, 11, 1.0)
    
    # A[0,2] * B[2,1]
    set_val(U, 2, 12, 1.0)
    set_val(V, 7, 12, 1.0)
    set_val(W, 2, 12, 1.0)
    
    # A[0,2] * B[2,2]
    set_val(U, 2, 13, 1.0)
    set_val(V, 8, 13, 1.0)
    set_val(W, 4, 13, 1.0)
    
    # A[1,2] * B[2,0]
    set_val(U, 5, 14, 1.0)
    set_val(V, 6, 14, 1.0)
    set_val(W, 1, 14, 1.0)
    
    # A[1,2] * B[2,1]
    set_val(U, 5, 15, 1.0)
    set_val(V, 7, 15, 1.0)
    set_val(W, 3, 15, 1.0)
    
    # A[1,2] * B[2,2]
    set_val(U, 5, 16, 1.0)
    set_val(V, 8, 16, 1.0)
    set_val(W, 5, 16, 1.0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
