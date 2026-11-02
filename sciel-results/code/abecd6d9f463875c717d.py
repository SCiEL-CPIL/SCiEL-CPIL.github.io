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
    
    Strategy: Split the middle dimension (size 3) into 2+1:
    - C[:, 0:2] = A[:, 0:2] @ B[0:2, :]  (M_{2,2,3})
    - C[:, 2:3] = A[:, 2:3] @ B[2:3, :]  (M_{2,1,3})
    
    For M_{2,2,3}, further split:
    - C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] (M_{2,2,2}, Strassen rank 7)
                   + A[:, 0:2] @ B[0:2, 2:3] (M_{2,2,1}, rank 4)
    For M_{2,1,3}: rank 6
    
    Total rank: 7 + 4 + 6 = 17
    """
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = 17
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]  # (6, 17)
    V = [[0.0] * rank for _ in range(m * p)]  # (9, 17)
    W = [[0.0] * rank for _ in range(p * n)]  # (6, 17)
    
    col = 0
    
    # Part 1: M_{2,2,2} - Strassen for A[:, 0:2] @ B[0:2, 0:2]
    # A indices: i*m + j for i in 0,1 and j in 0,1 -> 0,1,2,3
    # B indices: j*p + k for j in 0,1 and k in 0,1 -> 0,1,3,4
    # C indices: k*n + i for k in 0,1 and i in 0,1 -> 0,1,2,3
    
    # A elements: a=A[0,0]=idx0, b=A[0,1]=idx1, c=A[1,0]=idx2, d=A[1,1]=idx3
    # B elements: e=B[0,0]=idx0, f=B[0,1]=idx1, g=B[1,0]=idx3, h=B[1,1]=idx4
    
    # M1 = (a+d)(e+h)
    U[0][col] = 1.0; U[3][col] = 1.0
    V[0][col] = 1.0; V[4][col] = 1.0
    W[0][col] = 1.0; W[1][col] = 1.0; W[2][col] = 1.0; W[3][col] = 1.0
    col += 1  # col=1
    
    # M2 = (c+d)e
    U[2][col] = 1.0; U[3][col] = 1.0
    V[0][col] = 1.0
    W[2][col] = 1.0; W[3][col] = 1.0
    col += 1  # col=2
    
    # M3 = a(f-h)
    U[0][col] = 1.0
    V[1][col] = 1.0; V[4][col] = -1.0
    W[1][col] = 1.0; W[3][col] = 1.0
    col += 1  # col=3
    
    # M4 = d(g-e)
    U[3][col] = 1.0
    V[3][col] = 1.0; V[0][col] = -1.0
    W[0][col] = 1.0; W[2][col] = 1.0
    col += 1  # col=4
    
    # M5 = (a+b)h
    U[0][col] = 1.0; U[1][col] = 1.0
    V[4][col] = 1.0
    W[0][col] = -1.0; W[1][col] = 1.0
    col += 1  # col=5
    
    # M6 = (c-a)(e+f)
    U[2][col] = 1.0; U[0][col] = -1.0
    V[0][col] = 1.0; V[1][col] = 1.0
    W[3][col] = 1.0
    col += 1  # col=6
    
    # M7 = (b-d)(g+h)
    U[1][col] = 1.0; U[3][col] = -1.0
    V[3][col] = 1.0; V[4][col] = 1.0
    W[0][col] = 1.0; W[3][col] = -1.0
    col += 1  # col=7
    
    # Part 2: M_{2,2,1} - A[:, 0:2] @ B[0:2, 2:3]
    # C[:, 2] = A[:, 0:2] @ B[0:2, 2]
    # B index for column 2: j*p + 2 -> j=0: idx2, j=1: idx5
    # C index for column 2: k=2, i=0,1 -> 2*2+0=4, 2*2+1=5
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    U[0][col] = 1.0
    V[2][col] = 1.0
    W[4][col] = 1.0
    col += 1  # col=8
    
    U[1][col] = 1.0
    V[5][col] = 1.0
    W[4][col] = 1.0
    col += 1  # col=9
    
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    U[2][col] = 1.0
    V[2][col] = 1.0
    W[5][col] = 1.0
    col += 1  # col=10
    
    U[3][col] = 1.0
    V[5][col] = 1.0
    W[5][col] = 1.0
    col += 1  # col=11
    
    # Part 3: M_{2,1,3} - A[:, 2:3] @ B[2:3, :]
    # C[i,k] = A[i,2] * B[2,k]
    # A index for column 2: i*m + 2 -> i=0: idx2, i=1: idx5
    # B index for row 2: 2*p + k -> k=0: idx6, k=1: idx7, k=2: idx8
    # C index: k*n + i
    
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0, 1, 2
            idx_u = i * m + 2  # 2, 5
            idx_v = 2 * p + k  # 6, 7, 8
            idx_w = k * n + i  # 0,2,4 for i=0; 1,3,5 for i=1
            
            U[idx_u][col] = 1.0
            V[idx_v][col] = 1.0
            W[idx_w][col] = 1.0
            col += 1  # col goes 11->17
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
