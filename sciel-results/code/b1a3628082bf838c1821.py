"""Matrix multiplication 2,2,16 rank decomposition using Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 16 columns of A (which is 2x16) grouped into 8 blocks of 2 columns
    num_blocks = p // m  # 8 blocks
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For C = A * B where A, B are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[C00, C01], [C10, C11]]
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-c)(e+h)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # U index: i*m + j where i is row of A (0,1), j is col of A (0,1)
    # V index: j*p + k where j is row of B (0,1), k is col of B (0,1)
    # W index: k*n + i where k is col of C (0,1), i is row of C (0,1)
    
    rank = num_blocks * 7  # 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # U indices: A[i,j] -> i*m + j
    # U[0,0]=0, U[0,1]=1, U[1,0]=2, U[1,1]=3
    # V indices: B[j,k] -> j*p + k
    # V[0,0]=0, V[0,1]=1, V[1,0]=16, V[1,1]=17 (for block 0)
    # W indices: C[i,k] -> k*n + i
    # W[0,0]=0, W[0,1]=2, W[1,0]=1, W[1,1]=3 (for block 0)
    
    # For block b (0-indexed), the columns of A are 2b and 2b+1
    # So V index for B[j, 2b+k] = j*p + 2b + k
    
    col = 0
    for b in range(num_blocks):
        # Strassen terms
        # m1 = (a+d)(e+h): U[0,0]+U[1,1] times V[0,0]+V[1,1]
        # U: a=A[0,2b]=U[0], d=A[1,2b+1]=U[3]
        # V: e=B[0,2b]=V[2b], h=B[1,2b+1]=V[p+2b+1]
        # W: C00=W[2b*n+0], C11=W[(2b+1)*n+1]
        
        # m1: (A[0,2b] + A[1,2b+1]) * (B[0,2b] + B[1,2b+1])
        U[0, col] += 1.0  # A[0,2b]
        U[3, col] += 1.0  # A[1,2b+1]
        V[2*b, col] += 1.0  # B[0,2b]
        V[p + 2*b + 1, col] += 1.0  # B[1,2b+1]
        W[2*b * n + 0, col] += 1.0  # C[0,2b] = C00
        W[(2*b + 1) * n + 1, col] += 1.0  # C[1,2b+1] = C11
        col += 1
        
        # m2 = (c+d)e: (A[1,2b] + A[1,2b+1]) * B[0,2b]
        U[2, col] += 1.0  # A[1,2b]
        U[3, col] += 1.0  # A[1,2b+1]
        V[2*b, col] += 1.0  # B[0,2b]
        W[2*b * n + 1, col] += 1.0  # C[1,2b] = C10
        col += 1
        
        # m3 = a(f-h): A[0,2b] * (B[0,2b+1] - B[1,2b])
        U[0, col] += 1.0  # A[0,2b]
        V[2*b + 1, col] += 1.0  # B[0,2b+1]
        V[p + 2*b, col] -= 1.0  # -B[1,2b]
        W[(2*b + 1) * n + 0, col] += 1.0  # C[0,2b+1] = C01
        col += 1
        
        # m4 = d(g-e): A[1,2b+1] * (B[1,2b] - B[0,2b])
        U[3, col] += 1.0  # A[1,2b+1]
        V[p + 2*b, col] += 1.0  # B[1,2b]
        V[2*b, col] -= 1.0  # -B[0,2b]
        W[2*b * n + 0, col] += 1.0  # C00
        W[2*b * n + 1, col] += 1.0  # C10
        col += 1
        
        # m5 = (a+b)h: (A[0,2b] + A[0,2b+1]) * B[1,2b+1]
        U[0, col] += 1.0  # A[0,2b]
        U[1, col] += 1.0  # A[0,2b+1]
        V[p + 2*b + 1, col] += 1.0  # B[1,2b+1]
        W[2*b * n + 0, col] -= 1.0  # C00 (negative in C00 formula)
        W[(2*b + 1) * n + 0, col] += 1.0  # C01
        col += 1
        
        # m6 = (c-d)(f+h): (A[1,2b] - A[1,2b+1]) * (B[0,2b+1] + B[1,2b+1])
        U[2, col] += 1.0  # A[1,2b]
        U[3, col] -= 1.0  # -A[1,2b+1]
        V[2*b + 1, col] += 1.0  # B[0,2b+1]
        V[p + 2*b + 1, col] += 1.0  # B[1,2b+1]
        W[(2*b + 1) * n + 1, col] += 1.0  # C11
        col += 1
        
        # m7 = (b-c)(e+h): (A[0,2b+1] - A[1,2b]) * (B[0,2b] + B[1,2b+1])
        U[1, col] += 1.0  # A[0,2b+1]
        U[2, col] -= 1.0  # -A[1,2b]
        V[2*b, col] += 1.0  # B[0,2b]
        V[p + 2*b + 1, col] += 1.0  # B[1,2b+1]
        W[2*b * n + 0, col] += 1.0  # C00
        W[(2*b + 1) * n + 0, col] += 1.0  # C01
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
