"""
Optimized tensor decomposition for 2x3 by 3x9 matrix multiplication.

Strategy: 
- Split B into B[0:2, 0:8] (2x8), B[2, 0:8] (1x8), and B[:, 8] (3x1).
- A[:,0:2] (2x2) x B[0:2, 0:8] (2x8): 4 Strassen blocks, rank 28.
- A[:,2] (2x1) x B[2, 0:8] (1x8): schoolbook, rank 16.
- A (2x3) x B[:,8] (3x1): rank-3 decomposition, rank 3.
Total rank: 28 + 16 + 3 = 47.

Index conventions:
  U has shape (n*m, rank) = (6, rank), with U[i*m + j, c]
  V has shape (m*p, rank) = (27, rank), with V[j*p + k, c]
  W has shape (p*n, rank) = (18, rank), with W[k*n + i, c]
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-47 decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    - A[:,0:2] (2x2) x B[0:2, 0:8] (2x8): 4 Strassen blocks, rank 28.
    - A[:,2] (2x1) x B[2, 0:8] (1x8): schoolbook, rank 16.
    - A (2x3) x B[:,8] (3x1): rank-3 decomposition, rank 3.
    Total rank: 47.
    
    The trace convention is W[k*n+i, r].
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = 47
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 47)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 47)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 47)
    
    col = 0
    
    # A indices:
    # u_a = 0 (A[0,0]), u_b = 1 (A[0,1]), u_c = 3 (A[1,0]), u_d = 4 (A[1,1])
    # u_ar0 = 2 (A[0,2]), u_ar1 = 5 (A[1,2])
    u_a = 0
    u_b = 1
    u_c = 3
    u_d = 4
    u_ar0 = 2
    u_ar1 = 5
    
    # Part 1: A[:,0:2] (2x2) * B[0:2, 0:8] (2x8)
    # 4 blocks of 2x2 (columns 0-7 of B)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # Strassen's 7 products for 2x2x2
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output indices: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,k0] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0,k1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1,k0] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # Part 2: A[:,2] (2x1) * B[2, 0:8] (1x8)
    # 8 schoolbook products
    for k in range(8):
        v_b = 2 * p + k  # B[2, k]
        
        # Q_k = A[0,2] * B[2,k]
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A[1,2] * B[2,k]
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    # Part 3: A (2x3) * B[:,8] (3x1) - rank 3 decomposition
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8] + A[0,2]*B[2,8]
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8] + A[1,2]*B[2,8]
    
    # Product 0: for j=0
    # U[0,0]=1 (A[0,0]), U[3,0]=1 (A[1,0]), V[0*9+8,0]=1 (B[0,8])
    # W[8*2+0,0]=1 (C[0,8]), W[8*2+1,0]=1 (C[1,8])
    v_b0 = 0 * p + 8  # B[0,8]
    w_c00 = 8 * n + 0  # C[0,8]
    w_c10 = 8 * n + 1  # C[1,8]
    
    U[u_a][col] += 1      # A[0,0]
    U[u_c][col] += 1      # A[1,0]
    V[v_b0][col] += 1     # B[0,8]
    W[w_c00][col] += 1    # C[0,8]
    W[w_c10][col] += 1    # C[1,8]
    
    # Product 1: for j=1
    # U[1,1]=1 (A[0,1]), U[4,1]=1 (A[1,1]), V[1*9+8,1]=1 (B[1,8])
    v_b1 = 1 * p + 8  # B[1,8]
    
    U[u_b][col+1] += 1    # A[0,1]
    U[u_d][col+1] += 1    # A[1,1]
    V[v_b1][col+1] += 1   # B[1,8]
    W[w_c00][col+1] += 1  # C[0,8]
    W[w_c10][col+1] += 1  # C[1,8]
    
    # Product 2: for j=2
    # U[2,2]=1 (A[0,2]), U[5,2]=1 (A[1,2]), V[2*9+8,2]=1 (B[2,8])
    v_b2 = 2 * p + 8  # B[2,8]
    
    U[u_ar0][col+2] += 1   # A[0,2]
    U[u_ar1][col+2] += 1   # A[1,2]
    V[v_b2][col+2] += 1    # B[2,8]
    W[w_c00][col+2] += 1   # C[0,8]
    W[w_c10][col+2] += 1   # C[1,8]
    
    col += 3
    
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > 115:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
