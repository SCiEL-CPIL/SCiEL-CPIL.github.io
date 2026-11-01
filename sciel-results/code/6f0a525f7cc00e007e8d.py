"""
Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.

Strategy:
- The schoolbook rank is 2*2*14 = 56.
- We use Strassen-style tricks to reduce rank.
- For 2x2 * 2x14, we view this as 7 independent 2x2 * 2x2 multiplications.
  Each 2x2 * 2x2 can be done in rank 7 (Strassen) instead of rank 8.
  So total rank = 7 * 7 = 49.

The Strassen decomposition for 2x2 * 2x2:
  P1 = (A11 + A22)(B11 + B22)
  P2 = (A21 + A22) B11
  P3 = A11 (B12 - B22)
  P4 = A22 (B21 - B11)
  P5 = (A11 + A12) B22
  P6 = (A21 - A11) (B11 + B12)
  P7 = (A12 - A22) (B21 + B22)

  C11 = P1 + P4 - P5 + P7
  C12 = P3 + P5
  C21 = P2 + P4
  C22 = P1 - P2 + P3 + P6

For our 2x2 * 2x14 problem:
  A is 2x2, B is 2x14, C is 2x14.
  
  We split B into 7 blocks B^(t) of size 2x2 for t=0..6.
  C^(t) = A * B^(t) for each t.
  
  For each block, use Strassen with 7 multiplications.
  Total rank = 7 * 7 = 49.

Tensor convention:
  U has shape (n*m, rank) = (4, 49)
  V has shape (m*p, rank) = (28, 49)
  W has shape (p*n, rank) = (28, 49)
  
  U[i*m+j, r] - coefficient for A_{i,j} in rank-r term
  V[j*p+k, r] - coefficient for B_{j,k} in rank-r term  
  W[k*n+i, r] - coefficient for C_{i,k} in rank-r term
  
  The tensor element T_{i,j,k} = 1 for all valid i,j,k.
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
"""

import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # Strassen-based rank-49 decomposition
    # 7 blocks of 2x2, each using 7 Strassen multiplications
    num_blocks = p // 2  # 7
    strassen_per_block = 7
    rank = num_blocks * strassen_per_block  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # U rows: (i,j) -> i*m+j
    #   row 0: (0,0) A00
    #   row 1: (0,1) A01
    #   row 2: (1,0) A10
    #   row 3: (1,1) A11
    
    # V rows: (j,k) -> j*p+k
    #   j=0: rows 0-13 (B_{0,k})
    #   j=1: rows 14-27 (B_{1,k})
    
    # W rows: (k,i) -> k*n+i
    #   i=0: rows 0,2,4,...,26 (C_{0,k})
    #   i=1: rows 1,3,5,...,27 (C_{1,k})
    
    for t in range(num_blocks):
        k0 = 2 * t      # first column of block
        k1 = 2 * t + 1  # second column of block
        
        for s in range(strassen_per_block):
            col = t * strassen_per_block + s
            
            if s == 0:  # p1 = (A00+A11)(B_{0,k0}+B_{1,k1})
                U[0][col] = 1   # A00
                U[3][col] = 1   # A11
                V[0 * p + k0][col] = 1   # B_{0,k0}
                V[1 * p + k1][col] = 1   # B_{1,k1}
                W[k0 * n + 0][col] = 1   # C_{0,k0}
                W[k1 * n + 1][col] = 1   # C_{1,k1}
                
            elif s == 1:  # p2 = (A10+A11) * B_{0,k0}
                U[2][col] = 1   # A10
                U[3][col] = 1   # A11
                V[0 * p + k0][col] = 1   # B_{0,k0}
                W[k0 * n + 1][col] = 1   # C_{1,k0}
                W[k1 * n + 1][col] = -1  # C_{1,k1}
                
            elif s == 2:  # p3 = A00 * (B_{0,k1}-B_{1,k1})
                U[0][col] = 1   # A00
                V[0 * p + k1][col] = 1   # B_{0,k1}
                V[1 * p + k1][col] = -1  # B_{1,k1}
                W[k1 * n + 0][col] = 1   # C_{0,k1}
                W[k1 * n + 1][col] = 1   # C_{1,k1}
                
            elif s == 3:  # p4 = A11 * (B_{1,k0}-B_{0,k0})
                U[3][col] = 1   # A11
                V[1 * p + k0][col] = 1   # B_{1,k0}
                V[0 * p + k0][col] = -1  # B_{0,k0}
                W[k0 * n + 0][col] = 1   # C_{0,k0}
                W[k0 * n + 1][col] = 1   # C_{1,k0}
                
            elif s == 4:  # p5 = (A00+A01) * B_{1,k1}
                U[0][col] = 1   # A00
                U[1][col] = 1   # A01
                V[1 * p + k1][col] = 1   # B_{1,k1}
                W[k0 * n + 0][col] = -1  # C_{0,k0}
                W[k1 * n + 0][col] = 1   # C_{0,k1}
                
            elif s == 5:  # p6 = (A10-A00) * (B_{0,k0}+B_{0,k1})
                U[0][col] = -1  # A00
                U[2][col] = 1   # A10
                V[0 * p + k0][col] = 1   # B_{0,k0}
                V[0 * p + k1][col] = 1   # B_{0,k1}
                W[k1 * n + 1][col] = 1   # C_{1,k1}
                
            elif s == 6:  # p7 = (A01-A11) * (B_{1,k0}+B_{1,k1})
                U[1][col] = 1   # A01
                U[3][col] = -1  # A11
                V[1 * p + k0][col] = 1   # B_{1,k0}
                V[1 * p + k1][col] = 1   # B_{1,k1}
                W[k0 * n + 0][col] = 1   # C_{0,k0}
    
    # Verify correctness: check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k
    # Do a full exhaustive check (only 2*2*14 = 56 cases)
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank_fallback = n * m * p
        U_fb = [[0] * rank_fallback for _ in range(n * m)]
        V_fb = [[0] * rank_fallback for _ in range(m * p)]
        W_fb = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii*m+jj][col] = 1
                    V_fb[jj*p+kk][col] = 1
                    W_fb[kk*n+ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
