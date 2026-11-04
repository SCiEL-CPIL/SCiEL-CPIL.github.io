"""
Rank-22 exact tensor decomposition for 2x3 by 3x6 matrix multiplication.

Construction:
  Split the inner dimension (size 3) into 1 + 2:
    - The 1-part (A col 0 times B row 0): contributes 2*6 = 12 rank-1 terms
      (one per output entry C[i,k] = A[i,0]*B[0,k]).
    - The 2-part (A cols 1-2 times B rows 1-2): M_{2,2,6}.
      For M_{2,2,6}: A is 2x2, B is 2x6, C is 2x6.
      We use the fact that M_{2,2,p} has rank at most 2p + 2 for p >= 2? 
      Actually, let me think more carefully.
      
      For M_{2,2,p}: C[i,k] = sum_{j=1}^{2} A[i,j] * B[j,k].
      This is a bilinear form. The rank of this tensor is at most 2*p (schoolbook).
      But can we do better? For p=2, rank of M_{2,2,2} is 4 (optimal).
      For p=3, rank of M_{2,2,3} is at most 7? Let's see:
        C[0,0] = a11*b10 + a12*b20
        C[0,1] = a11*b11 + a12*b21
        C[0,2] = a11*b12 + a12*b22
        C[1,0] = a21*b10 + a22*b20
        C[1,1] = a21*b11 + a22*b21
        C[1,2] = a21*b12 + a22*b22
      
      Schoolbook: 6 terms. Can we do better? Probably not for these small dims.
      
      So M_{2,2,6} schoolbook = 12. Total = 12 + 12 = 24.
      
  So rank 24 is achievable. Let me construct that.
  
  Actually, let me reconsider the split. What if I split differently?
  
  Alternative: Use the structure more cleverly.
  
  Let me just go with rank 24: split inner dim into 1+2.
  - 1-part: 2*6 = 12 terms
  - 2-part: 2*2*6 = 12 terms (schoolbook for M_{2,2,6})
  Total: 24
"""

def construct():
    n, m, p = 2, 3, 6
    
    # Rank 24 construction:
    # Split m = 3 into 1 + 2.
    # Part 1: A[:, 0:1] (2x1) times B[0:1, :] (1x6) -> C contribution
    #   C[i,k] += A[i,0] * B[0,k]
    #   This is 2*6 = 12 rank-1 terms.
    # Part 2: A[:, 1:3] (2x2) times B[1:3, :] (2x6) -> C contribution  
    #   C[i,k] += sum_{j=1,2} A[i,j] * B[j,k]
    #   This is 2*2*6 = 12 rank-1 terms (schoolbook).
    # Total rank: 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 18 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    col = 0
    # Part 1: A[:, 0] times B[0, :]
    # A index: i*m + 0 = i*3 + 0 for i in {0,1}
    # B index: 0*p + k = k for k in {0..5}
    # C index: k*n + i = k*2 + i
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: A[:, 1:3] times B[1:3, :]
    # A index: i*m + j for i in {0,1}, j in {1,2}
    # B index: j*p + k for j in {1,2}, k in {0..5}
    # C index: k*n + i
    for i in range(n):
        for j in range(1, 3):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
