"""
Rank-11 exact tensor decomposition for 2x2 by 2x3 matrix multiplication.

We exploit the structure: the 2x2-by-2x3 product C (2x3) can be seen as:
  C[:, 0:2] = A @ B[:, 0:2]  (2x2 * 2x2, rank 7 via Strassen)
  C[:, 2]   = A @ B[:, 2]    (2x2 * 2x1, rank 4 direct)
Total rank = 7 + 4 = 11.

Tensor index convention:
  A is n x m = 2 x 2, indexed by (i, j) with i in [0,1], j in [0,1]
  B is m x p = 2 x 3, indexed by (j, k) with j in [0,1], k in [0,2]
  C is p x n = 3 x 2, indexed by (k, i) with k in [0,2], i in [0,1]
  W[k*n + i, r]

For each rank term r, we have:
  U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] summed over r gives C[k*n+i]

Strassen for 2x2*2x2 (columns 0,1 of B):
  Let A = [[a00, a01], [a10, a11]], B2 = [[b00, b01], [b10, b11]]
  C2 = A @ B2 = [[c00, c01], [c10, c11]]
  
  Strassen terms:
  M1 = (a00 + a11)(b00 + b11)
  M2 = (a10 + a11) * b01
  M3 = a00 * (b10 - b11)
  M4 = a11 * (b00 - b10)
  M5 = (a00 + a01) * b10
  M6 = (a01 - a11)(b10 + b11)
  M7 = (a10 - a00)(b00 + b01)
  
  c00 = M1 + M4 - M5 + M7
  c01 = M3 + M5
  c10 = M2 + M4
  c11 = M1 - M2 + M3 + M6

Direct for column 2 of B (b02, b12):
  c02 = a00*b02 + a01*b12
  c12 = a10*b02 + a11*b12
  These are 4 rank-1 terms.

Total: 7 + 4 = 11 terms.

We construct U (4x11), V (6x11), W (6x11) accordingly.
"""
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # A[i,j] -> U index i*m + j = i*2 + j
    # B[j,k] -> V index j*p + k = j*3 + k
    # C[k,i] -> W index k*n + i = k*2 + i
    
    # Strassen terms (r = 0..6) for columns k=0,1
    # M1 = (a00+a11)(b00+b11)
    #   U: a00=1, a11=1 at row 0*2+0=0 and 1*2+1=3
    #   V: b00=1, b11=1 at row 0*3+0=0 and 1*3+1=4
    #   W: contributes to c00 (k=0,i=0 -> idx 0), c11 (k=1,i=1 -> idx 3)
    #   c00 += M1, c11 += M1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[4, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0  # c00 and c11
    
    # M2 = (a10+a11) * b01
    #   U: a10=1 (row 1*2+0=2), a11=1 (row 3)
    #   V: b01=1 (row 0*3+1=1)
    #   W: c10 (k=0,i=1 -> idx 1)
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[1, 1] = 1.0
    W[1, 1] = 1.0  # c10
    
    # M3 = a00 * (b10 - b11)
    #   U: a00=1 (row 0)
    #   V: b10=1 (row 1*3+0=3), b11=-1 (row 4)
    #   W: c01 (k=1,i=0 -> idx 2)
    U[0, 2] = 1.0
    V[3, 2] = 1.0; V[4, 2] = -1.0
    W[2, 2] = 1.0  # c01
    
    # M4 = a11 * (b00 - b10)
    #   U: a11=1 (row 3)
    #   V: b00=1 (row 0), b10=-1 (row 3)
    #   W: c00 (idx 0), c10 (idx 1)
    U[3, 3] = 1.0
    V[0, 3] = 1.0; V[3, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0  # c00 and c10
    
    # M5 = (a00+a01) * b10
    #   U: a00=1 (row 0), a01=1 (row 1)
    #   V: b10=1 (row 3)
    #   W: c00 (idx 0) with -1, c01 (idx 2) with +1
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0  # -c00, +c01
    
    # M6 = (a01-a11)(b10+b11)
    #   U: a01=1 (row 1), a11=-1 (row 3)
    #   V: b10=1 (row 3), b11=1 (row 4)
    #   W: c11 (idx 3)
    U[1, 5] = 1.0; U[3, 5] = -1.0
    V[3, 5] = 1.0; V[4, 5] = 1.0
    W[3, 5] = 1.0  # c11
    
    # M7 = (a10-a00)(b00+b01)
    #   U: a10=1 (row 2), a00=-1 (row 0)
    #   V: b00=1 (row 0), b01=1 (row 1)
    #   W: c00 (idx 0)
    U[2, 6] = 1.0; U[0, 6] = -1.0
    V[0, 6] = 1.0; V[1, 6] = 1.0
    W[0, 6] = 1.0  # c00
    
    # Direct terms for column k=2 (r = 7..10)
    # c02 = a00*b02 + a01*b12 -> two terms
    # c12 = a10*b02 + a11*b12 -> two terms
    
    # Term 7: a00 * b02 -> c02 (k=2, i=0 -> idx 2*2+0=4)
    U[0, 7] = 1.0
    V[0*3+2, 7] = 1.0  # b02, row 2
    W[2*2+0, 7] = 1.0  # c02, idx 4
    
    # Term 8: a01 * b12 -> c02
    U[1, 8] = 1.0
    V[1*3+2, 8] = 1.0  # b12, row 5
    W[2*2+0, 8] = 1.0  # c02, idx 4
    
    # Term 9: a10 * b02 -> c12 (k=2, i=1 -> idx 2*2+1=5)
    U[2, 9] = 1.0
    V[0*3+2, 9] = 1.0  # b02, row 2
    W[2*2+1, 9] = 1.0  # c12, idx 5
    
    # Term 10: a11 * b12 -> c12
    U[3, 10] = 1.0
    V[1*3+2, 10] = 1.0  # b12, row 5
    W[2*2+1, 10] = 1.0  # c12, idx 5
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
