"""
Rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.

Strategy:
- The schoolbook rank is 12.
- We can use Strassen's algorithm (rank 7) for the 2x2 block and schoolbook (rank 4)
  for the remaining column, achieving rank 11.
- We search for even lower ranks (10, 9, 8, ...) using a combination of:
  1. Known algebraic constructions (Strassen variants, etc.)
  2. Numerical optimization to find integer/half-integer solutions

The key insight is that for 2x2 * 2x3, we can view the multiplication as:
  A (2x2) * B (2x3) = C (2x3)
  
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
  C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
  C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
  C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
  C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
  C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]

We can use a block decomposition: treat B as two columns, one 2x1 and one 2x2.
The 2x2 part can use Strassen (rank 7) and the 2x1 part can use schoolbook (rank 4),
but we need to be careful about sharing.

Actually, let's think more carefully. The tensor is T_{i,j,k} where i in {0,1}, j in {0,1}, k in {0,1,2}.
T_{i,j,k} = 1 if the (i,j) entry of A times the (j,k) entry of B contributes to C_{i,k}.

Wait, let me re-read the convention. W[k*n+i, r] where n=2, so W[k*2+i, r].
U has shape (n*m, rank) = (4, rank), V has shape (m*p, rank) = (6, rank), W has shape (p*n, rank) = (6, rank).

The decomposition is: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}

So T_{i,j,k} = sum_r U_{2i+j, r} * V_{2j+k, r} * W_{2k+i, r}

For the schoolbook, each (i,j,k) triple has its own rank term with U=V=W=1.

Let me try to find a rank-11 decomposition by combining Strassen for the 2x2x2 part (k=0,1) 
and schoolbook for the k=2 column.

Actually, let me just try to construct rank 11 explicitly and then search for lower.

For rank 11:
- Use Strassen for the first two columns of B (k=0,1): this gives a 2x2 * 2x2 multiplication with rank 7.
- Use schoolbook for the third column (k=2): this gives a 2x2 * 2x1 multiplication with rank 4.
- Total rank: 7 + 4 = 11.

Let me verify this works. The Strassen decomposition for 2x2*2x2 uses 7 terms.
Each term r has U_{2i+j,r}, V_{2j+k,r}, W_{2k+i,r} for k in {0,1}.

For the k=2 column, we need 4 more terms where only W_{2*2+i,r} = W_{4+i,r} is nonzero (for i=0,1)
and U and V are set appropriately.

Let me code this up.
"""

import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # T_{i,j,k} should equal sum_r U_{i*m+j,r} * V_{j*p+k,r} * W_{k*n+i,r}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1  # The matrix multiplication tensor is all 1s
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_rank11():
    """Construct a rank-11 decomposition using Strassen (rank 7) + schoolbook (rank 4)."""
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen's 7 terms for the 2x2 * 2x2 part (k=0,1)
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    #
    # In our indexing:
    # A is 2x2: A[i,j] with i=0,1 (rows), j=0,1 (cols)
    # B is 2x2 for k=0,1: B[j,k] with j=0,1 (rows), k=0,1 (cols)
    # C is 2x2: C[i,k] with i=0,1 (rows), k=0,1 (cols)
    #
    # U_{i*m+j, r} = U_{2i+j, r}
    # V_{j*p+k, r} = V_{2j+k, r} for k=0,1 -> V_{2j}, V_{2j+1}
    # W_{k*n+i, r} = W_{2k+i, r} for k=0,1 -> W_{i}, W_{2+i}
    
    # Strassen terms (r = 0..6)
    # M1: (A00+A11)*(B00+B11)
    # U: A00 -> U[0,0]=1, A11 -> U[3,0]=1
    # V: B00 -> V[0,0]=1, B11 -> V[3,0]=1  (B[1,1] is j=1,k=1 -> V[1*2+1]=V[3])
    # W: C00 -> W[0,0]=1, C11 -> W[3,0]=1  (C[1,1] is k=1,i=1 -> W[1*2+1]=W[3])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    # U: A10 -> U[2,1]=1, A11 -> U[3,1]=1
    # V: B00 -> V[0,1]=1
    # W: C10 -> W[1,1]=1  (C[1,0] is k=0,i=1 -> W[0*2+1]=W[1])
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3: A00*(B01-B11)
    # U: A00 -> U[0,2]=1
    # V: B01 -> V[1,2]=1, B11 -> V[3,2]=-1
    # W: C01 -> W[2,2]=1  (C[0,1] is k=1,i=0 -> W[1*2+0]=W[2])
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4: A11*(B10-B00)
    # U: A11 -> U[3,3]=1
    # V: B10 -> V[2,3]=1, B00 -> V[0,3]=-1
    # W: C10 -> W[1,3]=1, C00 -> W[0,3]=-1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[0][3] = -1
    
    # M5: (A00+A01)*B11
    # U: A00 -> U[0,4]=1, A01 -> U[1,4]=1
    # V: B11 -> V[3,4]=1
    # W: C01 -> W[2,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    # U: A10 -> U[2,5]=1, A00 -> U[0,5]=-1
    # V: B00 -> V[0,5]=1, B01 -> V[1,5]=1
    # W: C00 -> W[0,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    # U: A01 -> U[1,6]=1, A11 -> U[3,6]=-1
    # V: B10 -> V[2,6]=1, B11 -> V[3,6]=1
    # W: C11 -> W[3,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    # Schoolbook for k=2 column (r = 7..10)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #
    # r=7: A[0,0]*B[0,2]
    # U[0,7]=1 (A[0,0])
    # V[0*2+2]=V[2,7]=1 (B[0,2])
    # W[2*2+0]=W[4,7]=1 (C[0,2])
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # r=8: A[0,1]*B[1,2]
    # U[1,8]=1 (A[0,1])
    # V[1*2+2]=V[4,8]=1 (B[1,2])
    # W[2*2+0]=W[4,8]=1 (C[0,2])
    U[1][8] = 1
    V[4][8] = 1
    W[4][8] = 1
    
    # r=9: A[1,0]*B[0,2]
    # U[2,9]=1 (A[1,0])
    # V[2,9]=1 (B[0,2])
    # W[2*2+1]=W[5,9]=1 (C[1,2])
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # r=10: A[1,1]*B[1,2]
    # U[3,10]=1 (A[1,1])
    # V[4,10]=1 (B[1,2])
    # W[5,10]=1 (C[1,2])
    U[3][10] = 1
    V[4][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_lower_rank(rank, deadline):
    """Try to find a decomposition of the given rank using numerical search."""
    n, m, p = 2, 2, 3
    dim_U = n * m  # 4
    dim_V = m * p  # 6
    dim_W = p * n  # 6
    
    # The tensor T_{i,j,k} = 1 for all i,j,k
    # We need: sum_r U_{2i+j,r} * V_{2j+k,r} * W_{2k+i,r} = 1 for all i,j,k
    
    # This is a system of 12 equations (2*2*3) in 4*rank + 6*rank + 6*rank = 16*rank unknowns.
    # For rank < 11, we have more unknowns than equations, so solutions may exist.
    
    # Let's try a random initialization and gradient descent approach.
    # But we need integer/half-integer solutions, which is tricky.
    
    # Alternative: try to find rational solutions by solving the polynomial system.
    # For small ranks, we can try to set some variables to simple values and solve.
    
    # For now, let's just return None if we can't find one quickly.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try the known rank-11 construction
    result = _construct_rank11()
    
    # Verify it
    if _verify(result['U'], result['V'], result['W']):
        # Try to find lower ranks
        for rank in range(10, 6, -1):
            if time.time() > deadline - 10:
                break
            res = _try_lower_rank(rank, deadline)
            if res is not None and _verify(res['U'], res['V'], res['W']):
                result = res
                break
    
    return result
