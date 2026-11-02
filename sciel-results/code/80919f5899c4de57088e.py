"""
Matrix multiplication 2x2 * 2x15 tensor decomposition.
Uses the known rank-9 decomposition for 2x2 * 2x3 (Bordellas-Strassen)
with overlapping windows to share intermediate computations.

Strategy:
- 15 columns can be covered by 7 overlapping 3-column windows:
  [0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,10], [10,11,12], [12,13,14]
- Each window uses a rank-9 decomposition of M_{2,2,3}
- Total rank: 7 * 9 = 63... wait, that's worse than 60.

Let me reconsider. The schoolbook rank is 2*2*15 = 60.

Actually, for M_{2,2,p}, we can use:
- If p is even: p/2 * 7 (Strassen blocks) 
- If p is odd: floor(p/2) * 7 + 4 (one 2x2 Strassen + one 2x1 schoolbook)

For p=15: 7*7 + 4 = 53? No wait.
- 15 = 7*2 + 1
- 7 blocks of 2x2 * 2x2, each rank 7 (Strassen)
- 1 block of 2x2 * 2x1, each rank 4 (schoolbook)
- Total: 7*7 + 4 = 53

But I need correct Strassen formulas. Let me use the verified ones.

Actually, let me just use the schoolbook decomposition which is guaranteed correct
and achieves rank 60. Then try to improve with a simpler tiling approach.

For 2x2 * 2x1: C[i,0] = sum_j A[i,j] B[j,0], which is 2*2 = 4 schoolbook terms.
For 2x2 * 2x2: 2*2*2 = 8 schoolbook terms, but Strassen gives 7.

Let me implement Strassen correctly using the standard formulation:
A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB
C[0,0] = ae + bg
C[0,1] = af + bh  
C[1,0] = ce + dg
C[1,1] = cf + dh

Standard Strassen (verified):
M1 = (a+d)(e+h) = ae+ah+de+dh
M2 = (c+d)e = ce+de
M3 = a(f-h) = af-ah
M4 = d(g-e) = dg-de
M5 = (a+c)h = ah+ch
M6 = (d-b)(e+f) = de+df-be-bf
M7 = (b-c)(g+h) = bg+bh-cg-ch

C[0,0] = M1+M3-M4-M6 = ae+ah+de+dh + af-ah -dg+de -de-df+be+bf
       = ae+dh+af-dg+de-df+be+bf
This should equal ae+bg. Let me check: ae+dh+af-dg+de-df+be+bf vs ae+bg.
These are not equal in general. So this formula set is wrong.

Let me try another standard form. After research, the correct Strassen is:
M1 = (a+d)(e+h)
M2 = (c+d)(e)
M3 = a(f-h)
M4 = d(g-e)
M5 = (a+c)h
M6 = (d-b)(e+f)
M7 = (b-c)(g+h)

C00 = M1+M3-M4-M6
C01 = M4+M5
C10 = M2+M6
C11 = M1-M3+M5-M7

Let me verify with a=1,b=2,c=3,d=4, e=5,f=6,g=7,h=8:
C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

M1 = (1+4)(5+8) = 5*13 = 65
M2 = (3+4)*5 = 35
M3 = 1*(6-8) = -2
M4 = 4*(7-5) = 8
M5 = (1+3)*8 = 32
M6 = (4-2)*(5+6) = 2*11 = 22
M7 = (2-3)*(7+8) = -1*15 = -15

C00 = 65+(-2)-8-22 = 33. Should be 19. WRONG.

The formulas I keep using are incorrect. Let me derive from scratch or use a known correct implementation.

Given the difficulty, let me just use the schoolbook decomposition with rank 60,
which is definitely correct.

For a better approach, let me use the fact that for M_{2,2,3}, there exists a rank-10
decomposition (or possibly lower). But without a verified formula, I'll stick with schoolbook.

Actually, let me try a different tiling: use 5x3 blocks.
15 columns = 5 blocks of 3 columns.
For each 2x2 * 2x3 block, schoolbook rank = 2*2*3 = 12.
Total: 5*12 = 60. Same as before.

Or use 2x2 Strassen if I can get it right. Let me try once more with a different source.

After more thought, I'll implement the schoolbook decomposition correctly and return rank 60.
"""

import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # A is n x m = 2x2, B is m x p = 2x15, C is n x p = 2x15
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Tensor decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. The standard form is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * ... 
    # 
    # Actually the contract says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # The decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = ... 
    # 
    # No, the standard tensor decomposition for matrix multiplication is:
    # T[i,j,k] = sum_r U[i,j,r] * V[k,r] * ... 
    # 
    # Let me re-read the contract:
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "output factor follows the official trace convention W[k*n+i,r]"
    #
    # So the tensor is T[i,j,k] where i indexes A's rows (0..n-1), 
    # j indexes A's cols / B's rows (0..m-1), k indexes B's cols (0..p-1).
    # T[i,j,k] = delta (this is the matrix multiplication tensor, which is 1 
    # when we're computing C[i,k] = sum_j A[i,j]B[j,k])
    #
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't make dimensional sense for the tensor.
    #
    # Actually, for matrix multiplication tensor, the standard decomposition is:
    # sum_r U[i,j,r] * V[k,r] * ... no.
    #
    # Let me think about this differently. The matrix multiplication tensor T is 
    # a 3-tensor of size n*m*p where T[i,j,k] = 1 (it's the "multiplication" tensor).
    # A rank-r decomposition is T[i,j,k] = sum_{r=1}^R u_r[i,j] * v_r[j,k] * w_r[k,i]
    # where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}.
    #
    # So U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R).
    # And the condition is: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For the matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k (since 
    # C[i,k] = sum_j A[i,j] B[j,k] means each (i,j,k) triple contributes).
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # Schoolbook: for each (i,j,k), set U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, 
    # W[k*n+i, (i,j,k)] = 1. Then the product is 1 for that (i,j,k) and 0 for others.
    # Total rank = n*m*p = 2*2*15 = 60.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
