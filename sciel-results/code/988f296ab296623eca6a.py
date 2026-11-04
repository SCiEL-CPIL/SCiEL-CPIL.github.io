"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Reconstruction convention:
  C[i,k] = sum_c [sum_j U[i*m+j, c] * A[i,j]] * [sum_j V[j*p+k, c] * B[j,k]] * W[k*n+i, c]

Schoolbook: rank = n*m*p = 56, with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for c=(i,j,k).

Improvement: Exploit n=2 structure. Write A[1,j] = A[0,j] + D[j].
Then C[1,k] = C[0,k] + sum_j D[j]*B[j,k].

For each j, the product D[j]*B[j,k] = (A[1,j]-A[0,j])*B[j,k] = A[1,j]*B[j,k] - A[0,j]*B[j,k].
This means the "difference" part is already captured by the individual products.

The key insight: we can share the B[j,k] factors between the two rows.
For each j and k, we need:
  C[0,k] += A[0,j]*B[j,k]
  C[1,k] += A[1,j]*B[j,k]

Instead of 2 separate products per (j,k), we can use:
  P1 = A[0,j]*B[j,k]
  P2 = A[1,j]*B[j,k]
  P3 = (A[0,j]+A[1,j])*B[j,k]  (Strassen-like)
  
Then A[0,j]*B[j,k] = (P1 + P3 - P2)/2... no, that doesn't help.

Actually, the standard approach for [2,m,p]: 
  For each j, we compute A[0,j]*B[j,k] and A[1,j]*B[j,k] for all k.
  That's 2*p products per j, total 2*m*p = 56.

To reduce: use the fact that for fixed j, the map (A[0,j], A[1,j]) -> [A[0,j]*B[j,k]]_{k}
is a rank-1 map in A (scalar times B[j,:]). The tensor rank contribution per j is p (for each k).
Total: m*p*2 = 56.

Can we do better? The lower bound for [2,4,7] is at least max(2*4, 4*7, 2*7) = max(8,28,14) = 28.
But that's just a trivial bound. 

Known result: rank([2,4,7]) <= 2*4*7 = 56 (schoolbook).
With Strassen-like tricks on the n=2 dimension:
  Let S1 = A[0,:], S2 = A[1,:], D = A[1,:] - A[0,:]
  C[0,:] = S1 @ B
  C[1,:] = C[0,:] + D @ B
  
  So we need: S1 @ B (rank <= 4*7=28) and D @ B (rank <= 4*7=28).
  Total rank <= 56, same as schoolbook. No improvement.

However, we can use a different decomposition. Consider that the full tensor is:
  T[i,j,k] for the matmul, but structured as bilinear in (A,B).

Let me try a concrete rank-50 construction using partial sharing.

Actually, let me just verify the schoolbook works and return it. The task says score is rank (lower is better),
and schoolbook gives 56. If I can't find a valid smaller rank, I'll stick with 56.

Let me try rank 52: remove 4 terms by exploiting linearity.

For each k, C[0,k] = sum_j A[0,j]B[j,k] and C[1,k] = sum_j A[1,j]B[j,k].
The 14 output entries each need 4 products = 56 total.

A rank-1 term contributes to potentially multiple output entries if W has multiple nonzero entries.
In schoolbook, each term contributes to exactly one output entry (W has a single 1).

If we can make some W vectors have two nonzeros (e.g., W[k*n+0,c]=1 and W[k*n+1,c]=1),
then that term contributes to both C[0,k] and C[1,k]. But the U and V parts would need to 
produce the right linear forms.

For term c contributing to both C[0,k] and C[1,k]:
  U part: must be a linear form in A that, when multiplied with V part (linear form in B),
  gives a contribution to both C[0,k] and C[1,k].
  
  If U[i*m+j,c] is nonzero for both i=0 and i=1, and V[j*p+k,c] is nonzero,
  then the contribution to C[i,k] is (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c].

  For this to equal A[0,j']*B[j',k] for C[0,k] and A[1,j'']*B[j'',k] for C[1,k] simultaneously,
  we'd need the U part to factor as A[0,j']*A[1,j'']... which is not linear. So this doesn't work.

I'll stick with schoolbook rank 56, which is guaranteed valid.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result
