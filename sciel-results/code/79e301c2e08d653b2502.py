"""
Exact rank-12 tensor decomposition for 2x3 by 3x6 matrix multiplication.

We exploit the fact that the multiplication tensor T_{i,j,k} = 1 for all (i,j,k)
can be decomposed as sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1.

The construction:
- For each i (2 values), we have a 3x6 sub-problem (j x k) where we need
  sum_j U[i,j,r] * V[j,k,r] = 1 for all k.
- The W factor must satisfy W[k,i,r] such that the product works.

Actually, a cleaner approach: use the identity
  sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1

We can set W[k,i,r] = 1 for all (k,i,r) in the support, and then we need
  sum_r U[i,j,r] * V[j,k,r] = 1 for all (i,j,k).

But that's not quite right since W depends on r too.

Let me use a different approach. The key insight is:
  T_{i,j,k} = 1 = sum_r (1/3) * 3 * (1/6) * 6 * ... no.

Actually, the simplest non-trivial decomposition:
Note that 1 = sum_{j=0}^{2} (1/3) * 3 = ... no.

Let me think differently. We need rank r such that there exist U (6xr), V (18xr), W (12xr)
with sum_r U[i*3+j,r]*V[j*6+k,r]*W[k*2+i,r] = 1 for all i,j,k.

One valid decomposition of rank 12:
For each i in {0,1}, for each k in {0,...,5}, we assign one rank term.
In that term r = i*6+k:
  U[i*3+j, r] = 1/3 for all j in {0,1,2}
  V[j*6+k, r] = 1 for all j in {0,1,2}  
  W[k*2+i, r] = 1

Then sum_r U[i*3+j,r]*V[j*6+k,r]*W[k*2+i,r] = (1/3)*1*1 = 1/3, not 1.

So we need U[i*3+j,r] = 1 for all j. Then sum = 1*1*1 = 1. But that's rank 12 with:
  U[i*3+j, i*6+k] = 1 for all j
  V[j*6+k, i*6+k] = 1 for all j
  W[k*2+i, i*6+k] = 1

Let's verify: for fixed (i,j,k), only r = i*6+k has non-zero U[i*3+j,r]=1, V[j*6+k,r]=1, W[k*2+i,r]=1.
So the sum is 1. 

But wait, V[j*6+k, r] where r = i'*6+k'. For r = i*6+k, V[j*6+k, i*6+k] = 1 only if k matches. Yes.

So rank 12 works with all entries being 0 or 1. Let me double-check there are no conflicts.

U has 6 rows (i,j) and 12 cols (i',k'). U[i*3+j, i'*6+k'] = 1 if i'=i, else 0.
V has 18 rows (j,k) and 12 cols (i',k'). V[j*6+k, i'*6+k'] = 1 if k=k', else 0.
W has 12 rows (k,i) and 12 cols (i',k'). W[k*2+i, i'*6+k'] = 1 if k=k' and i=i', else 0.

For a given (i,j,k), the only nonzero term in the sum over r is r = i*6+k:
U[i*3+j, i*6+k] = 1 (since i'=i)
V[j*6+k, i*6+k] = 1 (since k'=k)
W[k*2+i, i*6+k] = 1 (since k'=k and i'=i)
Product = 1. All other r give 0. Sum = 1. Correct!

This gives rank 12. Can we do better? The minimum rank for 2x3x6 matmul... 
The trivial bound is n*m*p = 36. This construction gives 12 = n*p = 2*6.
Actually it's n*p = 12. Since m=3, we could potentially do n*m = 6 or m*p = 18, but 12 seems good.

Let me try rank 6: assign one term per (i,j) pair.
r = i*3+j. U[i*3+j, r] = 1. V[j*6+k, r] = 1 for all k. W[k*2+i, r] = 1/6 for all k.
Then sum = 1 * 1 * (1/6) = 1/6. Not 1.

If W[k*2+i, r] = 1, then sum over k of contributions... no, for fixed (i,j,k) only one r matters.
U[i*3+j, i*3+j] = 1, V[j*6+k, i*3+j] = 1, W[k*2+i, i*3+j] = 1. Sum = 1. But V[j*6+k, r] = 1 for all k means V row j*6+k has 1s in columns r where j matches. That's 2 columns (i=0 and i=1 for that j). So V[j*6+k, i*3+j] = 1 for both i values. 

Wait, r = i'*3+j'. For fixed j and k, V[j*6+k, r] is nonzero when j'=j. So V[j*6+k, i'*3+j] = 1 for i'=0,1.
Then for (i,j,k): sum_r U[i*3+j,r]*V[j*6+k,r]*W[k*2+i,r].
Nonzero when j'=j, so r in {0*3+j, 1*3+j} = {j, 3+j}.
U[i*3+j, j] = 1 if i=0, else 0. U[i*3+j, 3+j] = 1 if i=1, else 0.
So only one of these is nonzero, giving U=1, V=1, W[k*2+i, i*3+j].
We need W[k*2+i, i*3+j] = 1. So W[k*2+i, r] = 1 when i matches r's i-component.
W has 12 rows (k,i) and 6 cols (i',j'). W[k*2+i, i'*3+j'] = 1 if i=i', else 0.

This works! Rank 6. Let me verify:
For (i,j,k): r must satisfy j'=j (from V) and i'=i (from U). So r = i*3+j.
U[i*3+j, i*3+j] = 1. V[j*6+k, i*3+j] = 1. W[k*2+i, i*3+j] = 1. Sum = 1. 

Rank 6 is valid. Can we go lower? Rank 5? Probably not by symmetry arguments, but let's use rank 6.

Actually, can we get rank 4 or less? The tensor has "slice rank" considerations. For now, rank 6 is a solid improvement over 36.
"""

import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = 2, 3, 6
    rank = n * m  # 6
    
    U = [[0.0] * rank for _ in range(n * m)]   # 6 x 6
    V = [[0.0] * rank for _ in range(m * p)]   # 18 x 6
    W = [[0.0] * rank for _ in range(p * n)]   # 12 x 6
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U[i*m+j, r] = 1
            U[i * m + j][r] = 1.0
            # V[j*p+k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1.0
            # W[k*n+i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
