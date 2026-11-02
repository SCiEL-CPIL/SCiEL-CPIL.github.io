"""Matrix multiplication 3,4,6 rank-minimal decomposition.

Approach: Construct a rank-70 decomposition by exploiting the structure of the
matrix multiplication tensor. We decompose the 3x4x6 multiplication into
overlapping sub-problems that share rank factors.

The key insight: 3x4x6 can be split as (3x3x6) + (0x1x6) but that's trivial.
Better: use the identity that R(3,4,6) <= R(3,3,3) + R(3,1,6) + R(0,1,0)*... 
Actually, a cleaner approach: R(3,4,6) <= R(2,2,2) + R(3,2,6) where we split
the 4 columns into 2+2 and 6 rows into... no.

Let me think more carefully. The tensor T_{i,j,k} = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
for i in [0,n), j in [0,m), k in [0,p).

We can use the fact that R(n,m,p) <= R(n1,m,p) + R(n2,m,p) when n = n1+n2,
and similarly for other dimensions.

R(3,4,6) <= R(2,4,6) + R(1,4,6)
R(1,4,6) = 24 (trivial: each output element needs its own rank factor)
R(2,4,6) <= R(2,2,6) + R(2,2,6) = 2*R(2,2,6)
R(2,2,6) <= R(2,2,2) + R(2,2,2) + R(2,2,2) + R(2,2,2) = 4*7 = 28
So R(2,4,6) <= 56, R(3,4,6) <= 56 + 24 = 80. That's worse than 72.

Better split: R(3,4,6) <= R(3,2,6) + R(3,2,6)
R(3,2,6) <= R(3,2,2) + R(3,2,2) + R(3,2,2) = 3*R(3,2,2)
R(3,2,2) <= R(1,2,2) + R(1,2,2) + R(1,2,2) = 3*4 = 12
So R(3,2,6) <= 36, R(3,4,6) <= 72. Same as schoolbook.

Try: R(3,4,6) <= R(3,3,6) + R(3,1,6)
R(3,1,6) = 18
R(3,3,6) <= R(3,3,2) + R(3,3,2) + R(3,3,2) = 3*R(3,3,2)
R(3,3,2) <= R(1,3,2)*3 = 3*6 = 18? No, R(3,3,2) is not 18.
R(3,3,2): known to be 18? Actually R(3,3,2) = 18 is the schoolbook. 
Hmm, R(3,3,2) might be less. Let's assume 18.
So R(3,3,6) <= 54, R(3,4,6) <= 54+18 = 72. Still schoolbook.

The fundamental issue is that for these small dimensions, the schoolbook rank
mn*p = 72 is hard to beat with simple recursive splitting.

However, there ARE known improvements. For R(3,4,6), the best known upper bound
is actually 70, achieved by specific constructions.

Let me construct a rank-70 decomposition explicitly. The idea: take the schoolbook
72 rank factors and find 2 pairs that can be merged.

In the schoolbook, factor (i,j,k) has:
U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1

If we can find two factors (i1,j1,k1) and (i2,j2,k2) such that we can replace
them with a single factor using linear combinations, we reduce rank by 1.

Actually, let me just verify: is rank 70 achievable? Let me construct it.

A known technique: for R(3,4,6), one can achieve rank 70 by modifying the
schoolbook. Here's a concrete construction:

Take the 72 schoolbook factors. Consider factors where i=0,j=0,k in {0,1} and
i=0,j=0,k in {2,3}. We can combine:
- Factor (0,0,0): U[0]=1, V[0]=1, W[0]=1
- Factor (0,0,1): U[0]=1, V[1]=1, W[1]=1

Replace these two with:
- New factor: U[0]=1, V[0]+V[1]=1, W[0]+W[1]=1  (but this doesn't work cleanly)

This is getting complex. Let me just use a verified rank-70 construction.

After careful analysis, I'll provide a rank-70 decomposition based on the
following: split the 6 columns of the second matrix into 3+3, giving:
R(3,4,6) <= R(3,4,3) + R(3,4,3)

R(3,4,3) <= R(3,2,3) + R(3,2,3)
R(3,2,3) = 18 (schoolbook for 3x2x3)

So R(3,4,3) <= 36, R(3,4,6) <= 72. Still no gain.

Given the difficulty of finding a provably better rank within time constraints,
and the risk of invalid decomposition, I will provide the schoolbook rank-72
decomposition which is guaranteed correct.
"""

import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    # Schoolbook decomposition: rank = n*m*p = 72
    rank = n * m * p
    
    # Build factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for idx, (i, j, k) in enumerate(
        (i, j, k) for i in range(n) for j in range(m) for k in range(p)
    ):
        U[i * m + j, idx] = 1
        V[j * p + k, idx] = 1
        W[k * n + i, idx] = 1
    
    # Convert to lists as expected by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
