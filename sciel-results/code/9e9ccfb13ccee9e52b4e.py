"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Uses the n=2 structure: treat A = [A0; A1] where A0, A1 are 1x4 rows.
C0 = A0 @ B (1x7), C1 = A1 @ B (1x7).

Strategy: Use the identity C1 = C0 + D @ B where D = A1 - A0.
This means we need:
  - A decomposition for C0 = A0 @ B  (bilinear in A0 and B)
  - A decomposition for E = D @ B    (bilinear in D and B, where D is linear in A)
  - C1 = C0 + E (linear combination, absorbed into W factors)

The key insight: the tensor for [2,4,7] can be decomposed by:
  1. Computing A0 @ B using a [1,4,7] decomposition (schoolbook rank 28)
  2. Computing D @ B using another [1,4,7] decomposition (schoolbook rank 28)
  3. The addition C1 = C0 + E is free (just modify W entries)

Total rank = 28 + 28 = 56... same as schoolbook. No gain.

Better approach: Use the fact that for [1,4,7], we can potentially use fewer than 28
if we exploit structure. But [1,4,7] is essentially a dot product of length 4 with
each of 7 columns, so the tensor rank is exactly 4*7 = 28 (each output C[k] = sum_j A[j]*B[j,k]
is an independent bilinear form, and they don't share structure because B's columns are independent).

So for [2,4,7], the schoolbook rank 56 might actually be optimal or near-optimal.

Let me try a different angle: can we combine some of the 56 terms?

For the full [2,4,7] tensor, each rank-1 term contributes to the trilinear form.
The tensor has 2*4*7 = 56 "basic" products in schoolbook. 

Actually, let me reconsider. The tensor rank of [2,4,7] matrix multiplication:
- Lower bound: max(2*4, 4*7, 7*2) = max(8, 28, 14) = 28? No, that's not the right lower bound.
- The trivial lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
- Upper bound: n*m*p = 56 (schoolbook).

Can we achieve rank between 28 and 56? 

For [2,2,2], Strassen gives rank 7 < 8.
For [2,4,7], let's think about it as two [1,4,7] multiplications plus corrections.

Alternative: Use the fact that C0 and C1 share the same B. 
C0[k] = sum_j A0[j] * B[j,k]
C1[k] = sum_j A1[j] * B[j,k]

We can write:
C0[k] + C1[k] = sum_j (A0[j]+A1[j]) * B[j,k]
C1[k] - C0[k] = sum_j (A1[j]-A0[j]) * B[j,k]

So if we compute S = (A0+A1) @ B and D = (A1-A0) @ B, then:
C0 = (S - D) / 2
C1 = (S + D) / 2

This requires two [1,4,7] multiplications, each of rank 28, total 56. Same as before.

What if we use a different decomposition? For each k, C0[k] and C1[k] are both linear
in the 8 entries of A and 4 entries of B (for column k). 

Actually, I think the key realization is that for [2,4,7], we might not be able to beat 56
easily. Let me just ensure the schoolbook is correct and valid.

Let me verify the schoolbook decomposition carefully:
- For each (i,j,k) in [0,2)x[0,4)x[0,7), create a term c = i*28 + j*7 + k
- U[i*4+j, c] = 1
- V[j*7+k, c] = 1  
- W[k*2+i, c] = 1

The reconstruction: C[i,k] = sum_c (sum_j A[i,j]*U[i*4+j,c]) * (sum_j' B[j',k]*V[j'*7+k,c]) * W[k*2+i,c]

For term (i',j',k'):
- sum_j A[i,j]*U[i*4+j,c] = A[i,j'] if i=i', else 0
- sum_j' B[j',k]*V[j'*7+k,c] = B[j',k] if k=k', else 0... wait, V[j'*7+k,c] is nonzero only when j'=j' and k=k'. So sum_j' B[j',k]*V[j'*7+k,c] = B[j',k] if k=k', else 0.

Hmm, this doesn't quite work. Let me think about what the correct reconstruction formula is.

Given the interface: U (nm x rank), V (mp x rank), W (pn x rank), the standard interpretation
for matrix multiplication tensor decomposition is:

The bilinear map (A, B) -> C where C = A @ B, with A being (n,m) and B being (m,p),
can be written as:

vec(C) = sum_c (U_c^T vec(A)) * (V_c^T vec(B)) * W_c  (reshaped)

More precisely, for each output entry C[i,k] (indexed as k*n+i in W):
C[i,k] = sum_c (sum_{j} A[i,j] * U[i*m+j, c]) * (sum_{j'} B[j',k] * V[j'*p+k, c]) * W[k*n+i, c]

Wait, but that would make U and V act as row/column selectors. For schoolbook:
U[i*m+j, c] = 1 iff c corresponds to (i,j,*), so sum_j A[i,j']*U[i*m+j',c] = A[i,j]
V[j'*p+k, c] = 1 iff c corresponds to (*,j,k), so sum_j' B[j'',k']*V[j''*p+k',c] = B[j,k]

Then C[i,k] = sum_c A[i,j_c] * B[j_c,k_c] * W[k*n+i,c] where (i_c,j_c,k_c) is the index for c.
For c = (i',j',k'): contributes A[i,j'] * B[j',k'] * W[k*n+i, c]
W[k*n+i, c] = 1 iff k=k' and i=i'.
So C[i,k] = sum_{i'=i, j', k'=k} A[i,j'] * B[j',k] * 1 = sum_{j'} A[i,j'] * B[j',k] = (A@B)[i,k]. ✓

Great, so the schoolbook is correct. Rank = 56.

Now, can we do better? The directive says to exploit n=2 structure with a state-space/recurrence approach. Let me think about whether there's a valid sub-56 rank construction.

For now, I'll return the schoolbook rank 56 since I can verify its correctness.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56
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
