"""
Rank-minimal exact tensor decomposition for 2x2 matrix multiplied by 2x11 matrix.

Strategy:
- The multiplication tensor for A (2x2) * B (2x11) -> C (2x11) can be decomposed
  by treating B as 11 columns. For each column of B (a 2x1 vector), the product
  is a 2x2 matrix times a 2x1 vector.
- A 2x2 times 2x1 multiplication has tensor rank 4 (schoolbook) or can potentially
  be done with fewer. Actually, for 2x2 * 2x1, the output is 2x2 = 4 entries, each
  is a bilinear form. The rank is 4 (it's just 4 independent bilinear forms).
- However, we can use Strassen-like tricks across multiple columns.

Actually, let me think more carefully. The tensor is for (n,m,p) = (2,2,11).
U has shape (n*m, rank) = (4, rank)
V has shape (m*p, rank) = (22, rank)  
W has shape (p*n, rank) = (22, rank)

The decomposition satisfies:
C[k*n+i, j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, let me re-read the convention. The task says:
U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
W[k*n+i, r] - so W is indexed by (k*n+i) which goes from 0 to p*n-1 = 22-1.

The multiplication is: (A: n x m) * (B: m x p) = (C: n x p)
Standard: C[i,k] = sum_j A[i,j] * B[j,k]

So C[i,k] = sum_r U[i*m + j... wait.

Let me think about the standard decomposition. For each (i,j,k):
C[i,k] gets contribution A[i,j] * B[j,k].

In the tensor decomposition:
C[i,k] = sum_r U[i, r] * V[j, r] * W[k, r]  -- no, that's not right either.

The tensor T has indices (i, j, k) with T[i,j,k] = delta for the multiplication.
Actually, the multiplication tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].

The decomposition is:
sum_r U[i, r] V[j, r] W[k, r] = T[i,j,k]

Where T[i,j,k] = 1 for all (i,j,k) in the schoolbook case? No.

Let me use the convention from the seed code:
U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]

So U is indexed by (i, j) flattened as i*m+j, size n*m = 4
V is indexed by (j, k) flattened as j*p+k, size m*p = 22
W is indexed by (k, i) flattened as k*n+i, size p*n = 22

The product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal the tensor entry.

For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k], the tensor is:
T[i,j,k] such that when we contract, we get C[i,k].

The identity is: C[i,k] = sum_{j,c} A[i,j] V_part * ... 

Actually the standard is:
C[i,k] = sum_j A[i,j] B[j,k]

In tensor form with the given indexing:
sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should give the coefficient.

For schoolbook, for each (i,j,k), we have a term where:
U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1

And this contributes to... let's see. The product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] 
is 1 for that specific (i,j,k). But we need C[i,k] = sum_j A[i,j] B[j,k].

So the tensor entry T[i,j,k] should be 1 for all valid (i,j,k), and the decomposition 
gives sum over c of the products, which when multiplied by A and B gives the result.

More precisely: the bilinear form for C[i,k] is sum_j A[i,j] B[j,k].
This can be written as sum_j (A[i,j] * 1) * (B[j,k] * 1) but that's not a rank-1 tensor.

The rank-1 tensors are of the form: (A[i,j] coefficient) * (B[j,k] coefficient) -> C[i,k] coefficient.
A rank-1 tensor T = U ⊗ V ⊗ W where T[i,j,k] = U[i,j] V[j,k] W[k,i] (with the indexing).

For schoolbook, we need 4*1*11 = 44 terms, one for each (i,j,k).

Now, can we do better? 

Key insight: For a fixed i and k, C[i,k] = sum_j A[i,j] B[j,k].
This is a sum over j=0,1 of products. So for each (i,k), we need at least... well, 
it's a bilinear form in A[i,:] and B[:,k].

For the 2x2 * 2x1 case (p=1), we have C[i,0] = sum_j A[i,j] B[j,0] for i=0,1.
That's 2 outputs, each a sum of 2 products. The rank is 4 (can't reduce for this case 
since each output involves different A entries).

Actually wait - for 2x2 * 2x1, the outputs are:
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]

These share B[0,0] and B[1,0]. Can we use Strassen-like tricks? 
The rank of 2x2*2x1 is 4. (It's equivalent to 4 independent bilinear forms.)

For p=11, naively 4*11=44. But we can potentially share across k values.

Consider grouping columns. For two columns k1, k2:
C[i,k1] = A[i,0]*B[0,k1] + A[i,1]*B[1,k1]
C[i,k2] = A[i,0]*B[0,k2] + A[i,1]*B[1,k2]

For i=0: C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1]
C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2]
C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1]
C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2]

That's 4 outputs, each a sum of 2 products. 8 products total if done naively.
Can we do it with 7? Using Strassen on the "matrix" formed by these?

Actually, this is exactly the 2x2 * 2x2 problem if we consider B as a 2x2 matrix 
(columns k1,k2). And we know Strassen does 2x2*2x2 in rank 7.

So for each pair of columns, we can use Strassen (rank 7) instead of 4*2=8 schoolbook terms.

11 = 5*2 + 1. So 5 pairs (rank 7 each = 35) + 1 single column (rank 4) = 39.

Let me implement Strassen for 2x2*2x2 and schoolbook for the remaining column.

Strassen's 7 products for A(2x2) * B(2x2) = C(2x2):
Let A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]

P1 = (a00+a11)(b00+b11)
P2 = (a10+a11)b00
P3 = a00(b11-b01)
P4 = a11(b10-b00)
P5 = (a00+a01)b11
P6 = (a10-a00)(b00+b01)
P7 = (a01-a11)(b10+b11)

C00 = P1 + P4 - P5 + P7
C01 = P3 + P5
C10 = P2 + P4
C11 = P1 - P2 + P3 + P6

Now I need to express these as rank-1 tensor terms in the U,V,W framework.

For a single Strassen term like P1 = (a00+a11)(b00+b11):
This is (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k1]) for column k1.

In our indexing, for column k1 (say k=0 in the pair, so global k=2*s for pair s):
The contribution to C[i,k] from this product.

P1 contributes to:
C[0,k1] gets +P1
C[1,k1] gets -P2... no wait, P1 contributes to C00 and C11 of the 2x2 block.

Let me be more careful. For pair of columns (k_a, k_b) where k_a = 2*s, k_b = 2*s+1:

The 2x2 B block is B[:, [k_a, k_b]] = [[B[0,k_a], B[0,k_b]], [B[1,k_a], B[1,k_b]]]

Strassen products:
P1 = (A[0,0]+A[1,1]) * (B[0,k_a]+B[1,k_a])
P2 = (A[1,0]+A[1,1]) * B[0,k_a]
P3 = A[0,0] * (B[1,k_b]-B[0,k_b])
P4 = A[1,1] * (B[1,k_a]-B[0,k_a])
P5 = (A[0,0]+A[0,1]) * B[1,k_b]
P6 = (A[1,0]-A[0,0]) * (B[0,k_a]+B[0,k_b])
P7 = (A[0,1]-A[1,1]) * (B[1,k_a]+B[1,k_b])

Outputs:
C[0,k_a] = P1 + P4 - P5 + P7
C[0,k_b] = P3 + P5
C[1,k_a] = P2 + P4
C[1,k_b] = P1 - P2 + P3 + P6

Now each P_r is a product of a linear form in A and a linear form in B.
Each such product is a rank-1 tensor.

For P1: 
U part: coefficient of A[0,0] is 1, A[1,1] is 1, others 0. So U vector: [1, 0, 0, 1] (for indices (0,0),(0,1),(1,0),(1,1))
V part: coefficient of B[0,k_a] is 1, B[1,k_a] is 1. V is indexed by (j,k) -> j*p+k. 
  B[0,k_a] -> index 0*11+k_a = k_a
  B[1,k_a] -> index 1*11+k_a = 11+k_a
  So V vector has 1 at positions k_a and 11+k_a.
W part: This product contributes to C[0,k_a] with coeff +1 and C[1,k_b] with coeff +1.
  W is indexed by (k,i) -> k*n+i. 
  C[0,k_a] -> k=k_a, i=0 -> index k_a*2+0 = 2*k_a
  C[1,k_b] -> k=k_b, i=1 -> index k_b*2+1 = 2*k_b+1
  So W vector has 1 at positions 2*k_a and 2*k_b+1.

Let me code this up. I'll create helper to add a rank-1 term.

For schoolbook single column k:
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

4 terms:
1. A[0,0]*B[0,k] -> C[0,k]: U=[1,0,0,0], V has 1 at 0*11+k=k, W has 1 at k*2+0=2k
2. A[0,1]*B[1,k] -> C[0,k]: U=[0,1,0,0], V has 1 at 11+k, W has 1 at 2k
3. A[1,0]*B[0,k] -> C[1,k]: U=[0,0,1,0], V has 1 at k, W has 1 at k*2+1=2k+1
4. A[1,1]*B[1,k] -> C[1,k]: U=[0,0,0,1], V has 1 at 11+k, W has 1 at 2k+1

Now let me implement the full construct() function.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START

def _add_term(U, V, W, u_idx_vals, v_idx_vals, w_idx_vals):
    """Add a rank-1 term. u_idx_vals: list of (u_index, coeff), etc."""
    for ui, uc in u_idx_vals:
        U[ui] += uc
    for vi, vc in v_idx_vals:
        V[vi] += vc
    for wi, wc in w_idx_vals:
        W[wi] += wc


def _strassen_pair(U, V, W, k_a, k_b, n, m, p):
    """Add Strassen 7-product decomposition for columns k_a, k_b."""
    # P1 = (A00+A11)(B0ka+B1ka)
    _add_term(U, V, W,
              [(0, 1), (3, 1)],  # A[0,0] and A[1,1]
              [(0*p+k_a, 1), (1*p+k_a, 1)],  # B[0,k_a], B[1,k_a]
              [(k_a*n+0, 1), (k_b*n+1, 1)])  # C[0,k_a], C[1,k_b]
    
    # P2 = (A10+A11)B0ka
    _add_term(U, V, W,
              [(2, 1), (3, 1)],  # A[1,0], A[1,1]
              [(0*p+k_a, 1)],  # B[0,k_a]
              [(k_a*n+1, 1)])  # C[1,k_a]
    
    # P3 = A00(B1kb-B0kb)
    _add_term(U, V, W,
              [(0, 1)],  # A[0,0]
              [(1*p+k_b, 1), (0*p+k_b, -1)],  # B[1,k_b], -B[0,k_b]
              [(k_b*n+0, 1)])  # C[0,k_b]
    
    # P4 = A11(B1ka-B0ka)
    _add_term(U, V, W,
              [(3, 1)],  # A[1,1]
              [(1*p+k_a, 1), (0*p+k_a, -1)],  # B[1,k_a], -B[0,k_a]
              [(k_a*n+0, 1), (k_a*n+1, 1)])  # C[0,k_a], C[1,k_a]
    
    # P5 = (A00+A01)B1kb
    _add_term(U, V, W,
              [(0, 1), (1, 1)],  # A[0,0], A[0,1]
              [(1*p+k_b, 1)],  # B[1,k_b]
              [(k_a*n+0, -1), (k_b*n+0, 1)])  # -C[0,k_a], +C[0,k_b]
    
    # P6 = (A10-A00)(B0ka+B0kb)
    _add_term(U, V, W,
              [(2, 1), (0, -1)],  # A[1,0], -A[0,0]
              [(0*p+k_a, 1), (0*p+k_b, 1)],  # B[0,k_a], B[0,k_b]
              [(k_b*n+1, 1)])  # C[1,k_b]
    
    # P7 = (A01-A11)(B1ka+B1kb)
    _add_term(U, V, W,
              [(1, 1), (3, -1)],  # A[0,1], -A[1,1]
              [(1*p+k_a, 1), (1*p+k_b, 1)],  # B[1,k_a], B[1,k_b]
              [(k_a*n+0, 1), (k_b*n+1, 1)])  # C[0,k_a], C[1,k_b]


def _schoolbook_col(U, V, W, k, n, m, p):
    """Add schoolbook 4-term decomposition for column k."""
    # A00*B0k -> C0k
    _add_term(U, V, W,
              [(0, 1)],  # A[0,0]
              [(0*p+k, 1)],  # B[0,k]
              [(k*n+0, 1)])  # C[0,k]
    # A01*B1k -> C0k
    _add_term(U, V, W,
              [(1, 1)],  # A[0,1]
              [(1*p+k, 1)],  # B[1,k]
              [(k*n+0, 1)])  # C[0,k]
    # A10*B0k -> C1k
    _add_term(U, V, W,
              [(2, 1)],  # A[1,0]
              [(0*p+k, 1)],  # B[0,k]
              [(k*n+1, 1)])  # C[1,k]
    # A11*B1k -> C1k
    _add_term(U, V, W,
              [(3, 1)],  # A[1,1]
              [(1*p+k, 1)],  # B[1,k]
              [(k*n+1, 1)])  # C[1,k]


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    total_rank = 5 * 7 + 4  # 5 Strassen pairs + 1 schoolbook column
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    # Process 5 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9)
    for s in range(5):
        k_a = 2 * s
        k_b = 2 * s + 1
        _strassen_pair(U, V, W, k_a, k_b, n, m, p)
        col += 7
    
    # Process remaining column: k=10
    _schoolbook_col(U, V, W, 10, n, m, p)
    col += 4
    
    # Convert to lists of lists for return
    U_list = [[U[i][r] for r in range(total_rank)] for i in range(n * m)]
    V_list = [[V[i][r] for r in range(total_rank)] for i in range(m * p)]
    W_list = [[W[i][r] for r in range(total_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
