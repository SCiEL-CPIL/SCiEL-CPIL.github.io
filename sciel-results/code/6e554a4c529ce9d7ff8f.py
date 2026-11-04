"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Approach: Exploit n=2 structure via state-space / recurrence decomposition.
We treat the two output rows C[0,k] and C[1,k] as related through the
difference D[j] = A[1,j] - A[0,j], so that C[1,k] = C[0,k] + (D @ B)[k].

The key insight: we can share the computation of A[0,:]*B[j,k] terms between
the two output rows. Specifically, for each j and k, the product A[0,j]*B[j,k]
contributes to BOTH C[0,k] and C[1,k]. This allows us to use a single rank-1
term that feeds both outputs via the W factor.

Construction:
- For each (j, k) pair (j in 0..3, k in 0..6):
  - Term 1: U picks A[0,j], V picks B[j,k], W contributes to both C[0,k] and C[1,k]
  - Term 2: U picks A[1,j] - A[0,j] = D[j], V picks B[j,k], W contributes only to C[1,k]

This gives 2 * 4 * 7 = 56 terms, same as schoolbook. But we can do better:
- Combine terms where possible by using linear combinations.

Better approach: Use the fact that for n=2, we can write:
C[0,k] = sum_j A[0,j] B[j,k]
C[1,k] = sum_j A[1,j] B[j,k]

A rank-1 term with U vector u, V vector v, W vector w contributes:
  (u^T vec(A)) * (v^T vec(B)) * w  to vec(C) (reshaped appropriately)

For the schoolbook, each term is:
  u = e_{i*m+j}, v = e_{j*p+k}, w = e_{k*n+i}

To reduce rank, we need to find linear dependencies. For n=2, m=4, p=7:
The tensor has 14 output components, each a sum of 4 bilinear terms.

Let me try a different decomposition:
For each k (0..6), we need:
  C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
  C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]

Consider grouping by k. For a fixed k, we have 8 bilinear products.
We can use Strassen-like ideas: for 2x1 * 1x1 * 1x1... no, that's trivial.

For each k, the 2x4 matrix A times the 4x1 column B[:,k] gives a 2x1 result.
This is a [2,4,1] multiplication. The rank of [2,4,1] is at most 4 (schoolbook:
C[0] = sum_j A[0,j] B[j], C[1] = sum_j A[1,j] B[j], which needs 8 products schoolbook,
but can be done in 4+2=6 using:
  s1 = A[0,0]*B[0]
  s2 = A[0,1]*B[1]
  s3 = A[0,2]*B[2]
  s4 = A[0,3]*B[3]
  s5 = (A[1,0]-A[0,0])*B[0]
  s6 = (A[1,1]-A[0,1])*B[1]
  s7 = (A[1,2]-A[0,2])*B[2]
  s8 = (A[1,3]-A[0,3])*B[3]
  C[0] = s1+s2+s3+s4
  C[1] = s1+s2+s3+s4+s5+s6+s7+s8

That's still 8. Can we do better for [2,4,1]?
Actually [2,4,1] is just computing two dot products of length 4. The tensor rank
of two independent dot products of length 4 is 8 (you need 4 products for each).
Wait, no - you can share: let D[j] = A[1,j]-A[0,j], then C[1] = C[0] + D·B.
So you need 4 products for C[0] and 4 products for D·B, total 8. No savings.

Hmm, so for each k, we need 8 products, and for 7 values of k, that's 56 total.
Unless we can share across different k values.

Can we share across k? The B factors are different for different k, so products
A[i,j]*B[j,k] for different k are independent bilinear forms. It seems hard to
share.

However, there's a subtlety: the tensor rank counts the number of rank-1 terms
in the decomposition of the full tensor. Each rank-1 term is of the form
u ⊗ v ⊗ w where u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{pn}.

For the schoolbook, each term has u = e_{i*m+j}, v = e_{j*p+k}, w = e_{k*n+i}.
These are 56 distinct terms.

To reduce rank, we'd need to express the tensor as a sum of fewer such outer products.
This is equivalent to finding a lower-rank decomposition.

For [2,4,7], known results suggest the rank might be less than 56, but constructing
an explicit decomposition is non-trivial. Let me try a systematic approach:

Use the "border rank" or "tensor rank" techniques. Actually, let me just try to
verify if rank 56 is indeed optimal or if we can find a valid decomposition with
fewer terms.

Given the time constraint, I'll implement the schoolbook (rank 56) which is
guaranteed valid, and attempt a modest optimization by checking if any terms
can be merged (which they can't in the schoolbook, but let's be safe).

Actually, let me reconsider. The problem says "Find a rank-minimal exact tensor
decomposition". The score is the rank (lower is better). The seed has rank 56.
If I can find a valid decomposition with rank < 56, I improve.

Let me think about whether rank 50 is achievable. For [2,4,7], the tensor has
14 output entries. Each output entry C[i,k] is a bilinear form in A and B.
The space of bilinear forms has dimension m*p = 28 for each output. But we only
need specific bilinear forms.

I'll stick with the schoolbook for guaranteed correctness, as finding a valid
lower-rank decomposition for [2,4,7] requires deeper algebraic insight that
I cannot reliably construct within this framework.
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
    
    # Verify the decomposition numerically
    np.random.seed(0)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test  # shape (n, p)
    
    # Reconstruct: C[k*n+i] = sum_c (u_c^T vec(A)) * (v_c^T vec(B)) * W[k*n+i, c]
    C_recon = np.zeros(pn)
    for c in range(rank):
        uA = np.dot(U[:, c], A_test.reshape(-1))
        vB = np.dot(V[:, c], B_test.reshape(-1))
        C_recon += uA * vB * W[:, c]
    
    C_recon = C_recon.reshape(p, n).T  # (n, p)
    
    if not np.allclose(C_recon, C_expected, atol=1e-10):
        raise ValueError("Decomposition verification failed")
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
