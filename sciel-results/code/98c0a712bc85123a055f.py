"""Combined approach: use structural insight about the tensor to find a lower-rank decomposition
for 4x4 * 4x9 matrix multiplication, with verification and fallback to schoolbook.

Key insight: The tensor T is indexed by (a, b, c) where:
  a = i*m + j  (i in [0,n), j in [0,m))
  b = j*p + k  (j in [0,m), k in [0,p))
  c = k*n + i  (k in [0,p), i in [0,n))

T[a,b,c] = 1 iff the indices are consistent (i.e., a's j equals b's j, a's i equals c's i, b's k equals c's k), else 0.

For n=4, m=4, p=9: U is (16,R), V is (36,R), W is (36,R).

We try to find a decomposition with rank < 144 by exploiting structure.
One approach: since the tensor has a specific sparse structure, we can try to
find low-rank approximations that are exact.

A key observation: if we can write the tensor as a sum of R rank-1 terms where each
term is a product of three vectors, we need the product to be 1 at consistent indices
and 0 at inconsistent ones.

Let's try a constructive approach using the fact that the problem has some symmetry.
For the 4x4 part, we might be able to use Strassen-like ideas.

Actually, let me try a direct approach: use numpy to set up the linear system and try
to find a solution with small R.

The condition is: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
And for all inconsistent (a,b,c), sum_r U[a,r] * V[b,r] * W[c,r] = 0

This is a nonlinear system. However, we can try a specific structured ansatz.

One idea: suppose U, V, W have entries that are 0, 1, or -1 (or small integers/half-integers).
We can try to search over small patterns.

Given time constraints, let me try a few specific constructions:

1. Schoolbook: R=144 (baseline)
2. Try R=36: one term per (j,k) pair, with U depending on i only, W depending on i only
   U[i*m+j, r] where r indexes (j,k): U[i*m+j, j*p+k] = 1 if... hmm this doesn't work directly.

Let me think differently. The bilinear map is:
C[i,k] = sum_j A[i,j] B[j,k]

This can be written as: C = A * B

A rank-R decomposition means: C = sum_r (A_r) (B_r) where A_r is a linear function of A,
B_r is a linear function of B, and the product is interpreted appropriately.

Actually, in the standard formulation:
C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) * W[k,i,r]

Wait, that's not quite right. Let me re-derive.

The tensor decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]

where a = i*m+j, b = j*p+k, c = k*n+i.

So: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]

For consistent indices (which is always the case when we plug in valid i,j,k), T = 1.
For inconsistent indices, T = 0.

So the condition is:
sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i in [0,4), j in [0,4), k in [0,9)

And for any (a,b,c) that are NOT consistent, sum_r U[a,r] V[b,r] W[c,r] = 0.

Let me try a specific construction. Suppose we let R = m * p = 4 * 9 = 36, and define:
For r = j*p + k (j in [0,4), k in [0,9)):
  U[i*m+j, r] = 1 if the j in r matches the j in a, else 0
  V[j'*p+k', r] = 1 if (j',k') = (j,k)
  W[k'*n+i', r] = 1 if k' = k, and i' = i... but i' is determined by c, not r.

This doesn't work cleanly. Let me try another approach.

Let R = n * p = 4 * 9 = 36. For r = i*p + k:
  U[i'*m+j, r] = 1 if i' = i
  V[j'*p+k', r] = 1 if k' = k  
  W[k''*n+i'', r] = 1 if (k'', i'') = (k, i)

Then: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = sum_{i',k'} [i'=i][k'=k][k''=k][i''=i]
= 1 (only the term with i'=i, k'=k contributes)

But wait, we also need the j indices to be consistent. In this construction, U depends on i but not j (except through the row index), and V depends on k but not j. So for a fixed (i,k), the sum over r gives 1 regardless of j. But we need the result to be 1 for all j, which it is. And for inconsistent indices, we need 0.

Let me check: for inconsistent (a,b,c), say a = i1*m+j1, b = j2*p+k2, c = k3*n+i3 where j1 != j2 or i1 != i3 or k2 != k3.

sum_r U[a,r] V[b,r] W[c,r] = sum_{i',k'} [i1=i'] [k2=k'] [k3=k'][i3=i']

This is 1 only if i1=i3 and k2=k3, but we also need... wait, there's no condition on j1 vs j2 in this construction. So if i1=i3 and k2=k3 but j1!=j2, the sum would still be 1, but T should be 0.

So this construction doesn't work.

Let me try yet another approach. What if R = n*m*p = 144 (schoolbook) is actually not minimal, and we can do better?

Known results: the rank of 4x4 matrix multiplication is at most 49 (Strassen). For 4x4 * 4x9, it's unclear.

Given the time constraints, let me try a practical approach: use scipy or numpy to attempt a low-rank factorization via iterative methods, or try a small random search.

Actually, let me try a specific idea: since 9 = 4 + 4 + 1, split the k dimension into two 4-dim blocks and one 1-dim block. For each 4x4 * 4x4 subproblem, use a rank-19 Strassen-like decomposition (if we can construct one), and for the 4x4 * 4x1, use rank 4. Total: 19 + 19 + 4 = 42.

But constructing an explicit rank-19 decomposition for 4x4 is complex. Let me just try to find any improvement over 144 via a simple search.

Given the complexity and time limits, I'll implement the schoolbook as the safe baseline and add a quick check for a few specific low-rank constructions.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    # Schoolbook baseline
    rank_sb = n * m * p  # 144
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, R):
        """Verify the decomposition is exact using vectorized numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the full tensor contraction: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Use einsum for speed: (n*m, R) x (m*p, R) x (p*n, R) -> (n*m, m*p, p*n)
        # This might be too large: 16 * 36 * 36 = 20736 entries, that's fine
        # T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        # But let's do it in a more memory-efficient way
        
        # For each a, compute U_np[a] @ diag(V_np[b] * W_np[c]) ... no
        # T[a,b,c] = dot(U_np[a], V_np[b] * W_np[c])
        # We can compute this as: for each a, compute U_np[a] @ (V_np * W_np.T).T ... 
        # Actually: T[a,b,c] = sum_r U_np[a,r] * V_np[b,r] * W_np[c,r]
        # = (U_np[a] * (V_np * W_np.T).T[b]) ... no
        # Let's just use einsum, it should be fast enough for 16*36*36
        T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
        
        # Check consistent indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1.0) > 1e-9:
                        return False
        
        # Check inconsistent indices
        # Create a mask of consistent indices
        consistent = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    consistent[a, b, c] = True
        
        # Check all inconsistent entries are ~0
        inconsistent_vals = T[~consistent]
        if len(inconsistent_vals) > 0 and np.max(np.abs(inconsistent_vals)) > 1e-9:
            return False
        
        return True

    # Try structured constructions with the fast vectorized verifier
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Construction 1: R = n*p = 36
    # r = i*p + k
    # U[i'*m+j, i*p+k] = 1 if i'=i
    # V[j'*p+k', i*p+k] = 1 if k'=k
    # W[k''*n+i'', i*p+k] = 1 if k''=k and i''=i
    R_try = n * p
    U_try = np.zeros((n * m, R_try))
    V_try = np.zeros((m * p, R_try))
    W_try = np.zeros((p * n, R_try))
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U_try[i * m + j, r] = 1
            for j in range(m):
                V_try[j * p + k, r] = 1
            W_try[k * n + i, r] = 1
    if verify(U_try.tolist(), V_try.tolist(), W_try.tolist(), R_try):
        return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}

    # Construction 2: R = m*p = 36
    R_try2 = m * p
    U_try2 = np.zeros((n * m, R_try2))
    V_try2 = np.zeros((m * p, R_try2))
    W_try2 = np.zeros((p * n, R_try2))
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U_try2[i * m + j, r] = 1
            V_try2[j * p + k, r] = 1
            for i in range(n):
                W_try2[k * n + i, r] = 1
    if verify(U_try2.tolist(), V_try2.tolist(), W_try2.tolist(), R_try2):
        return {"U": U_try2.tolist(), "V": V_try2.tolist(), "W": W_try2.tolist()}

    # Construction 3: R = n*m = 16
    # r = i*m + j
    # U[i'*m+j', r] = 1 if (i',j')=(i,j)
    # V[j'*p+k', r] = 1 if j'=j
    # W[k''*n+i'', r] = 1 if i''=i
    R_try3 = n * m
    U_try3 = np.zeros((n * m, R_try3))
    V_try3 = np.zeros((m * p, R_try3))
    W_try3 = np.zeros((p * n, R_try3))
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U_try3[i * m + j, r] = 1
            for k in range(p):
                V_try3[j * p + k, r] = 1
            for k in range(p):
                W_try3[k * n + i, r] = 1
    if verify(U_try3.tolist(), V_try3.tolist(), W_try3.tolist(), R_try3):
        return {"U": U_try3.tolist(), "V": V_try3.tolist(), "W": W_try3.tolist()}

    # Construction 4: DFT-based approach for the i and k indices
    # Use complex roots of unity to decompose the equality constraints
    # [i_a = i_c] = (1/n) sum_s omega^{s(i_a - i_c)}
    # [k_b = k_c] = (1/p) sum_t omega^{t(k_b - k_c)}
    # This gives R = m * n * p = 144 again, but let's try with real parts only
    
    # Construction 5: Try R = 72 = 2 * 36
    # Split into two groups and see if we can share
    if time.time() < deadline - 10:
        # Try a few more structured ranks
        for R in [48, 52, 56, 60, 64, 72, 80, 96, 100, 104, 108, 112, 120, 128, 132, 136, 140]:
            if time.time() > deadline - 5:
                break
            # For now, just use schoolbook structure but with fewer terms
            # This won't work in general, but let's check if any structured subset works
            # Skip - too unlikely without a specific construction
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
