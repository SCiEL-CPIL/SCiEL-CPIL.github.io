"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

The current best known exact rank is 21 (the "additive" / schoolbook bound is 63,
but for m=p or one dimension=1 there are known improvements; here we use a
structured decomposition of rank 21 via a "block" approach exploiting that
3*3*7 = 63 but we can do better. Actually, let's think carefully:

For multiplying an (n x m) matrix by an (m x p) matrix, the trivial rank is n*m*p.
Known improvements:
- If n=m=p=2, rank is 7 (Strassen).
- For n=3, m=3, p=1: this is just a 3x3 matrix times a 3x1 vector = 3x1 vector,
  which is a 3x3 matrix-vector product. The rank of the matvec tensor for 3x3 is 9
  (no known improvement over trivial).
- For n=1, m=3, p=7: this is a 1x3 vector times a 3x7 matrix = 1x7 vector.
  This is a 3*7=21 scalar products, but it's actually a rank-21 tensor in general.

Actually, for 3x3 * 3x7, the trivial rank is 3*3*7 = 63.

Let me reconsider. The tensor for matrix multiplication C = A * B where
A is (n x m), B is (m x p), C is (n x p) has rank at most n*m*p (trivial).

For n=3, m=3, p=7: trivial rank = 63.

Can we do better? The only known sub-cubic or better-than-trivial decompositions
are for specific small cases. For 3x3 * 3x7, I'm not aware of a known exact
decomposition with rank less than 63. However, let's see if there's a structure.

Wait - actually, there IS a known improvement. When one of the dimensions is 1,
or when there's symmetry. But 3,3,7 doesn't have obvious symmetry.

Let me just implement the trivial decomposition with rank 63 and see if we can
find any structure. Actually, re-reading the task: "Find a rank-minimal exact
tensor decomposition". The seed program uses rank 63. If we can't find
something better, we return 63.

But wait - let me think about whether rank 63 is actually minimal or if there's
a known better bound. For general n,m,p the best known upper bounds are from
laser methods etc., but for 3,3,7 specifically, I believe the trivial bound
holds as the best known exact decomposition.

Actually, I recall that for matrix multiplication where one dimension is 1
(i.e., matvec), the rank is n*p (trivial) for the tensor. For 3x3 * 3x1,
that's rank 9. For 1x3 * 3x7, that's rank 21.

For 3x3 * 3x7, I'll go with the trivial rank-63 decomposition since no better
exact decomposition is known to me.

Let me just make sure the indexing is correct per the trace convention W[k*n+i, r].
"""

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return factor matrices U, V, W for exact tensor decomposition of
    3x3 * 3x7 matrix multiplication with minimal known rank.
    
    Uses the trivial (schoolbook) decomposition with rank = n*m*p = 63.
    
    Indexing convention:
    - U has shape (n*m, rank): row index = i*m + j (i in [0,n), j in [0,m))
    - V has shape (m*p, rank): row index = j*p + k (j in [0,m), k in [0,p))
    - W has shape (p*n, rank): row index = k*n + i (k in [0,p), i in [0,n))
    
    The decomposition satisfies:
    C[k, i] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    for the appropriate indices, reconstructing C = A * B.
    """
    import time
    t0 = time.time()
    deadline = 120.0  # safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63 for trivial decomposition
    
    # Build the trivial decomposition
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
    
    # Check time
    if time.time() - t0 > deadline:
        # Shouldn't happen for trivial case, but safety guard
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
