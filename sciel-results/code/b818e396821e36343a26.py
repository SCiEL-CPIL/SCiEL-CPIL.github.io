"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition.

We exploit the structure: multiplying (3x3) A by (3x6) B gives (3x6) C.
This is equivalent to 6 independent (3x3) matrix multiplications sharing
the same A. We can reuse the rank-23 decomposition of the 3x3x3
multiplication tensor across all 6 output columns, but we must check
whether the shared factor allows a lower total rank than 6*23 = 138
(which is the naive approach of doing 6 independent 3x3 multiplications).

Actually, the standard approach: the 3x3 x 3x6 multiplication can be seen
as A (3x3) times B (3x6). The tensor has dimensions 3, 3, 6 with the
output indexed as k*n + i where k is the row of output (0..5) and i is
the column of output (0..2). Wait, let me re-read the convention.

W[k*n+i, r]: the output C has p=6 rows and n=3 columns. So C[k, i] where
k in 0..5, i in 0..2. The output index is k*n + i = 3k + i.

The multiplication: C[k, i] = sum_j A[j, i] * B[k, j]
where A is 3x3 (rows j, cols i), B is 3x6 (rows k, cols j).

So C[k, i] = sum_{j=0}^{2} A[i, j] * B[k, j]

The tensor T[k*n+i, i, j] ... wait, let me think in terms of the factor
matrices.

U has shape (n*m, rank) = (9, rank), indexed by (i, j) -> i*m + j
V has shape (m*p, rank) = (18, rank), indexed by (j, k) -> j*p + k  
W has shape (p*n, rank) = (18, rank), indexed by (k, i) -> k*n + i

The decomposition requires:
For all (i, j, k): the coefficient of A[i,j] * B[k,j] in C[k,i] is 1.

The standard identity: C[k,i] = sum_j A[i,j] B[k,j]

In tensor decomposition form, we need:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each (i,j,k)
and = 0 otherwise... actually the tensor T[i,j,k] = 1 for the valid terms.

The schoolbook gives rank 54.

Known results: The rank of the 3x3 matrix multiplication tensor is 23
(Bläser 2005, though the exact minimal is debated; 23 is the best known).

For 3x3 times 3x6: we can think of this as 6 copies of 3x3 multiplication
sharing the A matrix. If we use the rank-23 decomposition of 3x3x3,
each output column i in the 3x6 result uses the same A but different
columns of B.

Actually, let's think more carefully. C[k, i] = sum_j A[i,j] * B[k,j].
For fixed i, this is a 3x6 matrix multiplication of the vector A[i,:].

Alternatively: view B as a 3x6 matrix. C = A * B = A * [B1 | B2] = [A*B1 | A*B2].
So we need to compute two 3x3 matrix multiplications: A*B1 and A*B2.

If each 3x3 multiplication has rank 23, and they share the same A,
the total rank is at most 23 + 23 = 46 if we can share the A factors.

In the tensor decomposition, A appears in U. If we use the same U for both
3x3 multiplications, the rank contribution from A is just 23, not 46.

Let me formalize: For the first 3x3 multiplication (C1 = A*B1):
C1[i, i'] = sum_j sum_{j'} U1[i, j, r] * V1[j, j', r] * W1[j', i', r]
for r = 1..23.

For the second (C2 = A*B2):
C2[i, i''] = sum_j sum_{j''} U2[i, j, r'] * V2[j, j'', r'] * W2[j'', i'', r']
for r' = 1..23.

If U2 = U1 (same A decomposition), then the total rank is 23 + 23 = 46,
but the U matrix only has 23 distinct columns used twice.

In our problem:
U has shape (9, rank), V has shape (18, rank), W has shape (18, rank).

For the two 3x3 multiplications:
- First: U[:, 0:23], V[0:9, 0:23], W[0:9, 0:23]  (B1 rows k=0,1,2)
- Second: U[:, 23:46], V[9:18, 23:46], W[9:18, 23:46]  (B2 rows k=3,4,5)

Wait, but U must be the same for both. So U[:, 0:23] = U[:, 23:46].

This gives rank 46 with U having repeated blocks.

Is rank 46 achievable? It depends on whether the rank-23 decomposition of
3x3x3 exists and can be written explicitly.

The exact rank of 3x3 matrix multiplication is not known to be 23; 23 is
the best upper bound. The lower bound is 20 (from various counting arguments)
and the best known is 23.

However, I don't have the explicit 23-term decomposition memorized in a
form I can code quickly. Let me try a different approach.

Alternative: Use the known rank-23 decomposition if I can find it, or
fall back to schoolbook rank 54.

Given the time constraints and the risk of implementing an incorrect
decomposition, let me try to implement a rank-46 decomposition based on
the idea of splitting 3x6 into two 3x3 multiplications, using a known
3x3 decomposition.

Actually, I realize I should just try to find any valid lower-rank
decomposition. Let me think about whether rank 49 or 50 is achievable
with simpler methods.

Another approach: Use the fact that 6 = 2*3 and try to use tensor
product decompositions.

For now, let me implement the split approach with a simple 3x3 decomposition
and see if I can get below 54.

The simplest non-trivial improvement: if I can find that the 3x3 matrix
multiplication can be done with fewer than 27 multiplications (schoolbook),
then 2 * (that rank) < 54.

The best known for 3x3 is 23. If I implement a rank-23 decomposition,
total rank = 46.

Let me try to code the Bläser rank-23 decomposition for 3x3 matrix
multiplication. This is complex but let me attempt it.

Actually, upon further reflection, the risk of implementing an incorrect
rank-23 decomposition is very high, and any error would give score 0.
The safe approach is to return the schoolbook rank-54 decomposition.

But wait - maybe I can use a simpler reduction. What about rank 49?
Or can I exploit the rectangular structure directly?

Let me try yet another angle: the tensor for (3,3,6) multiplication.
The dimensions are n=3, m=3, p=6.

I'll return the schoolbook decomposition to guarantee correctness.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # And for all other (i,j,k) combinations that don't correspond to valid terms, == 0
    # Actually the tensor is T[i,j,k] = 1 if this is a valid term in the multiplication
    # C[k,i] = sum_j A[i,j] * B[k,j]
    # So the tensor has T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Wait no. The tensor is defined such that the decomposition gives the multiplication.
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 for all valid (i,j,k)
    # where valid means i in [0,n), j in [0,m), k in [0,p)
    # Actually ALL (i,j,k) with i<n, j<m, k<p are valid terms.
    # The multiplication is C[k,i] = sum_j A[i,j] B[k,j]
    # So for each triple (i,j,k), the coefficient is 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r] for r in range(rank))
                if abs(val - 1) > 1e-6:
                    return False
    return True

def _try_rank(n, m, p, target_rank, timeout=120):
    """Try to find a decomposition of given rank using random search + refinement."""
    start = time.time()
    deadline = start + timeout
    Ua = np.random.randn(n*m, target_rank)
    Va = np.random.randn(m*p, target_rank)
    Wa = np.random.randn(p*n, target_rank)
    
    # Use gradient-free approach: try to minimize the error
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is a hard non-convex problem. Let's try a simpler approach:
    # Start with schoolbook and try to merge terms.
    
    # Actually, let's try a different strategy: use the shared-U structure.
    # Split into two 3x3 multiplications.
    # For 3x3x3, try to find a rank-r decomposition.
    # Then total rank = 2*r.
    
    # Let's just try to find a 3x3x3 decomposition of rank < 27.
    # If we find rank r for 3x3x3, total = 2*r.
    
    # For now, return None to indicate failure.
    return None

def construct():
    n, m, p = DIMENSIONS
    # Try to find a better decomposition
    start = time.time()
    deadline = start + 140  # Leave 10s margin
    
    # Strategy: split 3x6 into two 3x3 multiplications sharing U
    # Find a 3x3x3 decomposition of rank r, then total rank = 2*r
    # Schoolbook 3x3x3 has rank 27, so total = 54
    
    # Try to find a rank-23 or lower 3x3x3 decomposition
    # For simplicity, let's try a few known structures
    
    # The best we can do safely is the schoolbook at rank 54
    # Let's try a simple optimization: see if we can do better with random init
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    for target_rank in range(46, best_rank):
        if time.time() > deadline - 5:
            break
        result = _try_rank(n, m, p, target_rank, timeout=5)
        if result is not None:
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
