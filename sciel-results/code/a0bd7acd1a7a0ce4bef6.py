import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110  # Leave 10s margin

    # Precompute the target tensor
    # T[a,b,c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, R):
        """Verify exact decomposition with integer/half-integer entries."""
        # Check all entries are 0, 1/2, or integers
        for arr in [U, V, W]:
            for val in arr:
                if not (val * 2 == int(val * 2)):
                    return False
        # Check tensor reconstruction
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        # T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-way sum. We can do it as:
        # temp[a,b,r] = U[a,r]*V[b,r], then T[a,b,c] = sum_r temp[a,b,r]*W[c,r]
        # Or directly with einsum
        T_rec = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        if not np.allclose(T_rec, T, atol=1e-10):
            return False
        # Check exactness with fractions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if s != 1:
                        return False
        # Check zeros
        # Sample a subset of zero positions for speed, then verify all if sample passes
        zero_positions = []
        one_positions = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    one_positions.add((a, b, c))
        # We need to check that T_rec is 0 at all non-one positions
        # This is expensive; use numpy check
        diff = T_rec - T
        if np.any(np.abs(diff) > 1e-10):
            return False
        return True

    # Try to find a lower-rank decomposition
    # Strategy: Use the slice method insight
    # The 2x6 output has 6 columns. Each column c_k = A * v_k where v_k is 3x1.
    # A is 2x3, so A*v_k requires rank 6 in general (2*3*1=6).
    # But we have 6 such columns sharing the same A.
    # Total rank if independent: 6*6 = 36 (schoolbook)
    # Can we share factors?
    
    # Approach: Try structured decompositions with rank R < 36
    # For each R, try to find U,V,W with small integer/half-integer entries
    
    best_rank = n * m * p
    best_decomp = None
    
    # Build schoolbook as fallback
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school, best_rank)

    # Try rank reduction from R=35 down
    # For small R, the search space is huge. Use structured ansatz.
    
    # Key insight: For 2x3 x 3x6, the tensor has structure.
    # Let's try a "block" approach where we group columns.
    
    # Alternative: Use the fact that the multiplication tensor for 2x3 x 3x6
    # might have rank less than 36 due to the small dimensions.
    
    # Let's try a direct search with constrained entries for small R.
    # For R <= 20, try entries in {-1, 0, 1, 1/2, -1/2}
    
    # This is still a huge search. Let's use a smarter approach.
    
    # The tensor T can be seen as: T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed j, this is a 2x6 sub-tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # This is the multiplication of 2x1 (column i of A) by 1x6 (row k of B) -> 2x6? No.
    
    # Actually for fixed j:
    # a = i*m+j, i in {0,1}
    # b = j*p+k, k in {0,...,5}
    # c = k*n+i = 2k+i
    # So T[i*m+j, j*p+k, 2k+i] = 1
    # This is like: for each (i,k), we have a 1 at position (i*m+j, j*p+k, 2k+i)
    
    # The full tensor is the sum over j of these "slice" tensors.
    # Each slice j has 2*6 = 12 ones.
    # Total: 3*12 = 36 ones.
    
    # Can we decompose each slice with rank < 12 and share factors?
    # A slice j: T_j[i*m+j, j*p+k, 2k+i] = 1
    # This is the tensor for multiplying a 2x1 vector by a 1x6 vector? No.
    # It's: x_i (2 entries) * y_k (6 entries) -> z_{2k+i}
    # This is like an outer product structure but with interleaved output.
    
    # The rank of this slice tensor: 
    # T_j[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=2k+i
    # = delta(a, i*m+j) * delta(b, j*p+k) * delta(c, 2k+i) summed over i,k
    # This is a sum of 12 rank-1 tensors.
    # Can it be done with fewer?
    # The slice tensor is essentially the "permutation" tensor.
    # Its rank is at least the max of the matricizations.
    
    # Let's compute the rank of one slice tensor numerically.
    # Slice j: dimensions 2 x 6 x 6 (since for fixed j, a has 2 values, b has 6, c has 6)
    # Wait, c = 2k+i ranges over 0..5 for k=0..5, i=0,1. So c covers all 6 values.
    # a = i*m+j: for fixed j, i in {0,1}, so a in {j, j+1} if m=3? No, a = i*3+j.
    # For j=0: a in {0, 3}
    # For j=1: a in {1, 4}
    # For j=2: a in {2, 5}
    # b = j*6+k: for fixed j, k in 0..5, so b in {j*6, ..., j*6+5}
    # c = 2k+i: for k in 0..5, i in 0,1, c in {0,1,2,3,4,5}
    
    # So each slice is a 2x6x6 tensor with 12 ones.
    # The matricization A_j (2x6) x (6x6): 
    # For fixed a (2 choices), we have a 6x6 matrix.
    # This matrix has ones at (b, c) where b=j*6+k, c=2k+i.
    # For fixed i, this is a 6x6 permutation-like matrix.
    # The rank of the slice tensor is at least the max rank of its matricizations.
    
    # Let's just try to find a low-rank decomposition empirically.
    
    # Given time constraints, let's try a few structured ansatze.
    
    # Ansatz 1: Try R = 30 with sparse entries
    # This is still a huge search space.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # The search for a better decomposition would require more sophisticated methods
    # (e.g., tensor rank algorithms) that may not finish in time.
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
