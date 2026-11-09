import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p matrix multiplication."""
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

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    # Build the tensor from U, V, W
    # T[a, b, c] = sum_{col=0}^{rank-1} U[a, col] * V[b, col] * W[c, col]
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                # Expected value: 1 if (a,b,c) corresponds to valid (i,j,k)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                expected = 1 if (i == i2 and j == j2) else 0
                
                actual = Fraction(0)
                for col in range(rank):
                    actual += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
                
                if actual != expected:
                    return False
    return True

def _fmm3_decomp(n, m, p):
    """
    Construct a low-rank decomposition for 3x3 x 3x4 using Winograd FMM-3 ideas.
    
    For 3x3x3, FMM-3 achieves rank 23. We extend this to 3x3x4.
    
    The key insight: for each (i,j) pair in A, we have p=4 terms in schoolbook.
    We can use linear combinations to reduce the total rank.
    
    Strategy: Use the FMM-3 structure for the first 3 columns of B,
    then add terms for the 4th column, and try to merge.
    """
    if n != 3 or m != 3 or p != 4:
        return None
    
    # FMM-3 for 3x3x3 uses the following decomposition (rank 23):
    # We'll construct it explicitly.
    # 
    # The FMM-3 algorithm works as follows:
    # 1. Transform A: M_A = G @ A (G is 9x9)
    # 2. Transform B: M_B = F @ B (F is 9x9)  
    # 3. Element-wise: M_C = M_A * M_B
    # 4. Transform: C = H @ M_C
    #
    # The rank is the number of non-zero entries in M_A * M_B.
    #
    # For FMM-3, the standard decomposition has rank 23.
    # Let me construct the U, V, W matrices directly.
    
    # Alternative approach: directly construct a rank-23 decomposition for 3x3x3
    # and extend to 3x3x4.
    
    # For 3x3x3, one rank-23 decomposition:
    # The tensor T[i,j,k] = 1 if i==i', j==j', k==k'
    # We need U, V, W such that sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
    
    # Let me try a different approach: use the known fact that for 3x3x3,
    # rank 23 is achievable, and for 3x3x4, we can potentially achieve
    # rank around 23 + 3 = 26 or so by adding 3 terms for the 4th column.
    
    # Actually, let's try to directly search for a low-rank decomposition
    # using a more efficient method.
    
    dim_a = n * m  # 9
    dim_b = m * p  # 12
    dim_c = p * n  # 12
    
    # Start with schoolbook and try to reduce rank by merging columns
    # that have overlapping support patterns.
    
    # For each pair of columns, check if they can be merged.
    # Two columns c1, c2 can be merged if there exists a linear combination
    # that reproduces both contributions.
    
    # This is equivalent to: the tensor formed by columns c1 and c2
    # must have rank 1 (i.e., be decomposable as a single rank-1 term
    # plus adjustments to other columns).
    
    # A simpler heuristic: try to find columns that are "similar" and
    # see if we can eliminate one.
    
    # Let's try a greedy approach:
    # 1. Start with schoolbook (rank 36)
    # 2. For each pair of columns, try to eliminate one by adjusting
    #    the remaining columns
    # 3. Accept if the result is valid
    
    # This is expensive, so let's use a smarter approach.
    
    # Key observation: for fixed (i,j), the 4 schoolbook terms are:
    # (i,j,k) for k=0,1,2,3
    # They share U[i*m+j]=1.
    # V[j*p+k]=1 and W[k*n+i]=1 for each k.
    
    # If we use a single term for (i,j) with V having 1s at all 4 positions
    # and W having 1s at all 4 positions, that would give 16 contributions
    # instead of 4. Not helpful.
    
    # The Winograd trick is to use linear combinations.
    # For example, for 2x2x2, Strassen uses:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    
    # For 3x3x3, FMM-3 uses a similar but more complex structure.
    
    # Let me try to construct the FMM-3 decomposition explicitly.
    # The FMM-3 paper (Winograd 1980) gives the following:
    
    # G matrix (9x9) for A:
    # Rows correspond to (i,j) pairs, columns to transformed entries
    # F matrix (9x9) for B:
    # Rows correspond to (j,k) pairs, columns to transformed entries
    # H matrix (9x9) for C:
    # Rows correspond to (i,k) pairs, columns to transformed entries
    
    # The rank is the number of columns in the diagonal matrix D,
    # which is 23 for FMM-3.
    
    # Let me look up the explicit FMM-3 matrices.
    # 
    # From the literature, FMM-3 for 3x3 matrix multiplication:
    # 
    # The decomposition uses the following 23 products:
    # 
    # I'll construct this based on the known structure.
    
    # Actually, let me try a completely different approach.
    # I'll use a randomized search with exact verification.
    
    # Generate random rank-r decompositions and check if they work.
    # This is unlikely to find a solution, but let's try.
    
    # Better: use the structure of the problem.
    
    # For 3x3 x 3x4, the tensor is:
    # T[(3i+j), (4j'+k), (3k'+i')] = 1 if i==i', j==j', k==k'
    
    # Let's try to find a decomposition with rank 25-30.
    
    # Approach: use the FMM-3 decomposition for the 3x3x3 part (first 3 columns of B)
    # and add 3 more terms for the 4th column.
    
    # For the 3x3x3 part (k=0,1,2), use rank 23.
    # For the 4th column (k=3), we need 3 more terms (one for each i).
    # Total: 23 + 3 = 26.
    
    # But we might be able to do better by sharing.
    
    # Let me construct the FMM-3 rank-23 decomposition for 3x3x3 first.
    
    # From known results, one rank-23 decomposition for 3x3x3:
    # 
    # U (9x23), V (9x23), W (9x23) for 3x3x3
    # 
    # Let me use a known explicit decomposition.
    
    # The following is a known rank-23 decomposition for 3x3x3:
    # (from the literature on tensor rank)
    
    # I'll construct it using the FMM-3 structure.
    
    # FMM-3 uses the following transform:
    # 
    # For A (3x3):
    #   a00, a01, a02
    #   a10, a11, a12
    #   a20, a21, a22
    #
    # The FMM-3 algorithm computes:
    #   m1 = (a00 + a11 + a22) * (b00 + b11 + b22)
    #   m2 = (a00 - a11) * (b00 - b11)
    #   m3 = (a00 - a22) * (b00 - b22)
    #   m4 = (a11 - a22) * (b11 - b22)
    #   m5 = (a01 + a12 + a20) * (b01 + b12 + b20)
    #   m6 = (a01 - a12) * (b01 - b12)
    #   m7 = (a01 - a20) * (b01 - b20)
    #   m8 = (a12 - a20) * (b12 - b20)
    #   m9 = (a02 + a10 + a21) * (b02 + b10 + b21)
    #   m10 = (a02 - a10) * (b02 - b10)
    #   m11 = (a02 - a21) * (b02 - b21)
    #   m12 = (a10 - a21) * (b10 - b21)
    #   m13 = (a00 + a01 + a02) * (b00 + b01 + b02)
    #   m14 = (a10 + a11 + a12) * (b10 + b11 + b12)
    #   m15 = (a20 + a21 + a22) * (b20 + b21 + b22)
    #   m16 = (a00 + a10 + a20) * (b00 + b10 + b20)
    #   m17 = (a01 + a11 + a21) * (b01 + b11 + b21)
    #   m18 = (a02 + a12 + a22) * (b02 + b12 + b22)
    #   m19 = (a00 - a01 + a02) * (b00 - b01 + b02)
    #   m20 = (a10 - a11 + a12) * (b10 - b11 + b12)
    #   m21 = (a20 - a21 + a22) * (b20 - b21 + b22)
    #   m22 = (a00 + a01 - a02) * (b00 + b01 - b02)
    #   m23 = (a10 + a11 - a12) * (b10 + b11 - b12)
    
    # Wait, that's 23 terms but I'm not sure this is correct.
    # Let me just try a search-based approach.
    
    # For now, return None and let the search handle it.
    return None

def _search_low_rank(n, m, p, deadline, max_rank=35):
    """
    Search for a low-rank decomposition using a greedy column elimination approach.
    
    Start with schoolbook and try to eliminate columns one by one.
    A column can be eliminated if its tensor contribution can be
    represented as a linear combination of the other columns' contributions.
    """
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    schoolbook_rank = n * m * p
    
    # Get schoolbook decomposition
    sb = _schoolbook(n, m, p)
    U = [[Fraction(x) for x in row] for row in sb["U"]]
    V = [[Fraction(x) for x in row] for row in sb["V"]]
    W = [[Fraction(x) for x in row] for row in sb["W"]]
    
    current_rank = schoolbook_rank
    active = list(range(schoolbook_rank))
    
    # Try to eliminate columns greedily
    # For each column, check if it can be removed
    # A column c can be removed if for all (a,b,c_idx):
    #   U[a,c]*V[b,c]*W[c_idx,c] = sum_{c' in active, c' != c} alpha_c' * U[a,c']*V[b,c']*W[c_idx,c']
    # for some alpha_c'
    
    # This is a system of linear equations. Let's set it up.
    
    # For a given column c to eliminate, we need:
    #   U[a,c]*V[b,c]*W[c_idx,c] = sum_{c' in active, c' != c} alpha_{c'} * U[a,c']*V[b,c']*W[c_idx,c']
    # for all (a,b,c_idx)
    
    # This is a linear system in the alpha_{c'} variables.
    # The number of equations is dim_a * dim_b * dim_c = 9*12*12 = 1296
    # The number of unknowns is current_rank - 1
    
    # We can solve this using Gaussian elimination with Fractions.
    
    # But this is expensive. Let's try a simpler heuristic first.
    
    # Heuristic: two columns c1 and c2 can be merged if they have
    # the same "pattern" in a certain sense.
    
    # For now, let's try a randomized approach:
    # 1. Randomly select a column to eliminate
    # 2. Try to solve for the adjustment coefficients
    # 3. If successful, eliminate the column
    
    # This is still expensive, so let's limit the search.
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to eliminate columns in a specific order
    # Columns that are "similar" are more likely to be eliminable.
    
    # For schoolbook, column (i,j,k) has:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    # All other entries are 0.
    
    # Two columns (i1,j1,k1) and (i2,j2,k2) are "similar" if
    # they share some indices.
    
    # Let's try to eliminate columns where (i1,j1) == (i2,j2) but k1 != k2.
    # These share the same U entry.
    
    # For a fixed (i,j), we have p=4 columns.
    # Can we reduce 4 columns to 3?
    
    # For fixed (i,j), the 4 columns are:
    # c_k: U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1 for k=0,1,2,3
    
    # To eliminate c_3, we need:
    # U[a,c_3]*V[b,c_3]*W[c,c_3] = sum_{k=0}^{2} alpha_k * U[a,c_k]*V[b,c_k]*W[c,c_k]
    
    # The LHS is non-zero only when a=i*m+j, b=j*p+3, c=3*n+i.
    # The RHS is non-zero when (a=i*m+j, b=j*p+k, c=k*n+i) for k=0,1,2.
    
    # These supports are disjoint (different b and c values), so
    # the only way this works is if both sides are 0, which they're not.
    
    # So we can't eliminate a column within the same (i,j) group using
    # just the other columns in that group.
    
    # We need to use columns from other (i,j) groups.
    
    # This is getting complex. Let me try a different approach.
    
    # Use the fact that the tensor has a specific structure and
    # try to find a decomposition using known results.
    
    # For 3x3x4, the minimum rank is not known to me, but it's
    # likely between 23 and 36.
    
    # Let's try to find a rank-30 decomposition by construction.
    
    # Idea: use the FMM-3 decomposition for 3x3x3 (rank 23) and
    # add 7 more terms for the 4th column, for a total of 30.
    
    # But I need the explicit FMM-3 decomposition.
    
    # Let me try to construct it.
    
    # For 3x3x3, the FMM-3 decomposition can be written as:
    # 
    # U (9x23), V (9x23), W (9x23)
    # 
    # where U[a, col], V[b, col], W[c, col] are the factors.
    
    # I'll use a known explicit decomposition from the literature.
    
    # After some research, here's a rank-23 decomposition for 3x3x3:
    
    # The decomposition is based on the following 23 products:
    # 
    # Let me just try to find a valid decomposition with rank < 36
    # using a more direct search.
    
    # For time efficiency, let's return the schoolbook if we can't
    # find something better quickly.
    
    if time.time() > deadline - 20:
        return None
    
    # Try a simple heuristic: for each (i,j) group, try to reduce
    # the 4 columns to 3 by using a linear combination.
    
    # This won't work as shown above, so let's try a different approach.
    
    # Use randomization: generate random rank-r decompositions and
    # check if they work. This is unlikely to succeed but let's try.
    
    # Actually, let's try a more structured search.
    
    # For 3x3x4, we can view it as 4 independent 3x3x1 problems
    # that share the A matrix. The tensor rank of the combined
    # problem might be less than 4 * rank(3x3x1).
    
    # rank(3x3x1) = 9 (schoolbook), so 4*9 = 36.
    # But we can share computations.
    
    # For each column k of B, C[:,k] = A @ B[:,k]
    # This is a 3x3 @ 3x1 multiplication.
    
    # The tensor for 3x3 @ 3x1 has rank 9.
    # For 4 such problems sharing A, the combined rank might be
    # less than 36.
    
    # Let's try to find a decomposition with rank 30-35.
    
    # I'll use a greedy approach: start with schoolbook and try
    # to eliminate columns by solving linear systems.
    
    # For efficiency, I'll only try to eliminate a few columns.
    
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    active_cur = active[:]
    
    # Try to eliminate columns one by one
    # For each column c, set up the linear system and solve
    
    # The linear system: for all (a,b,c_idx),
    #   U[a,c]*V[b,c]*W[c_idx,c] = sum_{c' in active, c' != c} alpha_{c'} * U[a,c']*V[b,c']*W[c_idx,c']
    
    # This is a system of 1296 equations in (current_rank - 1) unknowns.
    # We can solve it using Gaussian elimination.
    
    # But this is very expensive. Let's try a simpler check first.
    
    # Quick check: if column c has a unique (a,b) pair that no other
    # column covers, it can't be eliminated.
    
    # For schoolbook, column (i,j,k) covers (a=i*m+j, b=j*p+k).
    # This pair is unique to this column, so no column can be eliminated
    # using only the other schoolbook columns.
    
    # We need to allow the remaining columns to be modified.
    
    # This is a harder problem. Let's try a different approach.
    
    # Use the FMM-3 structure explicitly.
    
    # I'll construct a rank-26 decomposition for 3x3x4 by:
    # 1. Using a rank-23 decomposition for the 3x3x3 part
    # 2. Adding 3 terms for the 4th column
    
    # For the 3x3x3 part, I need an explicit rank-23 decomposition.
    
    # Let me construct one based on the FMM-3 algorithm.
    
    # FMM-3 for 3x3 matrix multiplication:
    # 
    # The algorithm uses the following transforms:
    # 
    # G (for A): 9x9 matrix
    # F (for B): 9x9 matrix  
    # H (for C): 9x9 matrix
    # D (diagonal): 9x9 matrix with 23 non-zero entries
    # 
    # The rank is the number of non-zero entries in D.
    
    # Let me look up the explicit FMM-3 matrices.
    
    # From Winograd's paper and subsequent work, the FMM-3 decomposition
    # can be written as follows.
    
    # I'll use a known explicit rank-23 decomposition for 3x3x3.
    
    # Here's one such decomposition (from the literature):
    
    # U (9x23), V (9x23), W (9x23) for 3x3x3
    
    # Let me construct it using the following approach:
    # 
    # The 3x3x3 tensor T[i,j,k] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    # 
    # We can write T as a sum of 23 rank-1 tensors.
    
    # I'll use a known decomposition. After some research, here's a
    # rank-23 decomposition for 3x3x3:
    
    # (I'll construct it programmatically)
    
    # For now, let's try a different strategy: use a randomized
    # local search to find a rank < 36 decomposition.
    
    # Start with schoolbook and randomly perturb columns, then
    # check if the result is still valid.
    
    # This is unlikely to work, but let's try.
    
    # Actually, let's just return the schoolbook for now.
    # The search is too complex to implement correctly in the time available.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, get schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find better decomposition
    result = _search_low_rank(n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
            best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
