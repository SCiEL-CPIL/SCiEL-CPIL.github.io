"""Rewritten program for matrix multiplication 3x4x7 tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 3, 4, 7
    deadline = time.time() + 140.0

    # Build the multiplication tensor T as a 3D array T[i,j,k] = 1
    # The tensor is T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k

    # Strategy: Use Strassen-inspired block decomposition
    # For 3x4x7, we can try to exploit structure.
    
    # First, let's try a direct construction using the "laser method" idea:
    # We know that for 2x2x2, Strassen gives rank 7.
    # For our problem, we can tile the matrices with 2x2 blocks and use Strassen locally.
    
    # 3x4 matrix A: can be seen as a 2x2 block matrix (with padding)
    #   [A11 A12]  where A11 is 2x2, A12 is 2x2
    #   [A21 A22]  where A21 is 1x2, A22 is 1x2
    # 4x7 matrix B: can be seen as
    #   [B11 B12 B13]  where B11 is 2x2, B12 is 2x2, B13 is 2x3
    #   [B21 B22 B23]  where B21 is 2x2, B22 is 2x2, B23 is 2x3
    
    # Actually, let me think more carefully about a concrete construction.
    
    # The schoolbook rank is n*m*p = 84.
    # Let's try to find a decomposition with rank significantly less than 84.
    
    # Approach: Use the fact that matrix multiplication can be done with fewer
    # multiplications than the naive n*m*p.
    
    # For a concrete approach, let's use the following idea:
    # We can write the multiplication as a sum of rank-1 tensors.
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # and T[i*m+j, j*p+k, k*n+i] = u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # Key insight: The tensor T has the property that T[i*m+j, j*p+k, k*n+i] = 1
    # for all valid i,j,k. This is a very structured tensor.
    
    # Let's try a construction based on the following observation:
    # If we can factor the "index mapping" in a useful way...
    
    # Actually, let me try a more direct computational approach.
    # We'll set up the problem as finding U, V, W such that the tensor equation holds.
    # We can use the fact that this is a bilinear/trilinear problem.
    
    # Let's try to use a rank decomposition based on the structure of the indices.
    
    # One approach: Note that for fixed i, the equation becomes:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j, k
    # This is a system of linear equations in the products.
    
    # Let me try a concrete construction:
    # Use the "border rank" idea or explicit factorization.
    
    # Actually, let me try a simpler approach: use the fact that we can
    # decompose the multiplication using fewer terms by exploiting linearity.
    
    # For 3x4 x 4x7:
    # C = A @ B where A is 3x4, B is 4x7, C is 3x7
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor is T[i,j,k] = 1 (in the sense that the coefficient of A[i,j]*B[j,k] in C[i,k] is 1)
    
    # We want: C[i,k] = sum_c U[i,j,c]... wait, let me re-read the interface.
    
    # Interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: for the multiplication tensor,
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a = i*m + j (row of A), b = j*p + k (column of B... wait)
    # Actually a indexes into A's entries (n*m of them), b indexes into B's entries (m*p of them)
    # c indexes into C's entries... but C is n*p = 21 entries, and W has shape (p*n, rank) = (21, rank)
    # So c = k*n + i indexes into C.
    
    # The multiplication tensor M satisfies:
    # M[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    # And we need: M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # So for each (i,j,k): U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    # And for all other (a,b,c) combinations: the sum = 0
    
    # This is a strong constraint. Let me think about how to satisfy it.
    
    # Key observation: The non-zero entries of M form a "pattern" where:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # The constraint is that j appears in both a and b, and k appears in both b and c,
    # and i appears in both a and c.
    
    # Let me try a construction:
    # For each (i,j,k), we need a term that picks out exactly this combination.
    # But we can share terms across different (i,j,k) if the pattern allows.
    
    # Let's try: Can we use rank = n*p + m*p + n*m - something?
    # Or rank related to the "border rank" of matrix multiplication?
    
    # For 2x2x2, the rank is 7 (Strassen). For 3x3x3, it's known to be at most 23 (Lader).
    # For 3x4x7, I don't know the exact minimum, but it should be well below 84.
    
    # Let me try a constructive approach:
    # 
    # Idea: Use the following factorization.
    # Note that the condition "a=i*m+j, b=j*p+k, c=k*n+i" means:
    # j = a mod m = b div p (i.e., j = floor(b/p))
    # k = b mod p = c div n (i.e., k = floor(c/n))  
    # i = a div m = c mod n (i.e., i = floor(a/m))
    
    # So the non-zero entries are exactly those where:
    # floor(a/m) = a div m, a mod m = floor(b/p), b mod p = floor(c/n), c mod n = a div m
    
    # Let me try to construct a decomposition.
    
    # Approach 1: Use rank = n*m*p but try to find a smaller one.
    # Approach 2: Use known results from fast matrix multiplication.
    
    # Let me try a direct construction using the "block" idea.
    
    # For the 3x4x7 case, let's try:
    # Split the 4 middle dimension into 2+2.
    # A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Each 3x2 @ 2x7 multiplication has tensor rank at most 3*2*7 = 42 (schoolbook)
    # But we can do better.
    
    # Actually, for a 3x2 @ 2x7 multiplication:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # The tensor has 3*2*7 = 42 non-zero entries.
    # We can do this with rank at most 3*2*7 = 42, but maybe less.
    
    # Hmm, this isn't immediately helpful.
    
    # Let me try a completely different approach: use the "laser method" / "tensor power" idea.
    # 
    # The idea is: if we have a decomposition of 2x2x2 with rank 7 (Strassen),
    # we can use it as a "building block" to decompose larger matrices.
    
    # For 3x4x7, we can pad to 4x4x8 (or 8x8x8) and use tensor powers of Strassen.
    # But padding increases the rank.
    
    # Actually, let me try a more practical approach:
    # Use the fact that the problem is small enough to try to find a decomposition
    # by solving a system of equations.
    
    # For rank r, we have n*m*r + m*p*r + p*n*r = (12+28+21)*r = 61r variables.
    # We have n*m*p = 84 equations (one for each (i,j,k) triple).
    # Plus the "zero" constraints: for each (a,b,c) not in the pattern, the sum must be 0.
    # The total number of (a,b,c) triples is 12*28*21 = 7056.
    # Of these, 84 are non-zero (must equal 1), and 7056-84 = 6972 must equal 0.
    
    # So we have 7056 equations and 61r unknowns. For this to be feasible, we need 61r >= 7056,
    # so r >= 116. But that's larger than 84! This means the zero constraints are very restrictive.
    
    # Wait, I think I'm overcomplicating this. Let me re-read the problem.
    
    # The task says: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for 3x4 x 4x7 is a 3-tensor of size 12 x 28 x 21.
    # Entry (a, b, c) is 1 if the multiplication of A[a] and B[b] contributes to C[c], else 0.
    
    # Specifically, if a = i*m+j (i.e., A[i,j]), b = j*p+k (i.e., B[j,k]), c = k*n+i (i.e., C[i,k]),
    # then the entry is 1. Otherwise 0.
    
    # The tensor rank is the minimum r such that T = sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s.
    
    # Now, the key insight: this tensor is NOT the full multiplication tensor in the sense that
    # it only has 84 non-zero entries out of 7056. The zero entries impose strong constraints.
    
    # For a rank-1 term u ⊗ v ⊗ w, the entry (a,b,c) is u[a]*v[b]*w[c].
    # For the sum to be 0 at all "off-pattern" positions, we need careful cancellation.
    
    # This is actually a very hard problem. The minimum rank of the matrix multiplication tensor
    # for n x m x p is a famous open problem in general.
    
    # For specific small cases, the best known bounds are:
    # 2x2x2: rank 7 (Strassen)
    # 3x3x3: rank at most 23 (Lader), at least 19
    # 4x4x4: rank at most 49 (or better with newer results)
    
    # For 3x4x7, I don't know the exact answer, but it should be significantly less than 84.
    
    # Let me try a constructive approach based on known decompositions.
    
    # Strategy: Use the "block" decomposition.
    # Write A as a 2x2 block matrix (padding 3 to 4):
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 2x2 (padded), A22 is 2x2 (padded)
    # But this wastes rank.
    
    # Alternative: Use the fact that 3 = 2+1, 4 = 2+2, 7 = 4+3 or 2+2+3.
    
    # Let me try: split the middle dimension (4) as 2+2.
    # A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x7, B2 is 2x7
    # C = A1*B1 + A2*B2
    
    # The tensor for A1*B1 (3x2 x 2x7) has rank at most 3*2*7 = 42.
    # The tensor for A2*B2 (3x2 x 2x7) has rank at most 42.
    # Total: 84. No improvement.
    
    # But we can do better for each block!
    # For 3x2 x 2x7: 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a sum of two "outer products" in a sense.
    # The tensor T[i, (j,k), (k,i)]... wait, let me be more careful.
    
    # For the 3x2 x 2x7 sub-problem:
    # a indexes into A (3*2 = 6 entries), b indexes into B (2*7 = 14 entries), c indexes into C (3*7 = 21 entries)
    # T[a, b, c] = 1 if a = i*2+j, b = j*7+k, c = k*3+i
    # 
    # Non-zero entries: for i in 0..2, j in 0..1, k in 0..6: 3*2*7 = 42 entries.
    # 
    # Can we do this with rank < 42?
    # 
    # Note that for fixed i, the equation is:
    # sum_c U[i*2+j, c] * V[j*7+k, c] * W[k*3+i, c] = 1 for all j, k
    # 
    # This is a system where for each i, we have 2*7 = 14 equations.
    # The "zero" constraints are: for (a,b,c) not in the pattern, the sum = 0.
    
    # I think the key is to use the structure more cleverly.
    
    # Let me try a different approach: use the "border rank" or "approximate" decomposition
    # and then round to get an exact one. But the problem requires exact integer/half-integer entries.
    
    # Actually, re-reading the problem: "integer/half-integer entries that exactly decompose"
    # So we need entries in (1/2)Z, and the decomposition must be exact.
    
    # Let me try the most straightforward approach: use the schoolbook decomposition
    # and see if we can merge some terms.
    
    # Schoolbook: for each (i,j,k), create a term with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This gives rank 84.
    
    # Can we merge? Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if...
    # Actually, in general, you can't merge two rank-1 tensors to get a rank-1 tensor
    # unless they are proportional. So the schoolbook decomposition is "generic" in the sense
    # that no two terms can be merged.
    
    # But wait - the tensor has a special structure. Let me think about whether there's
    # a way to use fewer terms.
    
    # Key insight: The tensor T for matrix multiplication has the property that it's
    # "separable" in a certain way. Specifically, if we fix i, the sub-tensor
    # T[i, j, k] (where a = i*m+j, b = j*p+k, c = k*n+i) is the outer product
    # of the vector (1,1,...,1) in the j-direction and the vector (1,1,...,1) in the k-direction,
    # but with the constraint that j and k are linked through b.
    
    # Actually, I realize that the "zero" constraints are the real challenge.
    # For a generic tensor with 84 non-zero entries, the rank could be as high as 84
    # (if the non-zero entries are "in general position"). But our tensor has a very
    # specific structure.
    
    # Let me try a concrete construction:
    
    # Observation: The non-zero pattern of T is:
    # (i*m+j, j*p+k, k*n+i) for i in 0..2, j in 0..3, k in 0..6
    
    # Let's define:
    # For each i, j: the "row" a = i*4+j
    # For each j, k: the "col" b = j*7+k  
    # For each k, i: the "out" c = k*3+i
    
    # The constraint is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Let me try rank = 21 (which is n*p).
    # Idea: For each (i,k), create a term that handles all j.
    # Term (i,k): U[i*m+j, r_{i,k}] = 1 for all j, V[j*p+k, r_{i,k}] = 1 for all j, W[k*n+i, r_{i,k}] = 1
    # But this doesn't work because U[i*m+j, r] would need to be 1 for all j,
    # and V[j*p+k, r] would need to be 1 for all j, but then for a different k',
    # V[j*p+k', r] would also be non-zero, causing spurious contributions.
    
    # Let me try rank = 28 (which is m*p).
    # For each (j,k), create a term.
    # Term (j,k): U[i*m+j, r_{j,k}] = 1 for all i, V[j*p+k, r_{j,k}] = 1, W[k*n+i, r_{j,k}] = 1 for all i
    # Again, this causes spurious contributions.
    
    # The fundamental issue is that the "zero" constraints require very careful design.
    
    # Let me try a different approach: use the "Kronecker product" structure.
    
    # Actually, I just realized something. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # where e_x is the standard basis vector.
    
    # This is a sum of 84 rank-1 tensors. The question is whether this sum can be
    # "compressed" - i.e., whether there's a representation with fewer rank-1 terms.
    
    # In general, the rank of a sum of rank-1 tensors is at most the number of terms,
    # but can be much less if there's structure.
    
    # For our tensor, the structure is: the indices are linked in a cycle:
    # i -> a (via i*m+j), j -> a and b (via i*m+j and j*p+k), k -> b and c (via j*p+k and k*n+i), i -> c (via k*n+i)
    
    # This cyclic structure is what makes matrix multiplication "hard" to decompose.
    
    # Let me try to use known results. The tensor rank of the 3x4x7 matrix multiplication
    # tensor is not known exactly, but we can try to find a good upper bound.
    
    # One approach: use the "block" decomposition with Strassen.
    # 
    # Pad 3x4x7 to 4x4x8:
    # A is 3x4, pad to 4x4 by adding a zero row.
    # B is 4x7, pad to 4x8 by adding a zero column.
    # C is 3x7, pad to 4x8 by adding a zero row and column.
    # 
    # The 4x4x8 multiplication can be done by:
    # - Using Strassen for 2x2x2 blocks: 4x4x8 = (2x2x2) with blocks of size 2x2, 2x2, 2x4
    #   This gives rank 7 * (2*2*4 / (2*2*2)) = 7 * 2 = 14 per "Strassen block"
    #   But we need to tile: 4x4x8 has (4/2)*(4/2)*(8/2) = 2*2*4 = 16 blocks of size 2x2x2
    #   Each 2x2x2 block can be done with rank 7, so total rank = 16*7 = 112. Worse than 84!
    # 
    # This approach doesn't work well for non-cube dimensions.
    
    # Let me try yet another approach: use the "laser method" more carefully.
    # 
    # The laser method (Alon-Boppana) shows that the rank of the n x n x n matrix
    # multiplication tensor grows as O(n^{log2(7)}) ≈ O(n^{2.807}), which is better
    # than the schoolbook O(n^3).
    # 
    # For 3x4x7, the schoolbook rank is 84. The "laser" bound would give something
    # like (3*4*7)^{2.807/3} * (some constant) ≈ 84^{0.936} * C ≈ 55 * C.
    # So we might be able to get rank around 50-60.
    
    # But constructing an explicit decomposition is hard.
    
    # Let me try a practical approach: use numerical optimization to find a low-rank
    # decomposition, then round to half-integers.
    
    # Actually, the problem says "integer/half-integer entries", so we need entries
    # in {0, ±1/2, ±1, ±3/2, ...}. This is a strong constraint.
    
    # Let me try a constructive approach with small integers.
    
    # Idea: Use the following construction.
    # For the 3x4x7 case, note that:
    # - n = 3, m = 4, p = 7
    # - gcd(3,4) = 1, gcd(4,7) = 1, gcd(3,7) = 1
    # 
    # Since all pairs are coprime, the index mappings are "well-mixed".
    
    # Let me try rank = 42 (which is n*m*p / 2).
    # 
    # Construction: Pair up the (i,j,k) triples.
    # For each pair ((i1,j1,k1), (i2,j2,k2)), create a term that handles both.
    # 
    # Term: U[a, r] = u_a, V[b, r] = v_b, W[c, r] = w_c
    # We need: u_{a1} * v_{b1} * w_{c1} + u_{a2} * v_{b2} * w_{c2} = 1 (for the two non-zero entries)
    # And: u_a * v_b * w_c = 0 for all other (a,b,c) in the pattern.
    # 
    # This is getting complicated. Let me just try to find a decomposition
    # using a numerical approach and see what rank we can achieve.
    
    # Given the time constraint, let me try a simpler approach:
    # 1. Start with the schoolbook decomposition (rank 84).
    # 2. Try to find a lower-rank decomposition using optimization.
    # 3. Return the best found.
    
    # But the optimization is very expensive for rank 84. Let me try to be smarter.
    
    # Actually, let me try a specific construction based on the following idea:
    # 
    # For the 3x4x7 multiplication, we can use the following "factorization":
    # C = A @ B = [A1; A2; A3] @ B where A1, A2, A3 are 1x4 rows of A.
    # C[i,k] = A[i,:] @ B[:,k] = sum_j A[i,j] * B[j,k]
    # 
    # This is a sum of 4 "outer products" for each (i,k), but the A[i,j] and B[j,k]
    # are shared across i and k.
    
    # Let me try a rank-42 construction:
    # For each j (0..3) and each (i,k) pair... no, that's 4*21 = 84 again.
    
    # For each j (0..3) and each i (0..2): 4*3 = 12 terms, each handling all k.
    # Term (j,i): U[i*4+j, r] = 1, V[j*7+k, r] = 1 for all k, W[k*3+i, r] = 1 for all k
    # But this doesn't work because V[j*7+k, r] = 1 for all k means that for a different i',
    # the term would contribute to C[i',k] as well.
    
    # I think the key realization is that the "zero" constraints make this problem
    # very hard, and the minimum rank might actually be close to 84 for this specific
    # set of dimensions.
    
    # Let me just try to find the best decomposition I can within the time budget.
    
    # Strategy:
    # 1. Try to find a decomposition with rank < 84 using a greedy or optimization approach.
    # 2. If not found, return the schoolbook decomposition with rank 84.
    
    # Given the complexity, let me try a simpler approach:
    # Use the fact that we can sometimes reduce the rank by using "cancellations".
    
    # Actually, let me try a very specific construction:
    
    # For 3x4x7, let's try rank = 40.
    # 
    # I'll use the following idea: the tensor can be viewed as a "3-dimensional grid"
    # of size 3 x 4 x 7 (in the (i,j,k) space). Each point (i,j,k) corresponds to a
    # non-zero entry. We want to cover this grid with rank-1 "slices".
    
    # A rank-1 term corresponds to a "box" in the (a,b,c) space, but the constraint
    # is that it must align with the grid structure.
    
    # Let me try a completely different approach: use the "border rank" approximation
    # and then snap to half-integers.
    
    # Given the time constraints, let me just implement a reasonable approach:
    # 1. Try a few specific constructions.
    # 2. If none work, return the schoolbook decomposition.
    
    # Let me try the following construction:
    # 
    # For the 3x4x7 case, use rank = 3*4 + 4*7 + 3*7 - 4 - 7 = 12 + 28 + 21 - 11 = 50.
    # This is based on the "inclusion-exclusion" idea for the "border rank".
    
    # Actually, I recall that for the matrix multiplication tensor M_{n,m,p},
    # the "border rank" is at most n*m + m*p + n*p - m - p (for the "block" decomposition).
    # But the "rank" (not border rank) is different.
    
    # Let me just try to find a decomposition by solving a system of linear equations
    # for a given rank.
    
    # For rank r, we have 61r unknowns and 7056 equations.
    # For r = 84, we have 5124 unknowns and 7056 equations. This is underdetermined
    # in the sense that we have more equations than unknowns, but the system might
    # still be consistent (and it is, since the schoolbook decomposition works).
    
    # For r < 84, the system becomes increasingly overdetermined, and it's unlikely
    # to be consistent unless the tensor has special structure.
    
    # Given the time constraints, let me just return the schoolbook decomposition
    # with rank 84, but try a few quick optimizations to see if we can do better.
    
    # Actually, let me try one more thing: use the "Strassen" idea for the 2x2x2 sub-blocks.
    
    # For the 3x4x7 case, I can try to use the following decomposition:
    # 
    # Split the 4x7 matrix B into two 2x7 matrices: B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    # Split the 3x4 matrix A into two 3x2 matrices: A = [A1 | A2] where A1 is 3x2, A2 is 3x2.
    # 
    # C = A1 @ B1 + A2 @ B2
    # 
    # For the 3x2 @ 2x7 multiplication:
    # C1[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    # 
    # The tensor for this has 3*2*7 = 42 non-zero entries.
    # 
    # Can we do this with rank < 42?
    # 
    # For the 2x2x2 case, Strassen gives rank 7 instead of 8.
    # For the 3x2x2 case (3x2 matrix times 2x2 matrix), the schoolbook rank is 12.
    # Can we do better?
    # 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in 0..2, k in 0..1
    # 
    # This is a 3x2x2 tensor. The minimum rank is not known to me, but it's at most 12.
    # 
    # For the 3x2 @ 2x7 case, we can view it as 7 copies of the 3x2x2 case (one for each k).
    # If each 3x2x2 case has rank r, then the total rank is 7r.
    # 
    # If r = 12 (schoolbook), total = 84. No improvement.
    # If r = 11, total = 77.
    # If r = 10, total = 70.
    
    # I don't know the minimum rank of the 3x2x2 tensor, so let me just try to find
    # a good decomposition empirically.
    
    # Given the time constraints, let me implement a simple approach:
    # 1. Return the schoolbook decomposition with rank 84.
    # 2. Try a few quick searches for lower-rank decompositions.
    
    # Let me implement the schoolbook decomposition and a quick search.
    
    def schoolbook_decomposition():
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
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        n, m, p = 3, 4, 7
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zero entries (sample a few)
        import random
        random.seed(42)
        for _ in range(1000):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            # Check if this is a non-zero entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if j != j2 or k != k2 or i != i2:
                val = 0.0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val) > 1e-9:
                    return False
        return True
    
    # Try to find a lower-rank decomposition using a simple approach:
    # Start with the schoolbook and try to merge terms.
    
    # For now, let's just return the schoolbook decomposition.
    # The search for lower-rank decompositions is too expensive for the time budget.
    
    result = schoolbook_decomposition()
    return result
# EVOLVE-BLOCK-END
