"""
Matrix multiplication 4x4x5 rank-minimal decomposition.

Strategy: Use a Kronecker-Strassen hybrid approach.
- Decompose the 4x4x5 multiplication into blocks.
- Use Strassen's algorithm for the 4x4x4 part (rank 28 per 4x4x4 block).
- Handle the 4x4x1 part with schoolbook (rank 16).
- Total target rank: 28 + 16 = 44.

Actually, let's think more carefully. We're multiplying a 4x4 matrix by a 4x5 matrix.
We can view this as multiplying a 4x4 matrix by five 4x1 column vectors, but that's not how it decomposes nicely.

Better approach: Use the known optimal rank for small matrix multiplications.
For 4x4x4, the Strassen-based rank is 28 (not 27, since 27 is for 2x2x2 composed via Kronecker, but 4x4x4 has rank at most 28 via Strassen).
Actually, 4x4x4 can be done with rank 28 using Strassen on 2x2 blocks.

Let me construct this properly.

For 4x4x4 multiplication:
- Split each 4x4 matrix into four 2x2 blocks.
- Use Strassen's 7 multiplications for the 2x2x2 case.
- This gives rank 28 for 4x4x4 (7 * 4 = 28, since each 2x2x2 Strassen multiplication decomposes into 7 rank-1 terms, and we have 4 such block positions... wait, no).

Actually, Strassen for 2x2x2 has rank 7. When we use Kronecker product to get 4x4x4, we use the 2x2x4... no, that's not right either.

Let me think again. The standard way:
- 2x2x2 has rank 7 (Strassen).
- 4x4x4 can be done with rank 28 by treating it as 4 block-multiplications of 2x2x2, each with rank 7, so 4*7 = 28.

For 4x4x5:
- We can split the 5 into 4+1.
- The 4x4x4 part has rank 28.
- The 4x4x1 part: multiplying a 4x4 matrix by a 4x1 matrix. This is just 4 dot products of length 4, which is 4*4 = 16 rank-1 terms.
- Total: 28 + 16 = 44.

Let me implement this.
"""

import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 5]

def _fraction_to_halfint(frac):
    """Convert a Fraction to a half-integer float or int if possible."""
    # Check if it's an integer
    if frac.denominator == 1:
        return int(frac.numerator)
    elif frac.denominator == 2:
        return frac.numerator / 2.0
    else:
        # Convert to float, but ensure it's a half-integer
        return float(frac)

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build the decomposition using Strassen for the 4x4x4 part
    # and schoolbook for the 4x4x1 part.
    
    # First, let's define Strassen's 2x2x2 decomposition
    # For multiplying 2x2 matrices A * B = C:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B22)
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen coefficients for 2x2x2
    # U coefficients (for A matrix, 2x2=4 entries): (a11, a12, a21, a22)
    # V coefficients (for B matrix, 2x2=4 entries): (b11, b12, b21, b22)
    # W coefficients (for C matrix, 2x2=4 entries): (c11, c12, c21, c22)
    
    strassen_2x2 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1
        (Fraction(1), Fraction(0), Fraction(0), Fraction(1),
         Fraction(1), Fraction(0), Fraction(0), Fraction(1),
         Fraction(1), Fraction(0), Fraction(0), Fraction(1)),
        # M2
        (Fraction(0), Fraction(0), Fraction(1), Fraction(1),
         Fraction(1), Fraction(0), Fraction(0), Fraction(0),
         Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        # M3
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0),
         Fraction(0), Fraction(0), Fraction(1), Fraction(-1),
         Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        # M4
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1),
         Fraction(1), Fraction(1), Fraction(0), Fraction(-1),
         Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        # M5
        (Fraction(1), Fraction(1), Fraction(0), Fraction(0),
         Fraction(0), Fraction(0), Fraction(0), Fraction(1),
         Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        # M6
        (Fraction(-1), Fraction(0), Fraction(1), Fraction(0),
         Fraction(1), Fraction(1), Fraction(0), Fraction(0),
         Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        # M7
        (Fraction(0), Fraction(1), Fraction(0), Fraction(-1),
         Fraction(0), Fraction(0), Fraction(1), Fraction(1),
         Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
    ]
    
    # Now for 4x4x4, we use 4 block positions, each using Strassen
    # The 4x4 matrix is split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # For each block position (i,j) in C, we need to compute it
    # But Strassen computes all 4 blocks simultaneously with 7 multiplications.
    
    # So for 4x4x4, we use the same Strassen structure but with 2x2 blocks.
    # This means each "scalar" in the Strassen formula is actually a 2x2 matrix
    # multiplication, which itself needs rank 4 (schoolbook) or we can recursively use Strassen.
    
    # Wait, I'm overcomplicating this. Let me think of it differently.
    
    # The rank of 4x4x4 is at most 28. Here's how:
    # We can write 4x4x4 as a tensor product of 2x2x2 with itself? No, that would give rank 7*7=49.
    
    # Actually, the correct way: 4x4x4 can be done with rank 28 by using
    # the fact that it's equivalent to 4 independent 2x2x2 multiplications,
    # each of rank 7, giving 4*7 = 28.
    
    # But wait, that's not how it works. The 4x4x4 tensor is not a direct sum of four 2x2x2 tensors.
    
    # Let me use a different approach. I'll construct the decomposition explicitly.
    
    # For 4x4x4, one known decomposition has rank 28.
    # Let me build it using the Kronecker product of the 2x2x2 Strassen decomposition.
    
    # Actually, I think the simplest approach is to just use schoolbook for now
    # and see if we can find a better decomposition. But the task asks for rank-minimal.
    
    # Let me try a different strategy: use the known result that 4x4x4 has rank 28.
    # I'll construct it explicitly.
    
    # For 4x4x4:
    # The multiplication tensor can be decomposed using the Strassen method applied
    # at the block level. Each 2x2x2 block multiplication has rank 7.
    # Since there are 4 such blocks in the 4x4x4 multiplication, the total rank is 4*7 = 28.
    
    # Let me construct the U, V, W matrices for 4x4x4 with rank 28.
    
    # For each of the 4 block positions (block_i, block_j) where block_i, block_j in {0,1}:
    # We use Strassen's 7 multiplications.
    
    # The input A is 4x4, indexed as A[i,j] where i,j in {0,1,2,3}
    # We can think of i = 2*block_i + sub_i, j = 2*block_j + sub_j
    # Similarly for B and C.
    
    # Let me define the Strassen decomposition for 4x4x4 explicitly.
    
    # I'll build it step by step.
    
    # First, the schoolbook decomposition for reference
    rank_schoolbook = n * m * p  # 4*4*5 = 80
    
    # Let me try to construct the rank-44 decomposition.
    
    # Part 1: 4x4x4 part (columns 0 to 3 of the 4x5 matrix)
    # Rank 28
    # Part 2: 4x4x1 part (column 4 of the 4x5 matrix)
    # Rank 16
    
    # Total rank: 44
    
    # Let me build the U, V, W matrices.
    
    # For the 4x4x4 part:
    # We use Strassen's algorithm. The 4x4 matrix A is divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly for B (the first 4 columns) and C (the output, first 4 columns).
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # But each of these M's is a 2x2x2 multiplication, which has rank 7.
    # So for each M, we need 7 rank-1 terms.
    # Total: 7 * 7 = 49? No, that's not right.
    
    # I think I'm confusing myself. Let me start over with a cleaner approach.
    
    # The rank of the 4x4x4 multiplication tensor is 28.
    # This can be achieved by using the Strassen decomposition at the 2x2 level.
    # Each 2x2x2 multiplication has rank 7.
    # The 4x4x4 multiplication can be seen as 4 independent 2x2x2 multiplications
    # (for the 4 block positions), each of rank 7, giving 4*7 = 28.
    
    # Wait, that's still not quite right. The 4x4x4 tensor is not a direct sum.
    
    # Let me just use a known construction. I'll look up or derive the rank-28 decomposition for 4x4x4.
    
    # Actually, I recall that the rank of n x n x n matrix multiplication for n=4 is 28,
    # and this is achieved by the Strassen-based algorithm.
    
    # Let me construct it explicitly.
    
    # For 4x4x4, the decomposition has rank 28.
    # I'll build the U, V, W matrices with 28 columns.
    
    # The key insight: use the Kronecker product structure.
    # 4x4x4 = (2x2x2) ⊗ (2x2x2) in some sense, but the rank is 7*7=49 for the naive Kronecker,
    # which is worse than 28.
    
    # So the rank-28 decomposition must be more clever.
    
    # Let me try a different approach: just use schoolbook for now and return rank 80,
    # then try to improve. But the task wants rank-minimal.
    
    # Actually, let me just implement the Strassen-based decomposition properly.
    
    # I found a reference: the rank of 4x4x4 is 28, and it can be constructed
    # by applying Strassen's algorithm to 2x2 blocks.
    
    # Here's the construction:
    # For the 4x4x4 multiplication, we use 28 rank-1 terms.
    # Each term corresponds to one of the 28 "Strassen sub-multiplications".
    
    # Let me define the coefficients explicitly.
    
    # I'll use the following approach:
    # 1. For the 4x4x4 part, use a rank-28 decomposition.
    # 2. For the 4x4x1 part, use a rank-16 decomposition (schoolbook).
    # 3. Total rank: 44.
    
    # Let me build this.
    
    # Part 1: 4x4x4 with rank 28
    # I'll use the Strassen decomposition.
    
    # The Strassen decomposition for 2x2x2 has 7 terms.
    # For 4x4x4, we can view it as 4 block-multiplications of 2x2x2.
    # But the blocks interact, so we need to be careful.
    
    # Actually, I think the correct way is:
    # The 4x4x4 tensor has a decomposition of rank 28, which can be obtained
    # by using the fact that 4x4x4 is equivalent to multiplying a 4x4 matrix
    # by a 4x4 matrix, and the optimal algorithm uses 28 scalar multiplications.
    
    # Let me just hardcode the rank-28 decomposition for 4x4x4.
    
    # I'll construct it using the Strassen formulas.
    
    # For 4x4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # where each block is 2x2.
    
    # Strassen's 7 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Each M is a 2x2x2 multiplication, which has rank 7.
    # So for each M, we need 7 rank-1 terms.
    # Total: 7 * 7 = 49 terms.
    
    # But that's rank 49, not 28. So this approach doesn't give rank 28.
    
    # I think I was wrong about the rank being 28. Let me reconsider.
    
    # The rank of 4x4x4 might actually be higher than 28.
    # The Strassen algorithm for 4x4x4 uses 28 multiplications if we use
    # the block-Strassen approach where each block multiplication is done
    # with 7 multiplications, but the blocks are 2x2, so each block
    # multiplication is a 2x2x2 problem.
    
    # Wait, I think I see the confusion. The Strassen algorithm for n x n x n
    # where n is a power of 2 uses n^2 * log_2(7) / 3 * n^2 / ... no, that's not right.
    
    # The number of multiplications for Strassen on n x n x n is 7^(log2(n)) = n^(log2(7)).
    # For n=4, that's 7^2 = 49.
    # For n=2, that's 7^1 = 7.
    # For n=8, that's 7^3 = 343.
    
    # So the Strassen algorithm for 4x4x4 uses 49 multiplications, not 28.
    # The rank is at most 49, not 28.
    
    # I was confusing myself. The rank of 4x4x4 is at most 49 (via Strassen),
    # not 28.
    
    # So for 4x4x5, the schoolbook rank is 80, and the Strassen-based rank
    # would be 49 (for the 4x4x4 part) + 16 (for the 4x4x1 part) = 65.
    
    # That matches the knowledge from the run: "Deterministic rank-65 Kronecker-Strassen baseline".
    
    # So the target rank is 65, not 44.
    
    # Let me implement the rank-65 decomposition.
    
    # Part 1: 4x4x4 with rank 49 (Strassen)
    # Part 2: 4x4x1 with rank 16 (schoolbook)
    # Total: 65
    
    # Let me build the U, V, W matrices.
    
    # For the 4x4x4 part (Strassen, rank 49):
    # I'll use the recursive Strassen decomposition.
    
    # For the 4x4x1 part (schoolbook, rank 16):
    # Simple dot products.
    
    # Let me implement this.
    
    # First, let me define the Strassen decomposition for 2x2x2.
    # Then I'll use it recursively for 4x4x4.
    
    # Strassen 2x2x2 (rank 7):
    # As defined above.
    
    # For 4x4x4, I'll use the block-Strassen approach.
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # Use Strassen's 7 multiplications, where each multiplication
    # is a 2x2x2 problem (rank 7).
    # Total: 7 * 7 = 49.
    
    # Let me build the coefficients.
    
    # I'll represent the decomposition as a list of (u, v, w) triples,
    # where u is a vector of length n*m=16, v is a vector of length m*p=20,
    # and w is a vector of length p*n=20.
    
    # For the 4x4x4 part, I'll build 49 terms.
    # For the 4x4x1 part, I'll build 16 terms.
    
    # Total: 65 terms.
    
    # Let me implement this.
    
    # Helper function to create a zero vector
    def zero_vec(length):
        return [Fraction(0)] * length
    
    # Build the Strassen 4x4x4 decomposition
    # I'll do this by composing the 2x2x2 Strassen decomposition.
    
    # For 4x4x4:
    # A is 4x4, indexed as A[i,j] for i,j in 0..3
    # We split into blocks: A = [[A00, A01], [A10, A11]] where A00 is A[0:2,0:2], etc.
    
    # Strassen's 7 multiplications for the block level:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Each M is a 2x2x2 multiplication.
    # For each M, we use the Strassen 2x2x2 decomposition (7 terms).
    
    # So total: 7 * 7 = 49 terms.
    
    # Let me build the u, v, w vectors for each term.
    
    # For the block-level Strassen:
    # The "input" to each M is a linear combination of blocks of A and B.
    # The "output" is a linear combination of blocks of C.
    
    # For M1:
    # Input A: A00 + A11, which is a 2x2 matrix.
    # In terms of the original A indices:
    # A00 = A[0:2, 0:2], A11 = A[2:4, 2:4]
    # So the input to M1 from A is:
    # A[0,0] + A[2,2], A[0,1] + A[2,3], A[1,0] + A[3,2], A[1,1] + A[3,3]
    
    # Input B: B00 + B11
    # B00 = B[0:2, 0:2], B11 = B[2:4, 2:4]
    # So: B[0,0] + B[2,2], B[0,1] + B[2,3], B[1,0] + B[3,2], B[1,1] + B[3,3]
    
    # Output: C00 (the top-left 2x2 block of C)
    # C00 = C[0:2, 0:2]
    
    # So for M1, the u vector has 1s at positions corresponding to A[0,0], A[2,2], A[0,1], A[2,3], A[1,0], A[3,2], A[1,1], A[3,3]
    # The v vector has 1s at positions corresponding to B[0,0], B[2,2], B[0,1], B[2,3], B[1,0], B[3,2], B[1,1], B[3,3]
    # The w vector has 1s at positions corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # But M1 is itself a 2x2x2 multiplication, so we need to decompose it further using Strassen.
    
    # This is getting complicated. Let me just use a simpler approach.
    
    # I'll use the schoolbook decomposition for now, which gives rank 80.
    # Then I'll try to find a better decomposition.
    
    # Actually, let me just implement the rank-65 decomposition using
    # the Kronecker product of Strassen decompositions.
    
    # The key insight: the 4x4x4 multiplication can be decomposed using
    # the Kronecker product of two 2x2x2 Strassen decompositions.
    # This gives rank 7*7 = 49.
    
    # Similarly, the 4x4x1 multiplication can be decomposed using
    # the Kronecker product of a 2x2x1 and a 2x2x1, but 2x2x1 is just
    # a dot product, which has rank 4.
    # So 4x4x1 has rank 4*4 = 16.
    
    # Total: 49 + 16 = 65.
    
    # Let me implement this.
    
    # First, define the Strassen 2x2x2 decomposition as a list of (u, v, w) triples.
    # u, v, w are vectors of length 4 (for 2x2 matrices).
    
    strassen_2x2_terms = []
    
    # M1 = (A11 + A22) * (B11 + B22)
    u1 = [Fraction(1), Fraction(0), Fraction(0), Fraction(1)]
    v1 = [Fraction(1), Fraction(0), Fraction(0), Fraction(1)]
    w1 = [Fraction(1), Fraction(0), Fraction(0), Fraction(1)]
    strassen_2x2_terms.append((u1, v1, w1))
    
    # M2 = (A21 + A22) * B11
    u2 = [Fraction(0), Fraction(0), Fraction(1), Fraction(1)]
    v2 = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
    w2 = [Fraction(0), Fraction(0), Fraction(1), Fraction(0)]
    strassen_2x2_terms.append((u2, v2, w2))
    
    # M3 = A11 * (B21 - B22)
    u3 = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
    v3 = [Fraction(0), Fraction(0), Fraction(1), Fraction(-1)]
    w3 = [Fraction(0), Fraction(1), Fraction(0), Fraction(0)]
    strassen_2x2_terms.append((u3, v3, w3))
    
    # M4 = A22 * (B12 - B11)
    u4 = [Fraction(0), Fraction(0), Fraction(0), Fraction(1)]
    v4 = [Fraction(1), Fraction(1), Fraction(0), Fraction(-1)]
    w4 = [Fraction(0), Fraction(0), Fraction(0), Fraction(1)]
    strassen_2x2_terms.append((u4, v4, w4))
    
    # M5 = (A11 + A12) * B22
    u5 = [Fraction(1), Fraction(1), Fraction(0), Fraction(0)]
    v5 = [Fraction(0), Fraction(0), Fraction(0), Fraction(1)]
    w5 = [Fraction(-1), Fraction(0), Fraction(0), Fraction(0)]
    strassen_2x2_terms.append((u5, v5, w5))
    
    # M6 = (A21 - A11) * (B11 + B12)
    u6 = [Fraction(-1), Fraction(0), Fraction(1), Fraction(0)]
    v6 = [Fraction(1), Fraction(1), Fraction(0), Fraction(0)]
    w6 = [Fraction(0), Fraction(0), Fraction(0), Fraction(1)]
    strassen_2x2_terms.append((u6, v6, w6))
    
    # M7 = (A12 - A22) * (B21 + B22)
    u7 = [Fraction(0), Fraction(1), Fraction(0), Fraction(-1)]
    v7 = [Fraction(0), Fraction(0), Fraction(1), Fraction(1)]
    w7 = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
    strassen_2x2_terms.append((u7, v7, w7))
    
    # Now, for 4x4x4, I'll use the Kronecker product of two Strassen decompositions.
    # This gives 7*7 = 49 terms.
    
    # For each pair of Strassen terms (term_i, term_j), the Kronecker product
    # gives a term for 4x4x4.
    
    # The u vector for the Kronecker product is the Kronecker product of u_i and u_j.
    # Similarly for v and w.
    
    # Let me implement the Kronecker product of vectors.
    
    def kron_vectors(a, b):
        result = []
        for ai in a:
            for bj in b:
                result.append(ai * bj)
        return result
    
    # Build the 4x4x4 decomposition (rank 49)
    terms_4x4x4 = []
    for i in range(7):
        for j in range(7):
            u_i = strassen_2x2_terms[i][0]
            v_i = strassen_2x2_terms[i][1]
            w_i = strassen_2x2_terms[i][2]
            u_j = strassen_2x2_terms[j][0]
            v_j = strassen_2x2_terms[j][1]
            w_j = strassen_2x2_terms[j][2]
            
            u_kron = kron_vectors(u_i, u_j)
            v_kron = kron_vectors(v_i, v_j)
            w_kron = kron_vectors(w_i, w_j)
            
            terms_4x4x4.append((u_kron, v_kron, w_kron))
    
    # Now, for the 4x4x1 part (rank 16), I'll use schoolbook.
    # The 4x4x1 multiplication is: C[i,0] = sum_j A[i,j] * B[j,0]
    # This has rank 4*4 = 16 (one term for each (i,j) pair).
    
    terms_4x4x1 = []
    for i in range(4):
        for j in range(4):
            u = zero_vec(16)
            v = zero_vec(20)
            w = zero_vec(20)
            
            # A is 4x4, so A[i,j] is at position i*4 + j in the flattened vector
            u[i * 4 + j] = Fraction(1)
            
            # B is 4x5, and we're using the last column (column 4), so B[j,4] is at position j*5 + 4
            v[j * 5 + 4] = Fraction(1)
            
            # C is 5x4 (since p*n = 5*4), and C[i,0] is at position i*4 + 0
            # Wait, the convention is W[k*n+i, r], where k is the row of C and i is the column.
            # C is 5x4, so C[k, i] where k in 0..4, i in 0..3.
            # We want C[i, 0] where i in 0..3 (the row of the output).
            # So k = i, and the column is 0.
            w[i * 4 + 0] = Fraction(1)
            
            terms_4x4x1.append((u, v, w))
    
    # Combine all terms
    all_terms = terms_4x4x4 + terms_4x4x1
    rank = len(all_terms)
    
    # Build the U, V, W matrices
    U = [[Fraction(0)] * rank for _ in range(16)]
    V = [[Fraction(0)] * rank for _ in range(20)]
    W = [[Fraction(0)] * rank for _ in range(20)]
    
    for col, (u, v, w) in enumerate(all_terms):
        for row in range(16):
            U[row][col] = u[row]
        for row in range(20):
            V[row][col] = v[row]
        for row in range(20):
            W[row][col] = w[row]
    
    # Convert to the required format
    U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in U]
    V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in V]
    W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
