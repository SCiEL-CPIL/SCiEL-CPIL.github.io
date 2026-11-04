"""Rewritten program: recursive Strassen for 4x4 * 4x6 matrix multiplication.
Strategy: split into 4x4x4 (Strassen rank 7) and 4x4x2 (schoolbook rank 8) parts.
Total rank = 7 + 8 = 15? Wait, let me think again.

Actually for 4x4*4x6:
- Split B into two 4x3 blocks? No, p=6, so split into 4x3 and 4x3? 
- Better: split A (4x4) and B (4x6) such that we multiply 4x4 * 4x4 and 4x4 * 4x2.
  - For 4x4*4x4: use Strassen -> rank 7
  - For 4x4*4x2: schoolbook -> rank 4*4*2 = 32? No wait, 4*4*2=32.
  Total = 7 + 32 = 39? That's worse than 56/77.

Wait, the dimensions are n=4, m=4, p=6.
A is 4x4, B is 4x6, C is 4x6.
Indexing: U[i*m+j, c] where i in [0,4), j in [0,4). So U has 16 rows.
V[j*p+k, c] where j in [0,4), k in [0,6). So V has 24 rows.
W[k*n+i, c] where k in [0,6), i in [0,4). So W has 24 rows.

Tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k].

Split approach:
Option 1: Split into 4x4x4 (k=0..3) and 4x4x2 (k=4..5).
- 4x4x4 part: A (4x4) * B (4x4) -> C (4x4). Use Strassen rank 7.
- 4x4x2 part: A (4x4) * B (4x2) -> C (4x2). Schoolbook rank 32.
Total = 7 + 32 = 39.

Option 2: Split A into two 2x4 blocks?
- Top: A[0:2, 0:4] * B -> C[0:2, :]. This is 2x4 * 4x6 = 2x6.
- Bottom: A[2:4, 0:4] * B -> C[2:4, :]. This is 2x4 * 4x6 = 2x6.
Each 2x4*4x6 = 2x6. Schoolbook rank 2*4*6=48 each, total 96. Worse.

Option 3: Split B into B_left (4x4) and B_right (4x2).
- A * B_left -> C_left (4x4). Strassen rank 7.
- A * B_right -> C_right (4x2). Schoolbook rank 32.
Total = 39.

Option 4: Use Strassen on the full 4x4*4x6 by padding?
- Pad B to 4x8, compute 4x4*4x8 with Strassen (treating as 4 blocks of 2x2?).
- Strassen for 4x4*4x8: pad to 4x4*4x4 with two separate multiplications?
  That's not clean.

Let me reconsider. The schoolbook rank is 4*4*6 = 96.
The knowledge says: "Replacing the inefficient independent Strassen block decomposition (rank 56 for 4x4x4 part) with a proper recursive Strassen implementation (rank 49) reduces the total rank from 84 to 77"

Wait, 4x4x6. If we split into 4x4x4 and 4x4x2:
- 4x4x4 with proper recursive Strassen = rank 49? No, Strassen for 4x4x4 should be rank 7 (since 4=2^2, Strassen gives 2^3*7=56 for direct, but recursive Strassen on 4x4x4 = Strassen(2x2x4 blocks)... actually for n=m=p=4, recursive Strassen gives rank 2^3 * 7 = 56? No.

Actually, Strassen algorithm: for n x n matrices, rank = 7^(log2(n)). For n=4, that's 7^2 = 49. For n=2, it's 7.

So 4x4*4x4 using recursive Strassen = rank 49.
4x4*4x2 schoolbook = 4*4*2 = 32.
Total = 49 + 32 = 81.

Hmm, but the knowledge says 77. Let me think about 4x4x4 and 4x4x2 differently.

Actually wait - maybe we can do better on the 4x4x2 part. 
4x4*4x2: This is like computing C[i,k] for k in {4,5}. 
This is sum over j of A[i,j]*B[j,k] for 2 specific columns of B.
This is equivalent to multiplying a 4x4 matrix by a 4x2 matrix.
The tensor rank of matmul(4,4,2) - what is it?

For matmul(n, m, p) where p < min(n,m), the rank can be less than n*m*p.
For 4x4*4x2, one approach: think of it as 2 separate 4x4*4x1 multiplications, each of rank 4 (the column of A times B). So total rank 8? No, that's not right either.

Actually for A (m x n) * B (n x p) -> C (m x p), the rank is at least max(m*n, n*p, m*p) for the trivial lower bounds.
For 4x4*4x2: max(16, 8, 8) = 16. So rank >= 16.

Schoolbook gives 4*4*2 = 32. Can we do better?

Think of it as: for each output column k (k=4,5), C[:,k] = A * B[:,k].
This is a 4x4 matrix times a 4x1 vector. The rank of this "vector" multiplication is at least 4 (since we need at least 4 terms for a general 4-dim result from a 4x4 matrix).
Doing this independently for 2 columns gives rank 8.
But we can share computations across the two columns!

Actually, C[:,4:6] = A * B[:,4:6] where B[:,4:6] is 4x2.
This is the same as computing A times a 4x2 matrix.
One way: C[:,4:6] = (A * Q) * (Q^{-1} * B[:,4:6]) for some 4x4 matrix Q? No, that doesn't reduce rank.

Let me just use schoolbook for the 4x4x2 part = rank 32, and Strassen for 4x4x4 = rank 49.
Total = 81.

But the knowledge mentions rank 77. Let me see if there's a better split.

What if we split differently? 4x4*4x6:
- Split into 4x2*4x3 and... no, the dimensions don't work out nicely.

Actually, let me reconsider. The knowledge says "rank 49 for 4x4x4" and total 77. So 77 - 49 = 28 for the 4x4x2 part? But 4*4*2 = 32. Unless there's a better way.

Hmm, 28 for 4x4*4x2? That's 4*7. Maybe using a partial Strassen-like approach.

For now, let me implement the straightforward approach:
- 4x4x4 with recursive Strassen (rank 49)
- 4x4x2 with schoolbook (rank 32)
- Total rank = 81

This is better than 96 (schoolbook) and better than the 1.0 baseline.

Let me implement recursive Strassen for 4x4*4x4.

Strassen for 2x2:
P1 = (A00+A11)*(B00+B11)
P2 = (A10+A11)*B00
P3 = A00*(B01-B11)
P4 = A11*(B10-B00)
P5 = (A00+A01)*B11
P6 = (A10-A00)*(B00+B10)
P7 = (A01-A11)*(B01+B11)
C00 = P1+P4-P5+P7
C01 = P3+P5
C10 = P2+P4
C11 = P1-P2+P3+P6

For 4x4*4x4, recursively apply Strassen to 2x2 sub-blocks.
Each 4x4 = 4 2x2 blocks. The Strassen recursion gives 7 subproblems of 2x2, each of rank 7.
So rank = 7*7 = 49.

Let me code this up properly.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_factors(A_rows, B_rows, W_rows, offset):
    """
    Generate Strassen rank-7 factors for 2x2 * 2x2 -> 2x2.
    A_rows: indices of the 4 A-entries (A00, A01, A10, A11)
    B_rows: indices of the 4 B-entries (B00, B01, B10, B11)
    W_rows: indices of the 4 C-entries (C00, C01, C10, C11)
    offset: column offset in the factor matrices
    Returns list of (U_entry, V_entry, W_entry) tuples.
    Each is (row_index, column_offset, value) or similar.
    """
    factors = []
    # P1 = (A00+A11)*(B00+B11)
    # U: A00 contributes +1 to col offset, A11 contributes +1 to col offset
    # V: B00 contributes +1, B11 contributes +1
    # W: C00 += P1, C11 += P1
    factors.append((A_rows[0], B_rows[0], W_rows[0], offset, 1, 1, 1))
    factors.append((A_rows[0], B_rows[1], W_rows[0], offset, 1, 1, 1))
    factors.append((A_rows[3], B_rows[0], W_rows[0], offset, 1, 1, 1))
    factors.append((A_rows[3], B_rows[1], W_rows[0], offset, 1, 1, 1))
    factors.append((A_rows[0], B_rows[0], W_rows[3], offset, 1, 1, 1))
    factors.append((A_rows[0], B_rows[1], W_rows[3], offset, 1, 1, 1))
    factors.append((A_rows[3], B_rows[0], W_rows[3], offset, 1, 1, 1))
    factors.append((A_rows[3], B_rows[1], W_rows[3], offset, 1, 1, 1))
    
    # P2 = (A10+A11)*B00
    factors.append((A_rows[2], B_rows[0], W_rows[2], offset+1, 1, 1, 1))
    factors.append((A_rows[3], B_rows[0], W_rows[2], offset+1, 1, 1, 1))
    
    # P3 = A00*(B01-B11)
    factors.append((A_rows[0], B_rows[1], W_rows[1], offset+2, 1, 1, 1))
    factors.append((A_rows[0], B_rows[3], W_rows[1], offset+2, 1, -1, 1))
    
    # P4 = A11*(B10-B00)
    factors.append((A_rows[3], B_rows[2], W_rows[0], offset+3, 1, 1, 1))
    factors.append((A_rows[3], B_rows[0], W_rows[0], offset+3, 1, -1, 1))
    factors.append((A_rows[3], B_rows[2], W_rows[2], offset+3, 1, 1, 1))
    factors.append((A_rows[3], B_rows[0], W_rows[2], offset+3, 1, -1, 1))
    
    # P5 = (A00+A01)*B11
    factors.append((A_rows[0], B_rows[3], W_rows[1], offset+4, 1, 1, 1))
    factors.append((A_rows[1], B_rows[3], W_rows[1], offset+4, 1, 1, 1))
    factors.append((A_rows[0], B_rows[3], W_rows[0], offset+4, 1, -1, 1))
    factors.append((A_rows[1], B_rows[3], W_rows[0], offset+4, 1, -1, 1))
    
    # P6 = (A10-A00)*(B00+B10)
    factors.append((A_rows[2], B_rows[0], W_rows[3], offset+5, 1, 1, 1))
    factors.append((A_rows[2], B_rows[2], W_rows[3], offset+5, 1, 1, 1))
    factors.append((A_rows[0], B_rows[0], W_rows[3], offset+5, -1, 1, 1))
    factors.append((A_rows[0], B_rows[2], W_rows[3], offset+5, -1, 1, 1))
    
    # P7 = (A01-A11)*(B01+B11)
    factors.append((A_rows[1], B_rows[1], W_rows[0], offset+6, 1, 1, 1))
    factors.append((A_rows[1], B_rows[3], W_rows[0], offset+6, 1, 1, 1))
    factors.append((A_rows[3], B_rows[1], W_rows[0], offset+6, -1, 1, 1))
    factors.append((A_rows[3], B_rows[3], W_rows[0], offset+6, -1, 1, 1))
    factors.append((A_rows[1], B_rows[1], W_rows[1], offset+6, 1, 1, 1))
    factors.append((A_rows[1], B_rows[3], W_rows[1], offset+6, 1, 1, 1))
    factors.append((A_rows[3], B_rows[1], W_rows[1], offset+6, -1, 1, 1))
    factors.append((A_rows[3], B_rows[3], W_rows[1], offset+6, -1, 1, 1))
    
    return factors


def _strassen_4x4_factors(A_start, B_start, W_start):
    """
    Generate rank-49 factors for 4x4 * 4x4 -> 4x4 using recursive Strassen.
    A_start: starting row index in U for A entries (16 rows)
    B_start: starting row index in V for B entries (16 rows for this sub-block)
    W_start: starting row index in W for C entries (16 rows for this sub-block)
    Returns list of factors and the rank used.
    """
    factors = []
    # For 4x4, we have 2x2 sub-blocks.
    # A block layout: A00=(rows 0-1, cols 0-1), A01=(rows 0-1, cols 2-3), etc.
    # In our flat indexing: A[i*m+j] where m=4.
    # A[i,j] for i in 0..3, j in 0..3 -> index i*4+j
    
    # For the 4x4x4 sub-problem, we're multiplying A[0:4, 0:4] * B[0:4, 0:4]
    # But B in the full problem has 6 columns. Here we only use B[0:4, 0:4].
    
    # Sub-blocks (each 2x2):
    # A00: i=0,1; j=0,1 -> A indices: 0,1,4,5
    # A01: i=0,1; j=2,3 -> A indices: 2,3,6,7
    # A10: i=2,3; j=0,1 -> A indices: 8,9,12,13
    # A11: i=2,3; j=2,3 -> A indices: 10,11,14,15
    
    # B00: j=0,1; k=0,1 -> B indices: 0,1,4,5
    # B01: j=0,1; k=2,3 -> B indices: 2,3,6,7
    # B10: j=2,3; k=0,1 -> B indices: 8,9,12,13
    # B11: j=2,3; k=2,3 -> B indices: 10,11,14,15
    
    # C00: i=0,1; k=0,1 -> C indices: 0,1,4,5
    # C01: i=0,1; k=2,3 -> C indices: 2,3,6,7
    # C10: i=2,3; k=0,1 -> C indices: 8,9,12,13
    # C11: i=2,3; k=2,3 -> C indices: 10,11,14,15
    
    # Apply Strassen at the 4x4 level (7 sub-problems of 2x2x2)
    # P1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # P2 = (A10+A11)*B00 -> contributes to C10
    # P3 = A00*(B01-B11) -> contributes to C01
    # P4 = A11*(B10-B00) -> contributes to C00, C10
    # P5 = (A00+A01)*B11 -> contributes to C01, C00
    # P6 = (A10-A00)*(B00+B10) -> contributes to C11
    # P7 = (A01-A11)*(B01+B11) -> contributes to C00, C01
    
    # Each sub-problem is 2x2*2x2->2x2, rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Define the sub-block index arrays
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    B00 = [0, 1, 4, 5]
    B01 = [2, 3, 6, 7]
    B10 = [8, 9, 12, 13]
    B11 = [10, 11, 14, 15]
    
    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]
    
    rank_offset = 0
    
    # P1 = (A00+A11)*(B00+B11)
    # This is a 2x2*2x2 problem where A_sub = A00+A11, B_sub = B00+B11
    # U entries: for each element in A00 and A11, +1
    # V entries: for each element in B00 and B11, +1
    # W entries: C00 gets +P1, C11 gets +P1
    # The 2x2 factors: use _strassen_2x2_factors on the sub-blocks
    
    # Actually, let me think of this differently. 
    # P1 itself is a product of two 2x2 matrices. To compute it via rank-7,
    # we need 7 scalar multiplications.
    # But P1 feeds into C00 and C11 with coefficient +1.
    # So for each of the 7 Strassen products that compute P1, 
    # the W factor needs to account for both C00 and C11 contributions.
    
    # This is getting complex. Let me use a cleaner recursive approach.
    
    # I'll build the factors directly for the full 4x4*4x4 using the 7 Strassen
    # sub-programs, each of which is a 2x2*2x2 computed with rank 7.
    
    # For each of the 7 Strassen combinations, I have a 2x2*2x2 sub-problem.
    # For that sub-problem, I use the rank-7 Strassen factors.
    # The U and V entries are linear combinations of A/B sub-block elements.
    # The W entries are linear combinations of C sub-block elements.
    
    # Let me define a helper that takes:
    # - A_entries: list of (A_local_index, coefficient) 
    # - B_entries: list of (B_local_index, coefficient)
    # - C_entries: list of (C_local_index, coefficient)
    # - rank_col_offset
    # And generates the 7 rank-1 factors.
    
    def gen_subproblem_factors(A_entries, B_entries, C_entries, col_offset, A_base, B_base, C_base):
        """
        Generate rank-7 factors for a 2x2*2x2 sub-problem embedded in larger matrices.
        A_entries: list of (local_2x2_index, coeff) where local_2x2_index is 0..3
                   mapping to A[local_row, local_col] in the 2x2 block
        B_entries: similar for B
        C_entries: similar for C, with the output coefficients
        col_offset: starting column in the factor matrices
        A_base, B_base, C_base: base indices in the full factor matrices
        """
        # A local index to full U index:
        # local 0 -> (0,0) -> A_base + 0
        # local 1 -> (0,1) -> A_base + 1
        # local 2 -> (1,0) -> A_base + 2
        # local 3 -> (1,1) -> A_base + 3
        # Similarly for B and C.
        
        sub_factors = []
        # The 7 Strassen products for 2x2*2x2:
        # Each product is U_row * V_row contributing to W with some coefficient
        
        # I need to map A_entries to actual U rows.
        # A_entries has form [(local_idx, coeff), ...]
        # Each local_idx maps to A_base + local_idx
        # So U_row = A_base + local_idx, U_val = coeff
        
        # For each of the 7 Strassen terms, I need:
        # - Which A elements participate (with what coefficients)
        # - Which B elements participate
        # - Which C elements receive the result (with what coefficients)
        
        # Strassen 2x2:
        # P1 = (A00+A11)*(B00+B11) -> C00 += P1, C11 += P1
        # P2 = (A10+A11)*B00 -> C10 += P2
        # P3 = A00*(B01-B11) -> C01 += P3
        # P4 = A11*(B10-B00) -> C00 += P4, C10 += P4
        # P5 = (A00+A01)*B11 -> C01 += P5, C00 -= P5
        # P6 = (A10-A00)*(B00+B10) -> C11 += P6
        # P7 = (A01-A11)*(B01+B11) -> C00 += P7, C01 += P7
        
        # For each product, I create rank-1 terms.
        # A product like (sum a_i * A_i) * (sum b_j * B_j) contributes
        # (sum a_i * A_i) * (sum b_j * B_j) to each C_k with coefficient c_k.
        # This is a bilinear form. To decompose into rank-1:
        # Each pair (i,j) gives a rank-1 term: A_i * B_j with coefficient a_i * b_j * c_k.
        
        # But that would be rank 4 per product (2x2 = 4 terms), not rank 7 total.
        # The rank-7 comes from the 7 products, not from expanding each.
        
        # Wait, I'm confusing myself. The rank-7 Strassen for 2x2*2x2 means
        # 7 scalar multiplications total. Each scalar multiplication is a rank-1 term.
        # So for the 2x2*2x2 sub-problem, I get 7 rank-1 factors.
        
        # Let me just use the standard Strassen rank-7 decomposition.
        
        # Build lookup: for A_entries, create dict local_idx -> coeff
        A_lookup = {}
        for local_idx, coeff in A_entries:
            A_lookup[local_idx] = A_lookup.get(local_idx, 0) + coeff
        
        B_lookup = {}
        for local_idx, coeff in B_entries:
            B_lookup[local_idx] = B_lookup.get(local_idx, 0) + coeff
        
        C_lookup = {}
        for local_idx, coeff in C_entries:
            C_lookup[local_idx] = C_lookup.get(local_idx, 0) + coeff
        
        # Now generate 7 rank-1 factors based on Strassen
        # Each factor: (U_row, U_val, V_row, V_val, W_row, W_val)
        
        def get_A(local_idx):
            return A_base + local_idx, A_lookup.get(local_idx, 0)
        
        def get_B(local_idx):
            return B_base + local_idx, B_lookup.get(local_idx, 0)
        
        def get_C(local_idx):
            return C_base + local_idx, C_lookup.get(local_idx, 0)
        
        # P1 = (A00+A11)*(B00+B11)
        # Expand: A00*B00 + A00*B11 + A11*B00 + A11*B11
        # But in Strassen, this is ONE multiplication of two sums.
        # As a bilinear form, it's still 4 rank-1 terms if expanded.
        # No wait - in the tensor decomposition, each "multiplication" is a rank-1 term.
        # Strassen says we only need 7 such multiplications.
        
        # I think the issue is that I'm overcomplicating this.
        # The rank-7 Strassen for 2x2*2x2 means there are 7 pairs (u, v) such that
        # the output = sum of 7 terms, each being u_term * v_term mapped to outputs.
        
        # Let me just hardcode the 7 Strassen factors for 2x2*2x2.
        # Standard Strassen:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b10)
        # M7 = (a01-a11)*(b01+b11)
        # c00 = M1+M4-M5+M7
        # c01 = M3+M5
        # c10 = M2+M4
        # c11 = M1-M2+M3+M6
        
        # Each Mk is a product of a linear combination of a's and a linear combination of b's.
        # In the tensor decomposition, each Mk is a "gate" that multiplies 
        # one A-linear-combination with one B-linear-combination.
        # But for the rank-1 decomposition, each such product IS a rank-1 term
        # where the U vector has the A coefficients and V vector has the B coefficients.
        
        # So for M1: U has +1 at a00, +1 at a11; V has +1 at b00, +1 at b11.
        # The output c00 gets M1 * 1, c11 gets M1 * 1.
        # So W has +1 at c00, +1 at c11.
        
        # This is a rank-1 tensor term: U_vector ⊗ V_vector ⊗ W_vector
        # where U_vector = [1, 0, 0, 1, 0, 0, 0, 0, ...] (nonzero at a00, a11)
        # V_vector = [1, 1, 0, 0, ...] (nonzero at b00, b11)
        # W_vector = [1, 0, 0, 1, ...] (nonzero at c00, c11)
        
        # Great, so each Strassen product gives ONE rank-1 term.
        
        # Now, in our embedded sub-problem, the A/B/C entries are linear combinations
        # of the full A/B/C entries. So the U/V/W vectors need to account for that.
        
        # For M1 in the sub-problem:
        # A_part = sum over A_entries where local_idx in {0, 3} (a00, a11)
        #   with the coefficients from A_lookup
        # Similarly for B and C.
        
        factors = []
        col = col_offset
        
        # M1: (a00+a11)*(b00+b11), contributes to c00 (+1), c11 (+1)
        # A coefficients: a00: A_lookup[0], a11: A_lookup[3]
        a00_coeff = A_lookup.get(0, 0)
        a11_coeff = A_lookup.get(3, 0)
        b00_coeff = B_lookup.get(0, 0)
        b11_coeff = B_lookup.get(3, 0)
        c00_coeff = C_lookup.get(0, 0)
        c11_coeff = C_lookup.get(3, 0)
        
        # U: row A_base+0 has value a00_coeff, row A_base+3 has value a11_coeff
        # V: row B_base+0 has value b00_coeff, row B_base+3 has value b11_coeff
        # W: row C_base+0 has value c00_coeff * 1, row C_base+3 has value c11_coeff * 1
        # All nonzero.
        factors.append(('U', A_base + 0, col, a00_coeff))
        factors.append(('U', A_base + 3, col, a11_coeff))
        factors.append(('V', B_base + 0, col, b00_coeff))
        factors.append(('V', B_base + 3, col, b11_coeff))
        factors.append(('W', C_base + 0, col, c00_coeff))
        factors.append(('W', C_base + 3, col, c11_coeff))
        col += 1
        
        # M2: (a10+a11)*b00, contributes to c10 (+1)
        a10_coeff = A_lookup.get(2, 0)
        a11_coeff2 = A_lookup.get(3, 0)
        b00_coeff2 = B_lookup.get(0, 0)
        c10_coeff = C_lookup.get(2, 0)
        
        factors.append(('U', A_base + 2, col, a10_coeff))
        factors.append(('U', A_base + 3, col, a11_coeff2))
        factors.append(('V', B_base + 0, col, b00_coeff2))
        factors.append(('W', C_base + 2, col, c10_coeff))
        col += 1
        
        # M3: a00*(b01-b11), contributes to c01 (+1)
        a00_coeff2 = A_lookup.get(0, 0)
        b01_coeff = B_lookup.get(1, 0)
        b11_coeff2 = B_lookup.get(3, 0)
        c01_coeff = C_lookup.get(1, 0)
        
        factors.append(('U', A_base + 0, col, a00_coeff2))
        factors.append(('V', B_base + 1, col, b01_coeff))
        factors.append(('V', B_base + 3, col, -b11_coeff2))
        factors.append(('W', C_base + 1, col, c01_coeff))
        col += 1
        
        # M4: a11*(b10-b00), contributes to c00 (+1), c10 (+1)
        a11_coeff3 = A_lookup.get(3, 0)
        b10_coeff = B_lookup.get(2, 0)
        b00_coeff3 = B_lookup.get(0, 0)
        c00_coeff2 = C_lookup.get(0, 0)
        c10_coeff2 = C_lookup.get(2, 0)
        
        factors.append(('U', A_base + 3, col, a11_coeff3))
        factors.append(('V', B_base + 2, col, b10_coeff))
        factors.append(('V', B_base + 0, col, -b00_coeff3))
        factors.append(('W', C_base + 0, col, c00_coeff2))
        factors.append(('W', C_base + 2, col, c10_coeff2))
        col += 1
        
        # M5: (a00+a01)*b11, contributes to c01 (+1), c00 (-1)
        a00_coeff3 = A_lookup.get(0, 0)
        a01_coeff = A_lookup.get(1, 0)
        b11_coeff3 = B_lookup.get(3, 0)
        c01_coeff2 = C_lookup.get(1, 0)
        c00_coeff3 = C_lookup.get(0, 0)
        
        factors.append(('U', A_base + 0, col, a00_coeff3))
        factors.append(('U', A_base + 1, col, a01_coeff))
        factors.append(('V', B_base + 3, col, b11_coeff3))
        factors.append(('W', C_base + 1, col, c01_coeff2))
        factors.append(('W', C_base + 0, col, -c00_coeff3))
        col += 1
        
        # M6: (a10-a00)*(b00+b10), contributes to c11 (+1)
        a10_coeff2 = A_lookup.get(2, 0)
        a00_coeff4 = A_lookup.get(0, 0)
        b00_coeff4 = B_lookup.get(0, 0)
        b10_coeff2 = B_lookup.get(2, 0)
        c11_coeff2 = C_lookup.get(3, 0)
        
        factors.append(('U', A_base + 2, col, a10_coeff2))
        factors.append(('U', A_base + 0, col, -a00_coeff4))
        factors.append(('V', B_base + 0, col, b00_coeff4))
        factors.append(('V', B_base + 2, col, b10_coeff2))
        factors.append(('W', C_base + 3, col, c11_coeff2))
        col += 1
        
        # M7: (a01-a11)*(b01+b11), contributes to c00 (+1), c01 (+1)
        a01_coeff2 = A_lookup.get(1, 0)
        a11_coeff4 = A_lookup.get(3, 0)
        b01_coeff2 = B_lookup.get(1, 0)
        b11_coeff4 = B_lookup.get(3, 0)
        c00_coeff4 = C_lookup.get(0, 0)
        c01_coeff3 = C_lookup.get(1, 0)
        
        factors.append(('U', A_base + 1, col, a01_coeff2))
        factors.append(('U', A_base + 3, col, -a11_coeff4))
        factors.append(('V', B_base + 1, col, b01_coeff2))
        factors.append(('V', B_base + 3, col, b11_coeff4))
        factors.append(('W', C_base + 0, col, c00_coeff4))
        factors.append(('W', C_base + 1, col, c01_coeff3))
        col += 1
        
        return factors, 7
    
    # Now define the 7 Strassen sub-problems for 4x4*4x4
    # A sub-blocks:
    # A00: local indices 0,1,4,5 in the 4x4 -> but in the 2x2 sub-problem, 
    # the local 2x2 indices are 0,1,2,3 corresponding to the 2x2 block elements.
    # A00 block: elements A[0,0], A[0,1], A[1,0], A[1,1] -> full indices 0,1,4,5
    # In the 2x2 sub-problem, local 0=A[0,0]=full 0, local 1=A[0,1]=full 1,
    # local 2=A[1,0]=full 4, local 3=A[1,1]=full 5
    
    # So for A00, A_entries = [(0,1),(1,1),(2,1),(3,1)], A_base = 0
    # For A11, A_entries = [(0,1),(1,1),(2,1),(3,1)], A_base = 10
    
    # Let me define the sub-block definitions:
    # (local_indices, base) for each sub-block
    # Local indices are always [0, 1, 2, 3] with coefficient 1 for "positive" blocks
    
    subA00 = ([0,1,2,3], 0)    # A[0:2, 0:2]
    subA01 = ([0,1,2,3], 2)    # A[0:2, 2:4]
    subA10 = ([0,1,2,3], 8)    # A[2:4, 0:2]
    subA11 = ([0,1,2,3], 10)   # A[2:4, 2:4]
    
    subB00 = ([0,1,2,3], 0)    # B[0:2, 0:2]
    subB01 = ([0,1,2,3], 2)    # B[0:2, 2:4]
    subB10 = ([0,1,2,3], 8)    # B[2:4, 0:2]
    subB11 = ([0,1,2,3], 10)   # B[2:4, 2:4]
    
    subC00 = ([0,1,2,3], 0)    # C[0:2, 0:2]
    subC01 = ([0,1,2,3], 2)    # C[0:2, 2:4]
    subC10 = ([0,1,2,3], 8)    # C[2:4, 0:2]
    subC11 = ([0,1,2,3], 10)   # C[2:4, 2:4]
    
    all_factors = []
    rank_used = 0
    
    # For each Strassen product, define the A_entries, B_entries, C_entries
    # and call gen_subproblem_factors
    
    def make_entries(indices, base, coeffs=None):
        """Make entries list from indices with optional coefficients."""
        if coeffs is None:
            coeffs = [1]*len(indices)
        entries = []
        for i, (idx, c) in enumerate(zip(indices, coeffs)):
            entries.append((idx, base + i, c))
        # Wait, this isn't right. Let me reconsider.
        # entries should be [(local_2x2_idx, coeff), ...]
        # where local_2x2_idx is 0,1,2,3
        # and the full row is base + local_2x2_idx
        # So:
        result = []
        for local_idx in range(4):
            c = coeffs[local_idx]
            if c != 0:
                result.append((local_idx, c))
        return result, base
    
    # P1 = (A00+A11)*(B00+B11) -> C00 +, C11 +
    A_entries1, A_base1 = make_entries([0,1,2,3], 0, [1,1,1,1])  # A00
    A_entries1b, A_base1b = make_entries([0,1,2,3], 10, [1,1,1,1])  # A11
    # Combined A: need to merge A00 and A11 contributions
    # This is where it gets complex. Let me just handle it differently.
    
    # I think the cleanest approach is to just enumerate all the rank-1 terms
    # for the full 4x4*4x4 Strassen decomposition directly.
    
    # Let me take a step back and use a simpler but correct approach.
    # For 4x4*4x4, use the fact that it's equivalent to 2x2 blocks with Strassen.
    # I'll just build the full factor matrices using the recursive structure.
    
    # Actually, the simplest correct approach: use the known rank-49 decomposition
    # by applying Strassen at the 4x4 level (7 sub-problems of size 2x2x2).
    
    # Each sub-problem is 2x2*2x2, which has rank 7.
    # So total rank = 7*7 = 49.
    
    # Let me define each of the 7 sub-problems clearly:
    
    # P1: (A00+A11) * (B00+B11)
    #   A sub = A00+A11: U gets +1 from A00 elements and +1 from A11 elements
    #   B sub = B00+B11: V gets +1 from B00 elements and +1 from B11 elements  
    #   C out: C00 gets +result, C11 gets +result
    
    # To compute (A00+A11)*(B00+B11) as a 2x2 product, I need rank 7 for the 2x2.
    # But (A00+A11) is a 2x2 matrix, and (B00+B11) is a 2x2 matrix.
    # So I apply the 2x2 Strassen rank-7 to this sub-problem.
    
    # The U vector for the 2x2 sub-problem has entries from A00 and A11.
    # Specifically, for the 2x2 Strassen M1 = (a00+a11)*(b00+b11) of the sub-problem:
    #   a00 of sub = A00[0,0] + A11[0,0]
    #   a11 of sub = A00[1,1] + A11[1,1]
    #   etc.
    
    # This is getting really involved. Let me just use a brute-force but correct
    # recursive Strassen that I can verify.
    
    # Alternative: use the fact that matmul(4,4,4) has a known rank-49 decomposition
    # and just hardcode it or generate it systematically.
    
    # Let me try a different tactic: use numpy to verify and build the factors
    # by exploiting linearity.
    
    # For the purpose of this task, let me just use schoolbook for everything
    # and accept rank 96, which still beats the 1.0 baseline (which is also 96).
    # Wait, the baseline IS schoolbook rank 96. So I need to do better.
    
    # OK let me just carefully implement the 4x4x4 Strassen rank-49.
    
    # I'll use a recursive function that builds the factor matrices for
    # an NxNxN Strassen decomposition.
    
    return factors, rank_used  # This is a stub, let me redo this properly

# Let me restart with a cleaner implementation.

def _build_strassen_4x4x4(n):
    """
    Build rank-49 factor matrices for 4x4*4x4->4x4 using recursive Strassen.
    Returns (U_16x49, V_16x49, W_16x49) as numpy arrays.
    """
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Recursive Strassen: 4x4 = 2x(2x2) blocks
    # Apply Strassen at the 4x4 level: 7 sub-problems of 2x2*2x2
    # Each 2x2*2x2 uses rank-7 Strassen
    
    # Sub-block indices (row, col) in the 4x4 matrix -> flat index i*4+j
    # A00: rows 0-1, cols 0-1 -> [0,1,4,5]
    # A01: rows 0-1, cols 2-3 -> [2,3,6,7]
    # A10: rows 2-3, cols 0-1 -> [8,9,12,13]
    # A11: rows 2-3, cols 2-3 -> [10,11,14,15]
    
    # Similarly for B and C.
    
    def strassen_2x2_in_block(A_rows, B_rows, C_rows, col_start, sign_a=1, sign_b=1, a2_rows=None, sign_a2=1, b2_rows=None, sign_b2=1, c2_rows=None, sign_c2=1):
        """
        Apply rank-7 Strassen for a 2x2*2x2 where the 2x2 blocks may be sums.
        A_rows: list of 4 row indices for the first 2x2 block (local order: 00,01,10,11)
        B_rows: list of 4 row indices for B block
        C_rows: list of 4 row indices for C block
        col_start: starting column
        If a2_rows given, add it as a second 2x2 block with sign_a2
        Similarly for b2_rows, c2_rows
        """
        # Combine A blocks into a single 2x2 with coefficients
        # A_local[local_row, local_col] = sign_a * A[A_rows[local_row*2+local_col]] 
        #                                + (if a2_rows) sign_a2 * A[a2_rows[local_row*2+local_col]]
        
        def get_A_coeff(local_idx):
            # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            r = local_idx // 2
            c = local_idx % 2
            coeff = sign_a
            if a2_rows is not None:
                coeff += sign_a2
            # But we need to add to different rows!
            # Actually, this function should add entries to U, V, W directly.
            pass
        
        # This approach is too convoluted. Let me just directly enumerate.
        
        # For each of the 7 Strassen products, I'll add rank-1 terms.
        # Product 1: (A00+A11)*(B00+B11)
        #   U: +sign_a on A_rows[0] and A_rows[3] (which are (0,0) and (1,1))
        #      +sign_a2 on a2_rows[0] and a2_rows[3]
        #   V: +sign_b on B_rows[0] and B_rows[3]
        #      +sign_b2 on b2_rows[0] and b2_rows[3]
        #   W: +sign_c on C_rows[0] and C_rows[3]  (C00 and C11)
        #      +sign_c2 on c2_rows[0] and c2_rows[3]
        
        def add_term1(col):
            # (A00+A11)*(B00+B11) -> C00 +, C11 +
            U[A_rows[0], col] += sign_a
            U[A_rows[3], col] += sign_a
            if a2_rows is not None:
                U[a2_rows[0], col] += sign_a2
                U[a2_rows[3], col] += sign_a2
            V[B_rows[0], col] += sign_b
            V[B_rows[3], col] += sign_b
            if b2_rows is not None:
                V[b2_rows[0], col] += sign_b2
                V[b2_rows[3], col] += sign_b2
            W[C_rows[0], col] += sign_c
            W[C_rows[3], col] += sign_c
            if c2_rows is not None:
                W[c2_rows[0], col] += sign_c2
                W[c2_rows[3], col] += sign_c2
        
        def add_term2(col):
            # (A10+A11)*B00 -> C10 +
            U[A_rows[2], col] += sign_a
            U[A_rows[3], col] += sign_a
            if a2_rows is not None:
                U[a2_rows[2], col] += sign_a2
                U[a2_rows[3], col] += sign_a2
            V[B_rows[0], col] += sign_b
            if b2_rows is not None:
                V[b2_rows[0], col] += sign_b2
            W[C_rows[2], col] += sign_c
            if c2_rows is not None:
                W[c2_rows[2], col] += sign_c2
        
        def add_term3(col):
            # A00*(B01-B11) -> C01 +
            U[A_rows[0], col] += sign_a
            if a2_rows is not None:
                U[a2_rows[0], col] += sign_a2
            V[B_rows[1], col] += sign_b
            V[B_rows[3], col] -= sign_b
            if b2_rows is not None:
                V[b2_rows[1], col] += sign_b2
                V[b2_rows[3], col] -= sign_b2
            W[C_rows[1], col] += sign_c
            if c2_rows is not None:
                W[c2_rows[1], col] += sign_c2
        
        def add_term4(col):
            # A11*(B10-B00) -> C00 +, C10 +
            U[A_rows[3], col] += sign_a
            if a2_rows is not None:
                U[a2_rows[3], col] += sign_a2
            V[B_rows[2], col] += sign_b
            V[B_rows[0], col] -= sign_b
            if b2_rows is not None:
                V[b2_rows[2], col] += sign_b2
                V[b2_rows[0], col] -= sign_b2
            W[C_rows[0], col] += sign_c
            W[C_rows[2], col] += sign_c
            if c2_rows is not None:
                W[c2_rows[0], col] += sign_c2
                W[c2_rows[2], col] += sign_c2
        
        def add_term5(col):
            # (A00+A01)*B11 -> C01 +, C00 -
            U[A_rows[0], col] += sign_a
            U[A_rows[1], col] += sign_a
            if a2_rows is not None:
                U[a2_rows[0], col] += sign_a2
                U[a2_rows[1], col] += sign_a2
            V[B_rows[3], col] += sign_b
            if b2_rows is not None:
                V[b2_rows[3], col] += sign_b2
            W[C_rows[1], col] += sign_c
            W[C_rows[0], col] -= sign_c
            if c2_rows is not None:
                W[c2_rows[1], col] += sign_c2
                W[c2_rows[0], col] -= sign_c2
        
        def add_term6(col):
            # (A10-A00)*(B00+B10) -> C11 +
            U[A_rows[2], col] += sign_a
            U[A_rows[0], col] -= sign_a
            if a2_rows is not None:
                U[a2_rows[2], col] += sign_a2
                U[a2_rows[0], col] -= sign_a2
            V[B_rows[0], col] += sign_b
            V[B_rows[2], col] += sign_b
            if b2_rows is not None:
                V[b2_rows[0], col] += sign_b2
                V[b2_rows[2], col] += sign_b2
            W[C_rows[3], col] += sign_c
            if c2_rows is not None:
                W[c2_rows[3], col] += sign_c2
        
        def add_term7(col):
            # (A01-A11)*(B01+B11) -> C00 +, C01 +
            U[A_rows[1], col] += sign_a
            U[A_rows[3], col] -= sign_a
            if a2_rows is not None:
                U[a2_rows[1], col] += sign_a2
                U[a2_rows[3], col] -= sign_a2
            V[B_rows[1], col] += sign_b
            V[B_rows[3], col] += sign_b
            if b2_rows is not None:
                V[b2_rows[1], col] += sign_b2
                V[b2_rows[3], col] += sign_b2
            W[C_rows[0], col] += sign_c
            W[C_rows[1], col] += sign_c
            if c2_rows is not None:
                W[c2_rows[0], col] += sign_c2
                W[c2_rows[1], col] += sign_c2
        
        col = col_start
        add_term1(col); col += 1
        add_term2(col); col += 1
        add_term3(col); col += 1
        add_term4(col); col += 1
        add_term5(col); col += 1
        add_term6(col); col += 1
        add_term7(col); col += 1
        return col
    
    # Define 2x2 sub-blocks as flat index lists [00, 01, 10, 11]
    A00_idx = [0, 1, 4, 5]
    A01_idx = [2, 3, 6, 7]
    A10_idx = [8, 9, 12, 13]
    A11_idx = [10, 11, 14, 15]
    
    B00_idx = [0, 1, 4, 5]
    B01_idx = [2, 3, 6, 7]
    B10_idx = [8, 9, 12, 13]
    B11_idx = [10, 11, 14, 15]
    
    C00_idx = [0, 1, 4, 5]
    C01_idx = [2, 3, 6, 7]
    C10_idx = [8, 9, 12, 13]
    C11_idx = [10, 11, 14, 15]
    
    col = 0
    # P1 = (A00+A11)*(B00+B11)
    col = strassen_2x2_in_block(A00_idx, B00_idx, C00_idx, col,
                                  sign_a=1, a2_rows=A11_idx, sign_a2=1,
                                  sign_b=1, b2_rows=B11_idx, sign_b2=1,
                                  sign_c=1, c2_rows=C11_idx, sign_c2=1)
    # P2 = (A10+A11)*B00
    col = strassen_2x2_in_block(A10_idx, B00_idx, C10_idx, col,
                                  sign_a=1, a2_rows=A11_idx, sign_a2=1,
                                  sign_b=1,
                                  sign_c=1)
    # P3 = A00*(B01-B11)
    col = strassen_2x2_in_block(A00_idx, B01_idx, C01_idx, col,
                                  sign_a=1,
                                  sign_b=1, b2_rows=B11_idx, sign_b2=-1,
                                  sign_c=1)
    # P4 = A11*(B10-B00)
    col = strassen_2x2_in_block(A11_idx, B10_idx, C00_idx, col,
                                  sign_a=1,
                                  sign_b=1, b2_rows=B00_idx, sign_b2=-1,
                                  sign_c=1, c2_rows=C10_idx, sign_c2=1)
    # P5 = (A00+A01)*B11
    col = strassen_2x2_in_block(A00_idx, B11_idx, C01_idx, col,
                                  sign_a=1, a2_rows=A01_idx, sign_a2=1,
                                  sign_b=1,
                                  sign_c=1, c2_rows=C00_idx, sign_c2=-1)
    # P6 = (A10-A00)*(B00+B10)
    col = strassen_2x2_in_block(A10_idx, B00_idx, C11_idx, col,
                                  sign_a=1, a2_rows=A00_idx, sign_a2=-1,
                                  sign_b=1, b2_rows=B10_idx, sign_b2=1,
                                  sign_c=1)
    # P7 = (A01-A11)*(B01+B11)
    col = strassen_2x2_in_block(A01_idx, B01_idx, C00_idx, col,
                                  sign_a=1, a2_rows=A11_idx, sign_a2=-1,
                                  sign_b=1, b2_rows=B11_idx, sign_b2=1,
                                  sign_c=1, c2_rows=C01_idx, sign_c2=1)
    
    assert col == 49, f"Expected 49, got {col}"
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline with margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Split into 4x4x4 (k=0..3) and 4x4x2 (k=4..5)
    
    # Part 1: 4x4x4 using Strassen rank 49
    U_444, V_444, W_444 = _build_strassen_4x4x4(n)
    
    # Part 2: 4x4x2 using schoolbook rank 32
    rank_442 = n * m * 2  # 4*4*2 = 32
    U_442 = np.zeros((n * m, rank_442))
    V_442 = np.zeros((m * 2, rank_442))  # Only the last 2 columns of B
    W_442 = np.zeros((2 * n, rank_442))   # Only the last 2 rows of C
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U_442[i * m + j, col] = 1
                V_442[j * 2 + k, col] = 1
                W_442[k * n + i, col] = 1
                col += 1
    
    # Now assemble the full matrices
    # U: 16 x (49 + 32) = 16 x 81
    # V: 24 x 81
    # W: 24 x 81
    
    rank_total = 49 + rank_442  # 81
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))  # 24 rows
    W = np.zeros((p * n, rank_total))  # 24 rows
    
    # Place 4x4x4 part
    U[:, :49] = U_444
    V[:16, :49] = V_444  # B[0:4, 0:4] corresponds to V rows 0-15
    W[:16, :49] = W_444  # C[0:4, 0:4] corresponds to W rows 0-15
    
    # Place 4x4x2 part (columns 4,5 of B and C)
    # V_442 is 8x32 (m*2 = 4*2 = 8 rows for B cols 4,5)
    # These correspond to V rows 16-23 (B[0:4, 4:6])
    # V row index: j*p+k where j in 0..3, k in 4,5
    # j=0,k=4 -> 0*6+4=4? No wait, p=6.
    # V[j*p+k]: j=0,k=0->0, j=0,k=1->1, ..., j=0,k=5->5, j=1,k=0->6, ...
    # So B[0,4]=V[4], B[0,5]=V[5], B[1,4]=V[10], B[1,5]=V[11], etc.
    
    # V_442 row index: j*2+k where j=0..3, k=0,1 (local k for cols 4,5)
    # V_442[j*2+0] = B[j,4], V_442[j*2+1] = B[j,5]
    # Full V row: j*6+4 = 6j+4, j*6+5 = 6j+5
    
    for j in range(m):
        V[j*p+4, 49:49+rank_442] = V_442[j*2+0, :]  # B[j,4]
        V[j*p+5, 49:49+rank_442] = V_442[j*2+1, :]  # B[j,5]
    
    # W_442 is 8x32 (2*n = 2*4 = 8 rows for C cols 4,5)
    # W_442[k_local*n+i] where k_local=0,1 (for C cols 4,5), i=0..3
    # W_442[0*n+i] = C[i,4], W_442[1*n+i] = C[i,5]
    # Full W row: k*n+i where k=4,5 -> k*n+i = 4*4+i=16+i, 5*4+i=20+i
    
    for i in range(n):
        W[4*n+i, 49:49+rank_442] = W_442[0*n+i, :]  # C[i,4]
        W[5*n+i, 49:49+rank_442] = W_442[1*n+i, :]  # C[i,5]
    
    # Convert to lists and return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
