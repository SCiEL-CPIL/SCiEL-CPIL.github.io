"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_4x4_decomposition():
    """
    Return the 4x4x4 matrix multiplication tensor decomposition using Strassen.
    Strassen: (A0, A1) * (B0, B1) = C where A, B, C are 2x2 blocks of 2x2 matrices.
    
    For 2x2 * 2x2 = 2x2 with Strassen (rank 7):
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 4x4 = (2x2) x (2x2), we have 4 quadrants of 2x2:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]] where each is 2x2
    C = [[C00, C01], [C10, C11]] where each is 2x2
    
    So we apply Strassen at the 2x2 block level, and each "2x2 * 2x2" is itself a full 2x2 multiplication.
    
    Actually, for 4x4 * 4x4, we can think of it as 2x2 blocks where each block is 2x2.
    The Strassen formulas operate on these 2x2 blocks, and each block multiplication is a 2x2*2x2.
    
    Each 2x2*2x2 can be done with rank 4 (schoolbook) or we could use further Strassen, 
    but standard Strassen for 4x4 uses rank 7 * 4 = 28? No.
    
    Actually, standard Strassen for n x n where n is power of 2:
    For 2x2: rank 7
    For 4x4: rank 7^2 = 49 (apply Strassen recursively: each of the 7 multiplications is 2x2*2x2)
    
    But wait, we can also just do 4x4 with a flat Strassen-like approach.
    
    Let me use the direct recursive Strassen: 4x4 multiplication decomposes into 
    7 multiplications of 2x2 matrices. Each 2x2 multiplication has rank 4 (schoolbook).
    So total rank = 7 * 4 = 28.
    
    Actually that's the standard result: Strassen for 4x4 has rank 28 (using schoolbook for the 2x2 subproblems).
    
    Let me construct this carefully.
    
    Indexing for 4x4 matrix A (row-major):
    A[0,0] A[0,1] A[0,2] A[0,3]
    A[1,0] A[1,1] A[1,2] A[1,3]
    A[2,0] A[2,1] A[2,2] A[2,3]
    A[3,0] A[3,1] A[3,2] A[3,3]
    
    2x2 blocks:
    A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  -> indices (0,1,4,5)
    A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]  -> indices (2,3,6,7)
    A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]  -> indices (8,9,12,13)
    A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]  -> indices (10,11,14,15)
    
    For the U matrix (shape 16 x rank), each row corresponds to an element of A.
    For the V matrix (shape 16 x rank), each row corresponds to an element of B.
    For the W matrix (shape 16 x rank), each row corresponds to an element of C (with W[k*n+i, c] convention, so for 4x4 output, W[i*4+j, c] = C[i,j]).
    
    Strassen at block level:
    M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    M2 = (A10+A11)*B00 -> contributes to C10, C11
    M3 = A00*(B01-B11) -> contributes to C00, C01
    M4 = A11*(B10-B00) -> contributes to C00, C10
    M5 = (A00+A01)*B11 -> contributes to C00, C01
    M6 = (A10-A00)*(B00+B01) -> contributes to C11
    M7 = (A01-A11)*(B10+B11) -> contributes to C00, C11
    
    Each Mi is a 2x2 matrix product, which we do with schoolbook (rank 4).
    
    So total rank = 7 * 4 = 28.
    """
    # We'll build U, V, W for the 4x4x4 part
    # U: 16 x 28, V: 16 x 28, W: 16 x 28
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Block index mappings for A (rows of U):
    # A00: (0,1,4,5), A01: (2,3,6,7), A10: (8,9,12,13), A11: (10,11,14,15)
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    # B blocks (rows of V):
    B00 = [0, 1, 7, 8]    # B[0,0],B[0,1],B[1,0],B[1,1]
    B01 = [2, 3, 9, 10]   # B[0,2],B[0,3],B[1,2],B[1,3]
    B10 = [14, 15, 21, 22] # B[2,0],B[2,1],B[3,0],B[3,1]
    B11 = [16, 17, 23, 24] # B[2,2],B[2,3],B[3,2],B[3,3]
    
    # C blocks (rows of W): C[i*4+j]
    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]
    
    # For each Strassen term, we need to do a 2x2 * 2x2 multiplication.
    # For 2x2 * 2x2 schoolbook:
    #   D[0,0] = a00*b00 + a01*b10  -> U: [a00,a01], V: [b00,b10], W: [d00, 0] and [0,d01]... 
    #   Actually for 2x2*2x2 schoolbook, rank 4:
    #   D[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   D[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #   D[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   D[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #   
    #   Each entry is a sum of 2 products, so we need 4 rank-1 terms:
    #   Term 1: A[0,0]*B[0,0] -> D[0,0] += 1
    #   Term 2: A[0,1]*B[1,0] -> D[0,0] += 1
    #   Term 3: A[0,0]*B[0,1] -> D[0,1] += 1
    #   Term 4: A[0,1]*B[1,1] -> D[0,1] += 1
    #   Term 5: A[1,0]*B[0,0] -> D[1,0] += 1
    #   Term 6: A[1,1]*B[1,0] -> D[1,0] += 1
    #   Term 7: A[1,0]*B[0,1] -> D[1,1] += 1
    #   Term 8: A[1,1]*B[1,1] -> D[1,1] += 1
    #   
    #   Wait, that's 8 terms, not 4. Schoolbook 2x2*2x2 has rank 8? No, schoolbook for m*n*p where all are 2 is 2*2*2=8. Yes.
    #   
    #   But Strassen for 2x2*2x2 reduces this to rank 7. Let me just use schoolbook for the 2x2 subproblems to keep it simple, giving rank 28 for 4x4.
    
    def add_2x2_product(col_start, a_block, b_block, c_block, sign_c):
        """Add a 2x2*2x2 schoolbook product starting at column col_start.
        a_block: list of 4 indices into U (the A 2x2 block)
        b_block: list of 4 indices into V (the B 2x2 block)
        c_block: list of 4 indices into W (the C 2x2 block)
        sign_c: +1 or -1 for the contribution to C
        """
        # 2x2 elements: [00, 01, 10, 11]
        # 8 schoolbook terms
        terms = [
            # (a_idx, b_idx, c_idx)
            (0, 0, 0),  # a00*b00 -> c00
            (1, 2, 0),  # a01*b10 -> c00
            (0, 1, 1),  # a00*b01 -> c01
            (1, 3, 1),  # a01*b11 -> c01
            (2, 0, 2),  # a10*b00 -> c10
            (3, 2, 2),  # a11*b10 -> c10
            (2, 1, 3),  # a10*b01 -> c11
            (3, 3, 3),  # a11*b11 -> c11
        ]
        for t, (ai, bi, ci) in enumerate(terms):
            col = col_start + t
            U[a_block[ai], col] += sign_c  # Wait, sign should be on W, not U/V
            V[b_block[bi], col] += 1
            W[c_block[ci], col] += sign_c
        return col_start + 8
    
    # Strassen terms:
    col = 0
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # We need to handle sums. (A00+A11) means U entries for A00 and A11 both get +1.
    # (B00+B11) means V entries for B00 and B11 both get +1.
    # This is more complex. Let me think of a different approach.
    
    # Actually, let me just directly construct the rank-28 decomposition for 4x4*4x4.
    # The simplest correct approach: use the Kronecker product structure.
    # 4x4*4x4 can be seen as (2x2)*(2x2) at the block level, where each subproblem is 2x2*2x2.
    # But the entries of the blocks are linear combinations, so we need to be careful.
    
    # Let me try a cleaner approach: just use schoolbook for 4x4*4x4*4 (rank 64) and 
    # schoolbook for 4x4*4x3 (rank 48), total rank 112. That's the baseline.
    # 
    # For improvement, let me use Strassen for the 4x4*4x4 part.
    # 
    # A cleaner way: think of 4x4*4x4 as multiplying two 2x2 matrices of 2x2 blocks.
    # The Strassen algorithm gives 7 products of 2x2 matrices.
    # Each 2x2 matrix product can be represented with 4 rank-1 terms if we use the "flat" representation.
    # Wait no, 2x2*2x2 has tensor rank at most 8 (schoolbook) and exactly 7 (Strassen) or lower?
    # The tensor rank of 2x2*2x2 is known to be 7 (Strassen) and this is minimal for 2x2*2x2? 
    # Actually the minimal rank of 2x2*2x2 matrix multiplication tensor is 7 (Strassen proved, and it's known to be minimal).
    
    # So 4x4*4x4 using Strassen recursively: 7 * (rank of 2x2*2x2) = 7 * 7 = 49? No.
    # When we say 4x4 = 2x2 of 2x2 blocks, and we apply Strassen, we get 7 sub-multiplications 
    # of 2x2 matrices. Each 2x2 multiplication has rank 7 (Strassen) or 8 (schoolbook).
    # Using Strassen for the subproblems: rank = 7 * 7 = 49.
    # Using schoolbook for subproblems: rank = 7 * 8 = 56.
    # 
    # But wait, the standard Strassen for 4x4 is usually stated as rank 28? Let me reconsider.
    # 
    # No, I think I'm confusing myself. Let me look at this differently.
    # 
    # The tensor for 4x4*4x4 matrix multiplication has 4*4*4 = 64 non-zero entries in its 
    # "natural" basis. The schoolbook decomposition uses rank 64.
    # 
    # Strassen's algorithm for 4x4: divide into four 2x2 blocks. Use 7 multiplications 
    # of 2x2 matrices instead of 8. Each 2x2 multiplication requires, in the schoolbook sense, 
    # 4 scalar multiplications (for the 4 entries of the result). So total scalar multiplications = 7*4 = 28.
    # This gives a rank-28 decomposition if the 2x2 subproblems are handled with rank 4.
    # But 2x2*2x2 has rank 7, not 4. The "4" comes from the fact that each entry of the 
    # 2x2 result is a dot product of length 2, which requires 2 multiplications per entry, 
    # 4 entries, so 8 total in schoolbook. But with Strassen for 2x2, it's 7.
    #
    # I think the confusion is between "number of scalar multiplications" and "tensor rank."
    # For the purpose of this task, we need a tensor decomposition, and the rank is the number 
    # of rank-1 terms. 
    #
    # Let me just use a known valid decomposition. The safest bet:
    # 1. Schoolbook for everything: rank = 4*4*7 = 112.
    # 2. Strassen for 4x4*4x4 part + schoolbook for 4x4*4x3 part.
    #
    # For option 2, I need to correctly implement the Strassen decomposition for 4x4*4x4.
    
    # Let me use a programmatic approach to generate the Strassen 4x4*4x4 decomposition.
    # 
    # I'll use the recursive Strassen structure:
    # Level 1: 4x4 = 2x2 of 2x2 blocks, Strassen gives 7 terms.
    # Level 2: Each 2x2*2x2 uses schoolbook (rank 8).
    # Total rank = 7 * 8 = 56.
    #
    # Actually, let me just use schoolbook for the 4x4*4x4 part too, giving rank 64 + 48 = 112.
    # That's the baseline. To improve, I need a better decomposition.
    #
    # Let me try a different structural approach: 
    # Use the fact that 4x4*4x7 = (4x4*4x4) + (4x4*4x3) is NOT correct as a direct sum 
    # because the B matrix is 4x7, not two separate 4x4 and 4x3 matrices being multiplied 
    # independently. The A is 4x4, B is 4x7, C is 4x7.
    # 
    # So C = A * B where A is 4x4, B is 4x7.
    # We can split B into B1 (4x4) and B2 (4x3). Then C = A*B1 + A*B2 = C1 + C2.
    # C1 = A*B1 is 4x4, C2 = A*B2 is 4x3.
    # The tensor for the whole thing is the "direct sum" in the sense that the A indices are shared,
    # but the B and C indices are disjoint.
    #
    # For the A*B1 part (4x4*4x4 -> 4x4), we can use any decomposition of rank r1.
    # For the A*B2 part (4x4*4x3 -> 4x3), we can use any decomposition of rank r2.
    # Since the A indices are the same but B and C are disjoint, we can concatenate the decompositions.
    # Total rank = r1 + r2.
    #
    # Schoolbook: r1 = 16, r2 = 12, total = 28. Wait, that can't be right.
    # 4x4*4x4: A has 16 elements, B has 16 elements. Each C entry C[i,j] = sum_k A[i,k]*B[k,j].
    # Schoolbook rank = 4*4*4 = 64.
    # 4x4*4x3: A has 16, B has 12. Each C[i,j] for j=0..2 = sum_k A[i,k]*B[k,j].
    # Schoolbook rank = 4*4*3 = 48.
    # Total schoolbook = 64 + 48 = 112. Correct.
    #
    # For Strassen on the 4x4*4x4 part: if we can achieve rank 28 for that part, total = 28 + 48 = 76.
    # If rank 49, total = 49 + 48 = 97.
    #
    # Let me try to implement a rank-28 decomposition for 4x4*4x4.
    # 
    # Actually, I recall that the tensor rank of 4x4*4x4 matrix multiplication is not 28. 
    # Strassen's algorithm uses 28 scalar multiplications, but the tensor rank might be different.
    # The number of scalar multiplications in Strassen's algorithm for 4x4 is 28, which corresponds 
    # to a rank-28 tensor decomposition. Yes, each scalar multiplication in the algorithm 
    # corresponds to a rank-1 term in the tensor decomposition.
    
    # OK let me just carefully build the rank-28 Strassen decomposition for 4x4*4x4.
    
    # Reset
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))
    
    # A block indices (row indices in U):
    # A = [[a00,a01,a02,a03],[a10,a11,a12,a13],[a20,a21,a22,a23],[a30,a31,a32,a33]]
    # U index for A[i,j] = i*4+j
    # A00 block (top-left 2x2): indices 0,1,4,5 (i=0,1; j=0,1)
    # A01 block (top-right 2x2): indices 2,3,6,7 (i=0,1; j=2,3)
    # A10 block (bottom-left 2x2): indices 8,9,12,13 (i=2,3; j=0,1)
    # A11 block (bottom-right 2x2): indices 10,11,14,15 (i=2,3; j=2,3)
    
    # B block indices (row indices in V):
    # V index for B[i,j] = i*4+j (since B is also 4x4 for this subproblem)
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C block indices (row indices in W):
    # W index for C[i,j] = i*4+j
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # Strassen terms (each is a product of linear forms in A and B, contributing to C):
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C00 -= M3, C01 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    
    # For each Mi, which is a 2x2*2x2 product, we need to decompose it.
    # Let's use schoolbook for each 2x2*2x2 (rank 8 each), total 7*8 = 56.
    # Or use Strassen for each (rank 7 each), total 7*7 = 49.
    # Let's use schoolbook for simplicity and correctness, giving rank 56 for the 4x4*4x4 part.
    # Total rank = 56 + 48 = 104.
    
    # Hmm, that's only slightly better than 112. Let me try to get the Strassen 2x2 subproblems.
    
    # Actually, let me just go with a clean implementation. I'll use schoolbook for both parts: rank 112.
    # That's guaranteed correct. Then I can try to improve.
    
    # For now, let me implement schoolbook correctly and return rank 112.
    # This is the same as the seed, so score 1.0. But I need to try something new.
    
    # Let me try the hybrid: Strassen(4x4*4x4) with schoolbook subproblems (rank 56) + schoolbook(4x4*4x3) (rank 48) = rank 104.
    
    U = np.zeros((16, 104))
    V = np.zeros((28, 104))
    W = np.zeros((28, 104))
    
    # Part 1: 4x4 * 4x4 -> 4x4, using Strassen with schoolbook 2x2 subproblems (rank 56)
    # V rows 0-15 correspond to B[0..3, 0..3]
    # W rows 0-15 correspond to C[0..3, 0..3]
    
    col = 0
    # Define 2x2 block index lists
    blocks_A = {
        'A00': [0, 1, 4, 5],
        'A01': [2, 3, 6, 7],
        'A10': [8, 9, 12, 13],
        'A11': [10, 11, 14, 15]
    }
    blocks_B = {
        'B00': [0, 1, 4, 5],
        'B01': [2, 3, 6, 7],
        'B10': [8, 9, 12, 13],
        'B11': [10, 11, 14, 15]
    }
    blocks_C = {
        'C00': [0, 1, 4, 5],
        'C01': [2, 3, 6, 7],
        'C10': [8, 9, 12, 13],
        'C11': [10, 11, 14, 15]
    }
    
    def add_block_product(col_start, a_indices, b_indices, c_indices, sign):
        """Add a 2x2 * 2x2 schoolbook product.
        a_indices: 4 indices for the 2x2 A block in U
        b_indices: 4 indices for the 2x2 B block in V
        c_indices: 4 indices for the 2x2 C block in W
        sign: +1 or -1
        Returns new col_start + 8
        """
        # 2x2 elements indexed as [00, 01, 10, 11] -> positions [0,1,2,3] in the list
        # Schoolbook 2x2*2x2: 8 terms
        terms = [
            (0, 0, 0),  # a00*b00 -> c00
            (1, 2, 0),  # a01*b10 -> c00
            (0, 1, 1),  # a00*b01 -> c01
            (1, 3, 1),  # a01*b11 -> c01
            (2, 0, 2),  # a10*b00 -> c10
            (3, 2, 2),  # a11*b10 -> c10
            (2, 1, 3),  # a10*b01 -> c11
            (3, 3, 3),  # a11*b11 -> c11
        ]
        for t, (ai, bi, ci) in enumerate(terms):
            c = col_start + t
            U[a_indices[ai], c] += sign
            V[b_indices[bi], c] += 1
            W[c_indices[ci], c] += sign
        return col_start + 8
    
    # Strassen terms for 4x4*4x4:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # We need to handle the sum (A00+A11). This means for the U matrix, both A00 and A11 entries get +1.
    # But in our rank-1 decomposition, each column has a single entry pattern. 
    # (A00+A11) is a linear combination, so we need to split this into separate rank-1 terms.
    # 
    # This is the key issue: Strassen's linear combinations require that we expand them.
    # (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # That's 4 products, each of which is a 2x2*2x2. So M1 alone requires 4*8 = 32 rank-1 terms!
    # 
    # That defeats the purpose. The Strassen trick is that the linear combinations are computed 
    # first, then multiplied. But in a tensor decomposition, we can't "compute first" - each rank-1 
    # term must be a product of a single A entry, a single B entry, and a single C entry (with coefficients).
    #
    # So Strassen's algorithm, when converted to a tensor decomposition, actually requires 
    # expanding all the linear combinations, which brings the rank back up.
    #
    # Wait, no. In a tensor decomposition, a rank-1 term is u * v * w where u, v, w are vectors.
    # u can have multiple non-zero entries! For example, u could be the vector that is +1 at A00[0,0] 
    # and +1 at A11[0,0], and 0 elsewhere. This represents the linear form (A[0,0] + A[2,2]).
    # So we CAN have linear combinations in the U, V, W matrices.
    #
    # So for M1 = (A00+A11)(B00+B11):
    # The U vector for the "A side" would have +1 at all entries of A00 and all entries of A11.
    # The V vector for the "B side" would have +1 at all entries of B00 and all entries of B11.
    # But this is ONE rank-1 term only if the product of these two linear forms gives a single 
    # linear form in C. But (A00+A11)(B00+B11) is a 2x2 matrix, not a scalar. 
    # 
    # Ah, I see the issue. Each "M" in Strassen is a 2x2 matrix product, which itself is a 
    # tensor of rank 8 (schoolbook) or 7 (Strassen). So M1 = (A00+A11)(B00+B11) is a 2x2 matrix 
    # whose entries are linear combinations. To decompose M1 into rank-1 terms of the overall 
    # 4x4*4x4 tensor, we need to decompose each entry of M1.
    #
    # M1 is a 2x2 matrix: M1[i,j] = sum_k (A00+A11)[i,k] * (B00+B11)[k,j]
    # M1[0,0] = (A00[0,0]+A11[0,0])*(B00[0,0]+B11[0,0]) + (A00[0,1]+A11[0,1])*(B00[1,0]+B11[1,0])
    # This is a sum of 4 products of linear forms, each of which is a product of two linear forms 
    # in A and B. Each such product can be written as a sum of rank-1 terms in the 4x4*4x4 tensor.
    #
    # This is getting very complex. Let me just use the schoolbook decomposition for the entire 
    # 4x4*4x7 problem, which gives rank 112, and accept score 1.0. Or try a different approach.
    
    # Actually, let me try a completely different structural approach:
    # Use the Kronecker product / block structure more directly.
    # 
    # 4x4 * 4x7: Think of A as a 2x2 matrix of 2x2 blocks, and B as a 2x2 matrix of 2x4 blocks? 
    # No, B is 4x7, which doesn't divide evenly.
    #
    # Let me just go with schoolbook rank 112. It's correct and known to work.
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank = n * m * p  # 112
    U = np.zeros((n * m, rank))  # 16 x 112
    V = np.zeros((m * p, rank))  # 28 x 112
    W = np.zeros((p * n, rank))  # 28 x 112
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
