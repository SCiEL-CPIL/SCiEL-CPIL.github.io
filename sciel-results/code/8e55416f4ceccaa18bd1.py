"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Strategy: split B (4x9) into three 4x3 blocks.
For each 4x3 block, use a 2x2 block Strassen-like decomposition where
the 2x3 sub-block multiplication is done via 3 schoolbook terms.
Total: 3 blocks * 7 terms = 21 rank.

Actually, let me reconsider. The standard approach:
- Split B into B1 (4x4), B2 (4x4), B3 (4x1)
- For 4x4 * 4x4: use Strassen (rank 49)
- For 4x4 * 4x1: schoolbook (rank 16)
- Total: 49 + 49 + 16 = 114

But we can do better by splitting B into three 4x3 blocks and using
a modified Strassen for 4x4 * 4x3.

For 4x4 * 4x3, split into 2x2 blocks of A and 2x2/2x1 blocks of B:
A = [[A11, A12], [A21, A22]]  (each 2x2)
B = [[B11, B12, B13], [B21, B22, B23]]  (each 2x2 or 2x1)

Using Strassen-style:
M1 = (A11+A22)(B11+B21)  -> C11: +1, C22: +1  (but B is 4x3, so this needs care)

Actually, the simplest improvement: use the 4x4*4x9 directly with
a rank-114 construction but verify it's correct.

Let me just implement the rank-114 construction cleanly:
- B split into B1(4x4), B2(4x4), B3(4x1)
- Strassen for each 4x4*4x4: rank 49 each
- Schoolbook for 4x4*4x1: rank 16
- Total: 114
"""

import time


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen for 4x4*4x4 blocks and schoolbook for 4x4*4x1 block.
    Total rank: 49 + 49 + 16 = 114.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n, m, p = 4, 4, 9
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (36, rank)
    # W shape: (p*n, rank) = (36, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

    U_cols = []  # Each element is a list of length 16
    V_cols = []  # Each element is a list of length 36
    W_cols = []  # Each element is a list of length 36

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # 7 Strassen terms for 2x2 * 2x2 -> 2x2
    # Each: (A_mult, B_mult, C_mult) where each is dict {(row,col): coeff}
    # The product A_mult * B_mult contributes to C via C_mult
    strassen_2x2 = [
        # s1 = (a11+a22)(b11+b22): c11 += s1, c22 += s1
        (({(0,0): 1, (1,1): 1},
          {(0,0): 1, (1,1): 1},
          {(0,0): 1, (1,1): 1})),
        # s2 = (a21+a22)b11: c21 += s2, c22 -= s2
        (({(1,0): 1, (1,1): 1},
          {(0,0): 1},
          {(1,0): 1, (1,1): -1})),
        # s3 = a11(b12-b22): c12 += s3, c22 += s3
        (({(0,0): 1},
          {(0,1): 1, (1,1): -1},
          {(0,1): 1, (1,1): 1})),
        # s4 = a22(b21-b11): c11 += s4, c21 += s4
        (({(1,1): 1},
          {(1,0): 1, (0,0): -1},
          {(0,0): 1, (1,0): 1})),
        # s5 = (a11+a12)b22: c11 -= s5, c12 += s5
        (({(0,0): 1, (0,1): 1},
          {(1,1): 1},
          {(0,0): -1, (0,1): 1})),
        # s6 = (a21-a11)(b11+b12): c22 += s6
        (({(1,0): 1, (0,0): -1},
          {(0,0): 1, (0,1): 1},
          {(1,1): 1})),
        # s7 = (a12-a22)(b21+b22): c11 += s7
        (({(0,1): 1, (1,1): -1},
          {(1,0): 1, (1,1): 1},
          {(0,0): 1})),
    ]

    def strassen_4x4_block(a_row_off, b_col_off, c_col_off):
        """Add Strassen terms for multiplying 4x4 A by 4x4 B block.
        
        a_row_off: row offset in A (0 for top 2x2 block, 2 for bottom)
        b_col_off: column offset in B block
        c_col_off: column offset in C
        """
        for a_coeffs, b_coeffs, c_coeffs in strassen_2x2:
            u = [0] * 16
            v = [0] * 36
            w = [0] * 36

            # A part: 2x2 block starting at (a_row_off, 0) in A
            for (ar, ac), coeff_a in a_coeffs.items():
                for (br, bc), coeff_b in b_coeffs.items():
                    # A index: (a_row_off + ar, ac) -> global (a_row_off + ar, ac)
                    ai = a_row_off + ar
                    aj = ac
                    # B index: (br, b_col_off + bc) -> global row br, col b_col_off + bc
                    bj = br
                    bk = b_col_off + bc
                    # C index: (ar, c_col_off + bc) -> global row ar, col c_col_off + bc
                    ci = ar
                    ck = c_col_off + bc

                    u[ai * m + aj] += coeff_a
                    v[bj * p + bk] += coeff_b
                    w[ck * n + ci] += coeff_a * coeff_b * c_coeffs.get((ar, bc), 0)
                    # Wait, this is wrong. Let me redo.

            # Actually, the Strassen construction works as:
            # For each term t, we have:
            #   U_vec: linear combination of A elements
            #   V_vec: linear combination of B elements
            #   W_vec: the product contributes to C elements
            # The decomposition is: sum_t U_t * V_t = W (tensor)
            # So for each (i,j,k) in C: sum_t U[i,j]_t * V[j,k]_t = W[k,i]_t contribution

            # Let me restart this more carefully.

            break  # We'll redo below

    # Let me implement this more carefully from scratch.
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()

    def add_strassen_4x4(A_row_block, B_col_block, C_col_block):
        """Add 49 Strassen terms for 4x4 * 4x4 multiplication.
        
        A_row_block: 0 or 2 (which 2x2 row block of A, but A is 4x4 so we use both)
        Actually A is always the full 4x4. Let me restructure.
        """
        pass

    # Simpler approach: directly enumerate all 114 terms.
    # 
    # For 4x4 * 4x4 using Strassen, we have 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    #
    # Strassen gives 7 products of 2x2 matrices, each contributing to multiple C blocks.
    # Each 2x2 * 2x2 = 4 terms (schoolbook), so 7 * 4 = 28... 
    # No wait, each Strassen term is a rank-1 in the 2x2*2x2->2x2 tensor,
    # and each 2x2*2x2 has rank 7, so 7 terms per block multiplication.
    # But we need to compose these.

    # Actually, the standard Strassen for 4x4*4x4:
    # Split into 2x2 blocks. Each Strassen term involves products like (A00+A11)(B00+B11).
    # This is a 2x2 * 2x2 multiplication, which itself needs 4 rank-1 terms (schoolbook)
    # OR can be further decomposed.
    #
    # No - in the tensor decomposition framework, each "term" in Strassen is itself
    # a rank-1 decomposition of the 2x2*2x2->2x2 multiplication tensor restricted
    # to linear combinations.
    #
    # Let me think about it differently. The 4x4*4x4*4x4 tensor has 64*64*64 = 262144 entries.
    # Rank 49 means 49 terms, each with U (16-dim), V (16-dim), W (16-dim) vectors.
    #
    # The standard construction:
    # Use 2x2 block Strassen. The 7 Strassen combinations each produce a 2x2 result.
    # Each 2x2 result is placed in specific positions in the 4x4 C matrix.
    # But each "2x2 multiplication" in Strassen is itself a tensor of size 4*4*4 = 64,
    # and it has rank 7 (itself using Strassen) or rank 4 (schoolbook).
    #
    # If we use schoolbook for the inner 2x2*2x2: 7 * 4 = 28 terms. But that gives rank 28 for 4x4*4x4?
    # No, that's not right either.
    #
    # Let me just look up: rank of 4x4*4x4 matrix multiplication is 49.
    # The construction uses 2x2 Strassen recursively or a specific 49-term construction.

    # I'll use the known rank-49 construction for 4x4*4x4.
    # This can be derived from the 2x2 Strassen by applying it to 2x2 blocks
    # and then using schoolbook for the resulting 2x2 multiplications.
    # 
    # Actually, the cleanest way: the 4x4*4x4 tensor can be decomposed using
    # the fact that it's a "convolution" of 2x2 structures.
    #
    # Let me just hard-code the 49 terms using the standard method:
    # Apply Strassen at the 2x2 level, where each "multiplication" of 2x2 blocks
    # is done via 4 schoolbook terms.
    # 7 Strassen terms * 4 schoolbook per term = 28. That's not 49.
    #
    # I think I'm confusing myself. Let me use a different approach.
    #
    # The rank of n x n matrix multiplication:
    # 2x2: rank 7 (Strassen)
    # 4x4: rank 49 (known, from Alon et al. or similar)
    #
    # The 49-term construction for 4x4*4x4 can be obtained by:
    # Viewing 4x4 as 2x2 blocks of 2x2, and using the tensor product structure.
    #
    # Alternatively, I'll just implement the 114-term construction directly
    # by combining Strassen-2x2 for the outer structure and schoolbook for inner.

    # Let me use a cleaner formulation. 
    # 
    # For 4x4 * 4x4:
    # Write A, B, C as 2x2 blocks of 2x2 matrices.
    # A = [[A_00, A_01], [A_10, A_11]]
    # B = [[B_00, B_01], [B_10, B_11]]  
    # C = [[C_00, C_01], [C_10, C_11]]
    #
    # C_ij = sum_k A_ik * B_kj (matrix multiplication of 2x2 blocks)
    #
    # Using Strassen for the outer 2x2 structure:
    # M1 = (A_00 + A_11)(B_00 + B_11)
    # M2 = (A_10 + A_11) * B_00
    # M3 = A_00 * (B_01 - B_11)
    # M4 = A_11 * (B_10 - B_00)
    # M5 = (A_00 + A_01) * B_11
    # M6 = (A_10 - A_00)(B_00 + B_01)
    # M7 = (A_01 - A_11)(B_10 + B_11)
    #
    # C_00 = M1 - M2 + M4 + M5 - M7  (wait, let me check)
    # Actually:
    # C_00 = M1 + M4 - M2 + M5 - M7  -- no
    #
    # Standard Strassen:
    # C_00 = M1 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M3 + M6
    
    # Each M_i is a 2x2 * 2x2 = 2x2 matrix product.
    # Each such product has 4 rank-1 terms (schoolbook).
    # So total: 7 * 4 = 28 terms.
    #
    # But wait, that gives rank 28 for 4x4*4x4, which contradicts the known rank of 49.
    # 
    # I think the issue is that when I say "M1 = (A_00 + A_11)(B_00 + B_11)", 
    # this is a 2x2 matrix multiplication, and in the tensor decomposition,
    # each "scalar" multiplication within it becomes a term.
    #
    # So M1 as a 2x2*2x2->2x2 tensor has 4 terms (schoolbook):
    # (A_00+A_11)[r,s] * (B_00+B_11)[s,t] contributes to (M1)[r,t]
    #
    # So the full decomposition:
    # For each of the 7 Strassen combinations, for each of the 4 schoolbook terms:
    # Total = 7 * 4 = 28 terms.
    #
    # Hmm, but that should give rank 28 for 4x4*4x4. Let me verify...
    # Actually I think rank(4x4) might indeed be less than 49. Let me check.
    # 
    # Known bounds: 4x4 matrix multiplication has rank at most 49 (Bini et al.)
    # and at least 49 (Alon). So rank = 49.
    #
    # My Strassen-based construction gives 28 terms, which would mean rank <= 28.
    # That can't be right. Let me recount.
    #
    # Oh wait, I think the issue is that in the Strassen construction for 2x2,
    # the "multiplications" M1-M7 are scalar multiplications, not matrix multiplications.
    # When we go to 4x4 with 2x2 blocks, each "scalar" becomes a 2x2 matrix,
    # and each "scalar multiplication" becomes a 2x2*2x2 matrix multiplication.
    #
    # So the tensor for 4x4*4x4 is built from the 2x2 Strassen tensor,
    # where each entry is itself a 2x2*2x2->2x2 tensor.
    # The rank multiplies: rank(4x4) <= rank(2x2_strassen) * rank(2x2_schoolbook)
    # = 7 * 4 = 28? 
    #
    # No, that's not how tensor rank composes. The tensor product of two tensors
    # has rank at most the product of ranks, but the actual construction is different.
    #
    # Let me just directly construct the terms.
    #
    # For 4x4 * 4x4, using 2x2 blocks and Strassen:
    # 
    # The 7 Strassen combinations, each involving 2x2 matrices:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11  
    # M3 = A00(B01-B11) -> contributes to C01, C11
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00

    # Each Mi is a 2x2 matrix. To decompose Mi = Pi * Qi (where Pi, Qi are 2x2),
    # we use 4 schoolbook terms:
    # Mi[r,t] = sum_s Pi[r,s] * Qi[s,t]
    
    # So for each Strassen term and each (r,s,t) in the 2x2*2x2->2x2:
    # We get a term in the 4x4*4x4*4x4 tensor.
    
    # Total terms: 7 Strassen * 4 schoolbook = 28.
    # 
    # But this should give a valid rank-28 decomposition of 4x4*4x4!
    # Unless I'm making an error in the Strassen formula.
    
    # Let me verify with a small example. If A and B are 4x4 with 2x2 blocks,
    # and we use the above construction, do we get C = A*B?
    
    # I think the answer is yes, and rank(4x4*4x4) <= 28.
    # But the literature says rank = 49. So I must be wrong somewhere.
    
    # OH WAIT. I think I see the issue. In the Strassen formula, 
    # M1 = (A00+A11)(B00+B11) is a 2x2 matrix multiplication.
    # But in the TENSOR DECOMPOSITION sense, we're not doing matrix multiplication
    # of the blocks. We're doing element-wise linear combinations.
    #
    # No, that's not right either. Let me think again...
    #
    # The matrix multiplication tensor T[i,j,k] for i,j,k in {0,1,2,3}
    # (where i is row of C, j is the contraction index, k is col of C... 
    # no, T[i,j,k] = delta[i,j]*delta[j,k] in some convention.
    
    # I'm overcomplicating this. Let me just use the known fact that
    # rank(4x4*4x4) = 49 and use a known construction, or just use
    # the straightforward approach that gives rank 114 for 4x4*4x9.
    
    # For now, let me just implement the 114-term construction correctly.
    # Split B into B1(4x4), B2(4x4), B3(4x1).
    # For 4x4*4x4: use 49-term construction (I'll derive it).
    # For 4x4*4x1: use 16 schoolbook terms.
    
    # Deriving 49-term construction for 4x4*4x4:
    # Actually, let me use the recursive Strassen: 
    # rank(4x4) <= 7 * rank(2x2) = 7*7 = 49.
    # 
    # So the 49-term construction IS the recursive Strassen!
    # Each of the 7 outer Strassen terms involves a 2x2*2x2 multiplication,
    # which is done via 7 inner Strassen terms.
    # Total: 7 * 7 = 49.

    # Let me implement this properly.
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # 2x2 Strassen: 7 terms, each with (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: {(r,s): coeff} for the 2x2 A block
    # B_coeffs: {(s,t): coeff} for the 2x2 B block  
    # C_coeffs: {(r,t): coeff} for the 2x2 C block
    s2 = [
        # (A00+A11)(B00+B11) -> C00 +, C11 +
        (({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})),
        # (A10+A11)B00 -> C10 +, C11 -
        (({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})),
        # A00(B01-B11) -> C01 +, C11 +
        (({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})),
        # A11(B10-B00) -> C00 +, C10 +
        (({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})),
        # (A00+A01)B11 -> C00 -, C01 +
        (({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})),
        # (A10-A00)(B00+B01) -> C11 +
        (({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})),
        # (A01-A11)(B10+B11) -> C00 +
        (({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})),
    ]

    def add_4x4_strassen(A_i_off, B_k_off, C_i_off, C_k_off):
        """Add 49 terms for 4x4*4x4 where A rows start at A_i_off,
        B cols start at B_k_off, C rows start at C_i_off, C cols start at C_k_off.
        
        A is 4x4, B is 4x4, C is 4x4.
        We split into 2x2 blocks and apply Strassen recursively.
        """
        for outer_A, outer_B, outer_C in s2:
            for inner_A, inner_B, inner_C in s2:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                # Outer Strassen combines 2x2 blocks of A (row-wise) 
                # and 2x2 blocks of B (col-wise), contributing to C blocks.
                # 
                # A block (or, br): or in {0,1}, br in {0,1} -> 2x2 block of A
                #   global A row: A_i_off + or*2 + ir, col: br*2 + ic
                # B block (or, bk): or in {0,1}, bk in {0,1} -> 2x2 block of B
                #   global B row: or*2 + ic, col: B_k_off + bk*2 + ic
                # C block (or, bk): 
                #   global C row: C_i_off + or*2 + ir, col: C_k_off + bk*2 + ic
                #
                # The outer Strassen term specifies linear combinations of A blocks and B blocks.
                # The inner Strassen term specifies how the resulting 2x2*2x2->2x2 is decomposed.
                
                # For the outer term, outer_A tells us which A blocks are combined,
                # outer_B tells us which B blocks are combined.
                # The "product" is a 2x2 (of blocks) * 2x2 (of blocks) = 2x2 (of blocks).
                # Each entry of this 2x2 result is a 2x2*2x2 matrix product.
                # The inner Strassen decomposes each such product into 7 terms.
                
                # But wait, the inner Strassen operates on the "scalar" entries of
                # the block-level 2x2 matrices. So we need to be more careful.
                
                # Let me reconsider. At the block level:
                # A has 4 blocks: A00, A01, A10, A11 (each 2x2)
                # B has 4 blocks: B00, B01, B10, B11 (each 2x2)
                # C has 4 blocks: C00, C01, C10, C11 (each 2x2)
                #
                # Outer Strassen:
                # M_outer = (comb_A) * (comb_B) where comb_A is a 2x2 of A-blocks,
                # comb_B is a 2x2 of B-blocks.
                # M_outer is a 2x2 of C-blocks.
                #
                # But M_outer is NOT a matrix product. It's just a formal combination.
                # The actual computation is: C = A*B where each entry is a sum of products.
                #
                # I think the correct interpretation is:
                # The 4x4*4x4->4x4 tensor can be viewed as a "tensor of tensors"
                # at the block level. The outer structure is 2x2*2x2->2x2 (Strassen, rank 7),
                # and each entry is a 2x2*2x2->2x2 tensor (also Strassen, rank 7).
                # The total rank is 7*7 = 49.
                #
                # To construct the terms:
                # For each outer Strassen term (oa, ob, oc) and inner Strassen term (ia, ib, ic):
                # The term contributes to specific (i,j,k) positions in the 4x4*4x4*4x4 tensor.
                
                # Let me just enumerate all possibilities.
                #
                # Outer term oa: {(or, ocol): coeff_oa} where or, ocol in {0,1}
                #   This specifies a linear combination of A blocks.
                #   A block (or, ocol) has elements A[A_i_off + or*2 + ir, ocol*2 + ic]
                #   for ir, ic in {0,1}.
                #
                # Outer term ob: {(or, ocol): coeff_ob}
                #   Linear combination of B blocks.
                #   B block (or, ocol) has elements B[or*2 + ir, B_k_off + ocol*2 + ic]
                #
                # Inner term ia: {(ir, ic): coeff_ia}
                #   Within each 2x2 sub-block, linear combination.
                #
                # Inner term ib: {(ir, ic): coeff_ib}
                # Inner term ic: {(ir, ic): coeff_ic}
                
                # For this combined term, the U vector (A part):
                # U[A_i_off + or*2 + ir, ocol*2 + ic] += coeff_oa[or,ocol] * coeff_ia[ir,ic]
                # for all (or, ocol) in oa and (ir, ic) in ia
                
                # V vector (B part):
                # V[or*2 + ir, B_k_off + ocol*2 + ic] += coeff_ob[or,ocol] * coeff_ib[ir,ic]
                
                # W vector (C part):
                # C block (oc_or, oc_ocol) gets contribution from the product.
                # The inner Strassen says: within the 2x2 block, 
                # C[oc_or*2 + ir, oc_ocol*2 + ic] += coeff_oc[oc_or, oc_ocol] * coeff_ic[ir,ic]
                # But we also need the coefficient from the outer term...
                
                # Hmm, I think the structure is:
                # The outer Strassen term produces a 2x2 of C-blocks, where each C-block
                # is the "product" of the corresponding A-block combination and B-block combination.
                # The inner Strassen then decomposes each 2x2*2x2->2x2 into 7 rank-1 terms.
                
                # So for the (oc_or, oc_ocol) C-block, the contribution is:
                # coeff_oc[oc_or, oc_ocol] * (inner product of the 2x2 blocks)
                # where the inner product is decomposed via inner Strassen.
                
                # Let me just code it up:
                
                for (oor, ooc), c_ao in outer_A.items():
                    for (ior, ioc), c_ia in inner_A.items():
                        ar = A_i_off + oor * 2 + ior
                        ac = ooc * 2 + ioc
                        u[ar * 4 + ac] += c_ao * c_ia
                
                for (oor, ooc), c_bo in outer_B.items():
                    for (ior, ioc), c_ib in inner_B.items():
                        br = oor * 2 + ior
                        bc = B_k_off + ooc * 2 + ioc
                        v[br * 9 + bc] += c_bo * c_ib
                
                for (oor, ooc), c_co in outer_C.items():
                    for (ior, ioc), c_ic in inner_C.items():
                        cr = C_i_off + oor * 2 + ior
                        cc = C_k_off + ooc * 2 + ioc
                        w[cc * 4 + cr] += c_co * c_ic
                
                add_term(u, v, w)

    if time.time() > deadline:
        # Fallback: schoolbook
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u = [0.0] * 16
                    v = [0.0] * 36
                    w = [0.0] * 36
                    u[i * 4 + j] = 1.0
                    v[j * 9 + k] = 1.0
                    w[k * 4 + i] = 1.0
                    add_term(u, v, w)
    else:
        # B split into B1(4x4, cols 0-3), B2(4x4, cols 4-7), B3(4x1, col 8)
        # 4x4 * B1: 49 terms
        add_4x4_strassen(0, 0, 0, 0)
        
        # 4x4 * B2: 49 terms
        add_4x4_strassen(0, 4, 0, 4)
        
        # 4x4 * B3: schoolbook, 16 terms
        for i in range(4):
            for j in range(4):
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                u[i * 4 + j] = 1.0
                v[j * 9 + 8] = 1.0
                w[8 * 4 + i] = 1.0
                add_term(u, v, w)
    
    rank = len(U_cols)
    
    # Build output matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(16)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(36)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(36)]
    
    return {"U": U, "V": V, "W": W}
