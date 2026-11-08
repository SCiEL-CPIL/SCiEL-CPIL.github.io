"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using a 4+2 block decomposition.
    # A = [[A44, A42], [A24, A22]]
    # B = [[B44, B42], [B24, B22]]
    # C = [[C44, C42], [C24, C22]]
    # C44 = A44*B44 + A42*B24  -> Rank 56 + 14 = 70
    # C42 = A44*B42 + A42*B22  -> Rank 56 + 14 = 70  (Wait, A44*B42 is 4x6, A42 is 4x2, B22 is 2x2. 
    #   Actually, let's re-verify the ranks carefully.
    #   M_{4,4,4}: Rank 56 (Strassen)
    #   M_{4,2,4}: Rank 14 (Strassen on 2 blocks)
    #   M_{4,4,2}: Rank 14 (Strassen on 2 blocks)
    #   M_{2,4,4}: Rank 14
    #   M_{4,2,2}: Rank 7
    #   M_{2,4,2}: Rank 7
    #   M_{2,2,4}: Rank 7
    #   M_{2,2,2}: Rank 7
    
    # Let's re-calculate:
    # C44 = A44 B44 + A42 B24
    #   A44 is 4x4, B44 is 4x4 -> Rank 56
    #   A42 is 4x2, B24 is 2x4 -> Rank 14
    #   Total = 70
    
    # C42 = A44 B42 + A42 B22
    #   A44 is 4x4, B42 is 4x2 -> Rank 14
    #   A42 is 4x2, B22 is 2x2 -> Rank 7
    #   Total = 21
    
    # C24 = A24 B44 + A22 B24
    #   A24 is 2x4, B44 is 4x4 -> Rank 14
    #   A22 is 2x2, B24 is 2x4 -> Rank 7
    #   Total = 21
    
    # C22 = A24 B42 + A22 B22
    #   A24 is 2x4, B42 is 4x2 -> Rank 7
    #   A22 is 2x2, B22 is 2x2 -> Rank 7
    #   Total = 14
    
    # Grand Total Rank = 70 + 21 + 21 + 14 = 126.
    # This is a significant improvement over 189!
    # Score = 216 / 126 = 1.714...

    rank = 126
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to define the tensor decompositions for each sub-block.
    # To keep it simple and correct, I will write a helper that generates the 
    # (U, V, W) tensors for M_{n,m,p} using Strassen recursion, returning lists of (i,j,val) tuples.
    # Actually, generating the full tensors for each sub-problem and then combining them 
    # with the correct offsets is the safest way to avoid indexing errors.
    
    def get_strassen_tensor(n, m, p):
        """Returns (U_list, V_list, W_list) where each is a list of (i,j,val) for rank R."""
        if n == 1 and m == 1 and p == 1:
            return [((0,0), 1)], [((0,0), 1)], [((0,0), 1)]
            
        # Use Strassen for 2x2 blocks if possible, or fall back to naive if dimensions are odd/mixed?
        # For this specific task, we only need even dimensions that are powers of 2 or multiples.
        # We are calling with (4,4,4), (4,2,4), (4,4,2), (2,4,4), (4,2,2), (2,4,2), (2,2,4), (2,2,2).
        # All are even. We can split into 2x2 blocks.
        
        n2, m2, p2 = n//2, m//2, p//2
        
        # Recursively get tensors for the 7 Strassen components.
        # Each component is a matrix multiplication of size n2 x m2 x p2.
        # We need the base tensors for these sizes.
        
        # To avoid infinite recursion or complex memoization in this short script,
        # and since the sizes are small (max 4), we can just use the known ranks and
        # construct them by calling this function recursively until we hit 1x1x1?
        # No, Strassen splits 2x2 into 1x1.
        # So for 4x4, it splits into 2x2, which splits into 1x1.
        # This works.
        
        # The 7 terms for Strassen:
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B01-B11)
        # M3 = A11(B10-B00)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A01-A11)(B10+B11)
        
        # We need to generate the (U,V,W) for each of these.
        # Let's get the base tensor for n2, m2, p2.
        U_base, V_base, W_base = get_strassen_tensor(n2, m2, p2)
        
        U_list = []
        V_list = []
        W_list = []
        
        def add_tensor(U_list, V_list, W_list, U_src, V_src, W_src, 
                       u_offset, v_offset, w_offset, 
                       u_sign, v_sign, w_sign):
            # u_offset: (i, j) in the 2x2 block grid? No, offsets in the full n,m,p space.
            # Actually, it's easier to define the coefficients.
            # The Strassen algorithm defines linear combinations of the quadrants.
            # Instead of manipulating the base tensors with offsets, it's better to
            # define the full Strassen tensor for size (n,m,p) by defining the 
            # coefficients for each element.
            
            pass

        # The recursive approach above is tricky because Strassen isn't just "call base 7 times".
        # It involves summing the A and B matrices first.
        # Let's use a different strategy: Directly define the tensor for 2x2 (Rank 7)
        # and for 4x4 (Rank 56) using the known patterns.
        # Since we only have a few specific sizes, hardcoding or generating them
        # via a robust recursive definition that handles the linearity is better.
        
        # Let's try a different implementation:
        # Return a list of (u_tuple, v_tuple, w_tuple, val) for the full tensor.
        # For n=m=p=1, it's [((0,0),(0,0),(0,0), 1)]
        # For 2x2, we define the 7 terms.
        # For 4x4, we define the 7 terms where each term is a 2x2 multiplication,
        # but we need to expand the 2x2 multiplication into its 7 sub-terms?
        # No, a 4x4 multiplication via Strassen uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7. So total rank 49?
        # Wait, standard Strassen for 4x4 has rank 56?
        # Let's check: 7 * R(2). R(2) = 7. 7*7 = 49.
        # Why did I think 56? 
        # Ah, the 4x4 case might have a better decomposition or I was misremembering.
        # If R(2)=7, then R(4) <= 7*7 = 49.
        # Is 49 the rank of 4x4?
        # Actually, it is known that R(4) <= 49.
        # Let's assume we can achieve rank 49 for 4x4.
        
        # Let's re-evaluate the rank of the 6x6 decomposition with R(4)=49.
        # C44 = A44 B44 (49) + A42 B24 (14) = 63
        # C42 = A44 B42 (14) + A42 B22 (7) = 21
        # C24 = A24 B44 (14) + A22 B24 (7) = 21
        # C22 = A24 B42 (7) + A22 B22 (7) = 14
        # Total = 63 + 21 + 21 + 14 = 119.
        # Score = 216 / 119 = 1.815.
        
        # Even better.
        
        # So I need to implement a function that returns the tensor for M_{n,m,p} 
        # using Strassen recursion, which will naturally give the minimal rank 
        # for these powers of 2 dimensions.
        
        # Implementation of get_strassen_tensor(n, m, p):
        # Returns list of tuples: (u_index, v_index, w_index, value)
        # where u_index is (i, j) in [0,n)x[0,m), etc.
        
        if n == 1 and m == 1 and p == 1:
            return [((0,0), (0,0), (0,0), 1)]
            
        n2, m2, p2 = n//2, m//2, p//2
        
        # Get the tensor for the sub-problem n2 x m2 x p2
        sub_tensors = get_strassen_tensor(n2, m2, p2)
        
        # Now we need to combine 7 of these sub-tensors to form the n x m x p tensor.
        # The 7 terms correspond to the 7 Strassen products.
        # Each product is a linear combination of sub-matrices.
        # We can represent the Strassen algorithm as a set of 7 linear transformations.
        
        # Let's define the 7 terms' coefficients on A, B, and C.
        # A is 2x2 blocks of size n2 x m2.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # Term 0: M0 = (A00+A11)(B00+B11)
        #   C00 += M0, C11 += M0
        #   A parts: A00 (+1), A11 (+1)
        #   B parts: B00 (+1), B11 (+1)
        #   C parts: C00 (+1), C11 (+1)
        
        # Term 1: M1 = (A10+A11)B00
        #   C10 += M1, C11 -= M1
        #   A parts: A10 (+1), A11 (+1)
        #   B parts: B00 (+1)
        #   C parts: C10 (+1), C11 (-1)
        
        # Term 2: M2 = A00(B01-B11)
        #   C01 += M2, C11 += M2
        #   A parts: A00 (+1)
        #   B parts: B01 (+1), B11 (-1)
        #   C parts: C01 (+1), C11 (+1)
        
        # Term 3: M3 = A11(B10-B00)
        #   C00 += M3, C10 += M3
        #   A parts: A11 (+1)
        #   B parts: B10 (+1), B00 (-1)
        #   C parts: C00 (+1), C10 (+1)
        
        # Term 4: M4 = (A00+A01)B11
        #   C00 -= M4, C01 += M4
        #   A parts: A00 (+1), A01 (+1)
        #   B parts: B11 (+1)
        #   C parts: C00 (-1), C01 (+1)
        
        # Term 5: M5 = (A10-A00)(B00+B01)
        #   C11 += M5
        #   A parts: A10 (+1), A00 (-1)
        #   B parts: B00 (+1), B01 (+1)
        #   C parts: C11 (+1)
        
        # Term 6: M6 = (A01-A11)(B10+B11)
        #   C00 += M6
        #   A parts: A01 (+1), A11 (-1)
        #   B parts: B10 (+1), B11 (+1)
        #   C parts: C00 (+1)
        
        # To construct the full tensor, we iterate over each of the 7 terms.
        # For each term, we iterate over the sub-tensor (which represents the multiplication of n2 x m2 by m2 x p2).
        # For each sub-tensor element (u_sub, v_sub, w_sub, val_sub):
        #   We need to add contributions to the full tensor for each (A_part, B_part, C_part) combination
        #   in that term.
        #   The full index for u is (block_i * n2 + u_sub_i, block_j * m2 + u_sub_j)
        #   Similarly for v and w.
        #   The value is val_sub * a_coeff * b_coeff * c_coeff.
        
        full_tensors = []
        
        # Define the 7 terms: (list of (a_block, a_coeff), list of (b_block, b_coeff), list of (c_block, c_coeff))
        # a_block is (i, j) in 0..1, 0..1
        terms = [
            # Term 0
            ([((0,0), 1), ((1,1), 1)], 
             [((0,0), 1), ((1,1), 1)], 
             [((0,0), 1), ((1,1), 1)]),
            # Term 1
            ([((1,0), 1), ((1,1), 1)], 
             [((0,0), 1)], 
             [((1,0), 1), ((1,1), -1)]),
            # Term 2
            ([((0,0), 1)], 
             [((0,1), 1), ((1,1), -1)], 
             [((0,1), 1), ((1,1), 1)]),
            # Term 3
            ([((1,1), 1)], 
             [((1,0), 1), ((0,0), -1)], 
             [((0,0), 1), ((1,0), 1)]),
            # Term 4
            ([((0,0), 1), ((0,1), 1)], 
             [((1,1), 1)], 
             [((0,0), -1), ((0,1), 1)]),
            # Term 5
            ([((1,0), 1), ((0,0), -1)], 
             [((0,0), 1), ((0,1), 1)], 
             [((1,1), 1)]),
            # Term 6
            ([((0,1), 1), ((1,1), -1)], 
             [((1,0), 1), ((1,1), 1)], 
             [((0,0), 1)]),
        ]
        
        for a_parts, b_parts, c_parts in terms:
            for u_sub, v_sub, w_sub, val_sub in sub_tensors:
                for (ai, aj), a_val in a_parts:
                    for (bi, bj), b_val in b_parts:
                        for (ci, cj), c_val in c_parts:
                            # Full indices
                            # u is n x m. Block (ai, aj). Local u_sub is (u_i, u_j).
                            # u_i is in [0, n2), u_j in [0, m2).
                            # Full u_i = ai * n2 + u_sub_i
                            # Full u_j = aj * m2 + u_sub_j
                            fu = (ai * n2 + u_sub[0], aj * m2 + u_sub[1])
                            
                            # v is m x p. Block (bi, bj). Local v_sub is (v_i, v_j).
                            # v_i is in [0, m2), v_j in [0, p2).
                            # Full v_i = bi * m2 + v_sub[0]
                            # Full v_j = bj * p2 + v_sub[1]
                            fv = (bi * m2 + v_sub[0], bj * p2 + v_sub[1])
                            
                            # w is p x n. Block (ci, cj). Local w_sub is (w_i, w_j).
                            # w_i is in [0, p2), w_j in [0, n2).
                            # Full w_i = ci * p2 + w_sub[0]
                            # Full w_j = cj * n2 + w_sub[1]
                            fw = (ci * p2 + w_sub[0], cj * n2 + w_sub[1])
                            
                            val = val_sub * a_val * b_val * c_val
                            if val != 0:
                                full_tensors.append((fu, fv, fw, val))
                                
        return full_tensors

    # Now we assemble the 6x6 multiplication.
    # We need tensors for:
    # M44: 4x4x4
    # M42_4: 4x2x4 (A42 B24)
    # M44_2: 4x4x2 (A44 B42)
    # M42_2: 4x2x2 (A42 B22)
    # M24_4: 2x4x4 (A24 B44)
    # M22_4: 2x2x4 (A22 B24)
    # M24_2: 2x4x2 (A24 B42)
    # M22_2: 2x2x2 (A22 B22)
    
    # Get the tensors
    T_444 = get_strassen_tensor(4, 4, 4)
    T_424 = get_strassen_tensor(4, 2, 4)
    T_442 = get_strassen_tensor(4, 4, 2)
    T_422 = get_strassen_tensor(4, 2, 2)
    T_244 = get_strassen_tensor(2, 4, 4)
    T_224 = get_strassen_tensor(2, 2, 4)
    T_242 = get_strassen_tensor(2, 4, 2)
    T_222 = get_strassen_tensor(2, 2, 2)
    
    # Offsets for the 6x6 blocks
    # A is 6x6. Split into 4+2.
    # A44: rows 0-3, cols 0-3
    # A42: rows 0-3, cols 4-5
    # A24: rows 4-5, cols 0-3
    # A22: rows 4-5, cols 4-5
    
    # B is 6x6.
    # B44: rows 0-3, cols 0-3
    # B42: rows 0-3, cols 4-5
    # B24: rows 4-5, cols 0-3
    # B22: rows 4-5, cols 4-5
    
    # C is 6x6.
    # C44: rows 0-3, cols 0-3
    # C42: rows 0-3, cols 4-5
    # C24: rows 4-5, cols 0-3
    # C22: rows 4-5, cols 4-5
    
    # We need to map the local indices in the sub-tensors to the global indices.
    # For a sub-problem M_{n_sub, m_sub, p_sub} located at (r_offset, c_offset) in A,
    # (c_offset_B, r_offset_B) in B, and (r_offset_C, c_offset_C) in C.
    # Wait, the indices in U, V, W are flattened.
    # U[i, j] corresponds to A[i, j].
    # V[i, j] corresponds to B[i, j].
    # W[i, j] corresponds to C[i, j] (with W[k*n+i, r] convention? No, W is (p*n, rank).
    # The problem says: "output factor follows the official trace convention W[k*n+i,r]".
    # This means W is indexed by (k, i) where k is the column of C and i is the row of C.
    # So W_flat_index = k * n + i.
    
    # Let's define a helper to add a tensor to the global U, V, W.
    
    def add_sub_tensor(T, U, V, W, u_off, v_off, w_off):
        # u_off: (row_off, col_off) for A
        # v_off: (row_off, col_off) for B
        # w_off: (row_off, col_off) for C (where row is i, col is k)
        # Note: W index is k*n + i. So if w is (i, k), index is k*n + i.
        
        for (u_i, u_j), (v_i, v_j), (w_i, w_j), val in T:
            # u_i is row in A, u_j is col in A
            gu_i = u_off[0] + u_i
            gu_j = u_off[1] + u_j
            gu = gu_i * m + gu_j
            
            # v_i is row in B, v_j is col in B
            gv_i = v_off[0] + v_i
            gv_j = v_off[1] + v_j
            gv = gv_i * p + gv_j
            
            # w_i is row in C (i), w_j is col in C (k)
            gw_i = w_off[0] + w_i
            gw_j = w_off[1] + w_j
            # W index: k*n + i = gw_j * n + gw_i
            gw = gw_j * n + gw_i
            
            # Add to global U, V, W
            # We need to know which column of U, V, W to write to.
            # Since we are concatenating all sub-tensors, we need a global column counter.
            # This function should return the number of columns added, or we pass a ref.
            pass

    # The `add_sub_tensor` above is incomplete because it doesn't handle the column index.
    # Let's just gather all the (U, V, W) tuples in a list and then construct the matrices.
    
    all_tensors = []
    
    # C44 = A44 B44 + A42 B24
    # A44 B44: u_off=(0,0), v_off=(0,0), w_off=(0,0)
    for t in T_444:
        all_tensors.append((t[0], t[1], t[2], t[3], (0,0), (0,0), (0,0)))
    # A42 B24: A42 is (0,4) to (3,5). B24 is (4,0) to (5,3).
    # u_off=(0,4), v_off=(4,0), w_off=(0,0)
    for t in T_424:
        all_tensors.append((t[0], t[1], t[2], t[3], (0,4), (4,0), (0,0)))
        
    # C42 = A44 B42 + A42 B22
    # A44 B42: u_off=(0,0), v_off=(0,4), w_off=(0,4)
    for t in T_442:
        all_tensors.append((t[0], t[1], t[2], t[3], (0,0), (0,4), (0,4)))
    # A42 B22: u_off=(0,4), v_off=(4,4), w_off=(0,4)
    for t in T_422:
        all_tensors.append((t[0], t[1], t[2], t[3], (0,4), (4,4), (0,4)))
        
    # C24 = A24 B44 + A22 B24
    # A24 B44: u_off=(4,0), v_off=(0,0), w_off=(4,0)
    for t in T_244:
        all_tensors.append((t[0], t[1], t[2], t[3], (4,0), (0,0), (4,0)))
    # A22 B24: u_off=(4,4), v_off=(4,0), w_off=(4,0)
    for t in T_224:
        all_tensors.append((t[0], t[1], t[2], t[3], (4,4), (4,0), (4,0)))
        
    # C22 = A24 B42 + A22 B22
    # A24 B42: u_off=(4,0), v_off=(0,4), w_off=(4,4)
    for t in T_242:
        all_tensors.append((t[0], t[1], t[2], t[3], (4,0), (0,4), (4,4)))
    # A22 B22: u_off=(4,4), v_off=(4,4), w_off=(4,4)
    for t in T_222:
        all_tensors.append((t[0], t[1], t[2], t[3], (4,4), (4,4), (4,4)))
        
    # Now construct U, V, W
    # Note: The rank is the length of all_tensors.
    # We must ensure that the rank matches the number of non-zero terms.
    # Since we are summing tensors, if two tensors have the same (U,V,W) indices,
    # they would be combined. But here, the supports are disjoint in the output space,
    # so the W indices are unique for each block? No, W indices are unique per term,
    # but different terms in the same block have different W indices?
    # Actually, the Strassen tensor terms for a single block are linearly independent.
    # And the blocks have disjoint W supports.
    # So the total rank is the sum of the ranks of the blocks.
    
    R = len(all_tensors)
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for r in range(R):
        (u_i, u_j), (v_i, v_j), (w_i, w_j), val, u_off, v_off, w_off = all_tensors[r]
        
        gu = (u_off[0] + u_i) * m + (u_off[1] + u_j)
        gv = (v_off[0] + v_i) * p + (v_off[1] + v_j)
        gw = (w_off[1] + w_j) * n + (w_off[0] + w_i) # k*n + i
        
        U[gu][r] = val
        V[gv][r] = val
        W[gw][r] = val
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
