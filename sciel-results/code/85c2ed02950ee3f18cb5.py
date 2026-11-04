import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit

    # Strassen 2x2x2 scalar decomposition: 7 terms
    # Each term is (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, sign) for A[i,j]
    # v_entries: list of (j, k, sign) for B[j,k]
    # w_entries: list of (k, i, sign) for C[k,i]
    
    def get_strassen_2x2x2():
        # Returns list of 7 tuples: (u_list, v_list, w_list)
        # where each list contains (idx1, idx2, sign)
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        terms.append((
            [(0,0,1), (1,1,1)],   # A: A00 + A11
            [(0,0,1), (1,1,1)],   # B: B00 + B11
            [(0,0,1), (1,1,1)]    # C: C00 + C11
        ))
        
        # M2 = (A10+A11)B00 -> C10 += 1
        terms.append((
            [(1,0,1), (1,1,1)],   # A: A10 + A11
            [(0,0,1)],            # B: B00
            [(0,1,1)]             # C: C10 (k=0,i=1 -> row 1, col 0)
        ))
        
        # M3 = A00(B01-B11) -> C01 += 1
        terms.append((
            [(0,0,1)],            # A: A00
            [(0,1,1), (1,1,-1)],  # B: B01 - B11
            [(1,0,1)]             # C: C01 (k=1,i=0 -> row 0, col 1)
        ))
        
        # M4 = A11(B10-B00) -> C10 += 1
        terms.append((
            [(1,1,1)],            # A: A11
            [(1,0,1), (0,0,-1)],  # B: B10 - B00
            [(0,1,1)]             # C: C10
        ))
        
        # M5 = (A00+A01)B11 -> C11 += 1
        terms.append((
            [(0,0,1), (0,1,1)],   # A: A00 + A01
            [(1,1,1)],            # B: B11
            [(1,1,1)]             # C: C11
        ))
        
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        terms.append((
            [(1,0,1), (0,0,-1)],  # A: A10 - A00
            [(0,0,1), (0,1,1)],   # B: B00 + B01
            [(1,1,1)]             # C: C11
        ))
        
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        terms.append((
            [(0,1,1), (1,1,-1)],  # A: A01 - A11
            [(1,0,1), (1,1,1)],   # B: B10 + B11
            [(0,0,1)]             # C: C00
        ))
        
        return terms
    
    inner_terms = get_strassen_2x2x2()
    
    # Big Strassen terms for 4x4x4 block decomposition
    # Each big term: (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of (bi, bj, sign) - which A blocks are involved
    # b_blocks: list of (bj, bk, sign) - which B blocks are involved
    # c_blocks: list of (bi, bk, sign) - which C blocks receive contributions
    
    big_terms = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)B00 -> C10
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1)]
        },
        # M3: A00(B01-B11) -> C01
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,1,1)]
        },
        # M4: A11(B10-B00) -> C10
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(1,0,1)]
        },
        # M5: (A00+A01)B11 -> C11
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(1,1,1)]
        },
        # M6: (A10-A00)(B00+B01) -> C11
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]
        },
        # M7: (A01-A11)(B10+B11) -> C00
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1)]
        }
    ]
    
    # Total rank: 49 (4x4x4 Strassen) + 16 (4x4x1 schoolbook) = 65
    total_rank = 49 + 16
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 65
    
    term_idx = 0
    
    # Process each big term and each inner Strassen term
    for big in big_terms:
        a_blocks = big['a']  # list of (bi, bj, sign)
        b_blocks = big['b']  # list of (bj, bk, sign)
        c_blocks = big['c']  # list of (bi, bk, sign)
        
        # Create lookups
        a_lookup = {}
        for bi, bj, s in a_blocks:
            a_lookup[(bi, bj)] = s
        b_lookup = {}
        for bj, bk, s in b_blocks:
            b_lookup[(bj, bk)] = s
        c_lookup = {}
        for bi, bk, s in c_blocks:
            c_lookup[(bi, bk)] = s
        
        for inner in inner_terms:
            u_list, v_list, w_list = inner
            
            # For U: for each (li, lj, su) in u_list
            # The local index (li, lj) refers to the block position in the 2x2 block matrix
            # So block (li, lj) must be in a_lookup
            # Global A index: row = 2*li + local_row, col = 2*lj + local_col
            # But wait - the inner Strassen term operates on the 2x2 scalar matrix
            # where each entry is a 2x2 block. The (li, lj) entry of this 2x2 matrix
            # is the block A[li, lj].
            # The inner term picks scalar entries from this 2x2 matrix.
            # But actually, the inner term's indices (li, lj) ARE the block indices.
            # Each block is 2x2, so we need to expand.
            
            # Actually, I think the correct interpretation is:
            # The big term defines a 2x2 matrix A_big where A_big[li, lj] = sum of blocks
            # The inner Strassen term picks entries from A_big, B_big, C_big
            # Each entry of A_big is a 2x2 block, so we need to decompose the 2x2x2
            # scalar multiplication within each block product.
            
            # Let me reconsider. The product A_big * B_big is a 2x2 matrix where
            # each entry is a 2x2 block product. The Strassen decomposition at the
            # scalar level within each 2x2 block product gives 7 terms.
            
            # So for big term t and inner term s:
            # The inner term s specifies which scalar positions in the 2x2 block matrices
            # are used. For example, inner term M1 uses A[0,0], A[1,1], B[0,0], B[1,1], C[0,0], C[1,1].
            # These are the block positions. Each block position (li, lj) corresponds to a 2x2 block.
            # The scalar Strassen term then specifies that we multiply specific scalar entries.
            
            # Wait, I think I'm overcomplicating this. Let me think of it differently.
            
            # The 4x4x4 multiplication tensor can be seen as:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # 
            # Partition into 2x2 blocks:
            # A = [[A00, A01], [A10, A11]] where each block is 2x2
            # B = [[B00, B01], [B10, B11]]
            # C = [[C00, C01], [C10, C11]]
            #
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            #
            # Strassen at block level:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            #
            # C00 = M1 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 + M5 + M6
            #
            # Each M_t is a 2x2 matrix = product of two 2x2 matrices.
            # Each 2x2 matrix product can be decomposed into 7 scalar rank-1 terms.
            #
            # So for each big term t, we have a 2x2 matrix product P = X * Y
            # where X and Y are 2x2 matrices formed from linear combinations of blocks.
            # We decompose P into 7 scalar rank-1 terms using Strassen.
            # Each scalar rank-1 term contributes to the C matrix.
            
            # The key question: what are X and Y for each big term?
            # For big term M1: X = A00+A11 (a 2x2 matrix), Y = B00+B11 (a 2x2 matrix)
            # X[i,j] = (A00[i,j] + A11[i,j]) for i,j in {0,1}
            # Y[i,j] = (B00[i,j] + B11[i,j]) for i,j in {0,1}
            #
            # The 2x2x2 scalar tensor for this big term is:
            # T_big[i,j,k] = X[i,j] * Y[j,k]
            # And this contributes to C blocks according to the big term's c_blocks.
            
            # Now, decompose T_big into 7 scalar rank-1 terms using Strassen:
            # T_big[i,j,k] = sum_{s=1}^{7} U_s[i,j] * V_s[j,k] * W_s[k,i]
            # where U_s, V_s, W_s are defined by the inner Strassen terms.
            
            # But we need to map back to global indices.
            # For a given inner Strassen term s with u_list, v_list, w_list:
            # U_s[i,j] is non-zero only for specific (i,j) pairs.
            # For example, if u_list = [(0,0,1), (1,1,1)], then:
            #   U_s[0,0] = 1, U_s[1,1] = 1, others = 0
            #
            # Now, X[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
            # X[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
            # X[1,0] = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
            # X[1,1] = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
            #
            # So X[i,j] = sum over blocks in a_blocks of sign * A[2*bi+i, 2*bj+j]
            #
            # Therefore, for the rank-1 term:
            # The contribution to U[global_i, global_j, term] comes from:
            # For each (li, lj, su) in u_list:
            #   X[li, lj] contributes su * X[li, lj] to the "U factor"
            #   X[li, lj] = sum_{(bi, bj, sb) in a_blocks} sb * A[2*bi+li, 2*bj+lj]
            #   So for each (bi, bj, sb) in a_blocks:
            #     U[2*bi+li, 2*bj+lj, term] += su * sb
            #
            # Similarly for V and W.
            
            # Let me implement this:
            
            # U contributions
            for li, lj, su in u_list:
                for bi, bj, sb in a_blocks:
                    if bi == li and bj == lj:
                        global_i = 2 * bi + li  # Wait, this is wrong.
                        # Actually, X[li, lj] = sum over blocks where (block_row, block_col) = (li, lj)
                        # No! X is a 2x2 matrix where X[r, c] = sum of block (r, c) entries.
                        # X[li, lj] = sum_{blocks} ... 
                        
                        # Let me re-think. For big term M1:
                        # X = A00 + A11
                        # X[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
                        # X[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
                        # X[1,0] = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
                        # X[1,1] = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
                        
                        # So X[li, lj] = sum over (bi, bj) in a_blocks: 
                        #   if the block (bi, bj) exists, then A[2*bi+li, 2*bj+lj]
                        #   But only if (bi, bj) = (li, lj)? No!
                        #   X[0,0] includes A00[0,0] AND A11[0,0].
                        #   So X[li, lj] = sum over ALL blocks in a_blocks:
                        #     A[2*bi+li, 2*bj+lj]
                        #   Wait, that's not right either. A00[0,0] is A[0,0], A11[0,0] is A[2,2].
                        #   These are different entries!
                        
                        # I see my confusion. The block matrix X is 2x2 where each entry
                        # is a 2x2 matrix. So X[0,0] is the top-left 2x2 block of X,
                        # which equals A00 + A11 (as 2x2 matrices).
                        # So X[0,0][0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
                        # X[0,0][0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
                        # etc.
                        
                        # The inner Strassen term operates on the 2x2 scalar matrix X,
                        # where X[i,j] means the scalar at row i, col j of the 2x2 matrix X.
                        # But X is actually a 2x2 block matrix! Each "entry" X[i,j] is a 2x2 block.
                        
                        # So the inner Strassen term's index (li, lj) refers to which BLOCK
                        # of X is being used. The block X[li, lj] is a 2x2 matrix.
                        # Then we need to further decompose the product of these 2x2 blocks.
                        
                        # This is the recursive structure! The inner Strassen term
                        # decomposes the 2x2x2 block tensor into 7 block rank-1 terms.
                        # Each block rank-1 term is a product of three 2x2 blocks.
                        # Then we decompose each 2x2x2 block product into 7 scalar rank-1 terms.
                        
                        # So for big term t and inner term s:
                        # The inner term s specifies which blocks are multiplied.
                        # For example, if inner term M1 uses X[0,0], X[1,1], Y[0,0], Y[1,1], Z[0,0], Z[1,1]:
                        #   The block rank-1 term is (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1]) * (Z[0,0]+Z[1,1])
                        #   But this isn't a single rank-1 term; it's a sum.
                        
                        # I think the correct approach is:
                        # 1. The big Strassen decomposition gives 7 terms, each involving
                        #    specific A-blocks and B-blocks.
                        # 2. Each such term is a 2x2 matrix product.
                        # 3. We decompose each 2x2 matrix product into 7 scalar rank-1 terms.
                        # 4. For each scalar rank-1 term, we map to global indices.
                        
                        # So for big term t with a_blocks and b_blocks:
                        #   X is a 2x2 matrix where X[i,j] = sum of A[2*bi+i, 2*bj+j] 
                        #     over (bi,bj) in a_blocks
                        #   Y is a 2x2 matrix where Y[i,j] = sum of B[2*bi+i, 2*bj+j]
                        #     over (bi,bj) in b_blocks
                        #
                        # Wait, that's not right. For M1: X = A00 + A11
                        # X[i,j] for i,j in {0,1} is a scalar.
                        # X[0,0] = A00[0,0] + A11[0,0]
                        # X[0,1] = A00[0,1] + A11[0,1]
                        # X[1,0] = A00[1,0] + A11[1,0]
                        # X[1,1] = A00[1,1] + A11[1,1]
                        
                        # So X[i,j] = sum over (bi,bj) in a_blocks: A[2*bi+i, 2*bj+j]
                        # But only if the block (bi,bj) is in a_blocks.
                        
                        # For M1: a_blocks = [(0,0,1), (1,1,1)]
                        # X[0,0] = A[2*0+0, 2*0+0] + A[2*1+0, 2*1+0] = A[0,0] + A[2,2] ✓
                        # X[0,1] = A[2*0+0, 2*0+1] + A[2*1+0, 2*1+1] = A[0,1] + A[2,3] ✓
                        # X[1,0] = A[2*0+1, 2*0+0] + A[2*1+1, 2*1+0] = A[1,0] + A[3,2] ✓
                        # X[1,1] = A[2*0+1, 2*0+1] + A[2*1+1, 2*1+1] = A[1,1] + A[3,3] ✓
                        
                        # Great! So X[i,j] = sum over (bi,bj,sb) in a_blocks: sb * A[2*bi+i, 2*bj+j]
                        
                        # Now, the inner Strassen term s has u_list, v_list, w_list.
                        # The rank-1 term is:
                        #   U_factor[i,j] = sum over (li,lj,su) in u_list: su * X[li,lj]
                        #   V_factor[j,k] = sum over (lj,lk,sv) in v_list: sv * Y[lj,lk]
                        #   W_factor[k,i] = sum over (lk,li,sw) in w_list: sw * Z[lk,li]
                        #
                        # But this is a sum of products, not a single rank-1 term!
                        # The Strassen decomposition at the scalar level says:
                        #   X[i,j] * Y[j,k] = sum_s U_s[i,j] * V_s[j,k] * W_s[k,i]
                        #
                        # Where U_s, V_s, W_s are sparse vectors.
                        #
                        # So for a specific inner term s:
                        #   U_s[i,j] is non-zero only for specific (i,j)
                        #   For each (li,lj,su) in u_list, U_s[li,lj] = su
                        #
                        # But X[li,lj] = sum over blocks, so:
                        #   The contribution to the global U matrix is:
                        #   For each (li,lj,su) in u_list:
                        #     For each (bi,bj,sb) in a_blocks:
                        #       global_i = 2*bi + li
                        #       global_j = 2*bj + lj
                        #       U[global_i, global_j, term] += su * sb
                        #
                        # Similarly for V and W.
                        #
                        # And W: for each (lk,li,sw) in w_list:
                        #   Z[lk,li] = sum over (bi,bk,sb) in c_blocks: sb * C[2*bi+lk, 2*bk+li]
                        #   Wait, C[i,k] where i is row, k is col.
                        #   C block (bi, bk) covers C[2*bi..2*bi+1, 2*bk..2*bk+1]
                        #   Z[lk,li] corresponds to C block... 
                        #   In the inner Strassen term, w_list has (lk, li, sw) meaning C[lk, li]
                        #   But C is indexed as C[row, col] = C[k, i] in the tensor notation.
                        #   So C[lk, li] means row=lk, col=li.
                        #   This corresponds to block (li, lk)? No.
                        #   C[row, col] where row is from B's row index, col is from A's col index.
                        #   In block terms: C[bi, bk] where bi is A's block row, bk is B's block col.
                        #   C[lk, li] with lk, li in {0,1} corresponds to which block?
                        #   The full C is 4x4. C[r, c] where r in 0..3, c in 0..3.
                        #   C[lk, li] is a specific scalar entry.
                        #   It belongs to block (bi, bk) where bi = lk // 2 = 0 (since lk in {0,1}),
                        #   and bk = li // 2 = 0.
                        #   Wait, lk and li are 0 or 1, so they're local indices within a block.
                        #   The block is determined by which c_block entry (bi, bk) we're in.
                        
                        # For W:
                        #   For each (lk, li, sw) in w_list:
                        #     For each (bi, bk, sb) in c_blocks:
                        #       global_k = 2*bk + lk  # C row index
                        #       global_i = 2*bi + li  # C col index
                        #       W[global_k * n + global_i, term] += sw * sb
            
            # Now I understand. Let me implement:
            
            # U contributions
            for li, lj, su in u_list:
                for bi, bj, sb in a_blocks:
                    global_i = 2 * bi + li
                    global_j = 2 * bj + lj
                    U[global_i * m + global_j][term_idx] += su * sb
            
            # V contributions
            for lj, lk, sv in v_list:
                for bi, bj, sb in b_blocks:
                    # B block (bi, bj) covers B[2*bi+i, 2*bj+j]
                    # v_list has (lj, lk, sv) meaning B[lj, lk] in the 2x2 block matrix
                    # B[li, lj] of block matrix = sum over blocks: B[2*bi+li, 2*bj+lj]
                    global_j = 2 * bi + lj
                    global_k = 2 * bj + lk
                    V[global_j * p + global_k][term_idx] += sv * sb
            
            # W contributions
            for lk, li, sw in w_list:
                for bi, bk, sb in c_blocks:
                    # C[lk, li] where lk is row, li is col (in local 2x2 coords)
                    # C block (bi, bk) covers C[2*bi+i, 2*bk+j]
                    # C[lk, li] in block (bi, bk) -> C[2*bi+lk, 2*bk+li]
                    global_k = 2 * bi + lk  # C row
                    global_i = 2 * bk + li  # C col
                    W[global_k * n + global_i][term_idx] += sw * sb
            
            term_idx += 1
    
    # Schoolbook for 4x4x1 (column 4 of B)
    for i in range(n):
        for j in range(m):
            col_idx = 49 + i * m + j
            U[i * m + j][col_idx] = 1
            V[j * p + 4][col_idx] = 1
            W[4 * n + i][col_idx] = 1
    
    if time.time() - start_time > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
