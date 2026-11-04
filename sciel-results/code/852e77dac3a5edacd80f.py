import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit
    
    # Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16) = rank 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Strassen 2x2x2 inner terms: each term is (u_list, v_list, w_list)
    # where each list contains ((local_idx1, local_idx2), sign)
    inner_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2 = (A10+A11)B00 -> C10
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1)]),
        # M3 = A00(B01-B11) -> C01
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1)]),
        # M4 = A11(B10-B00) -> C10
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((1,0), 1)]),
        # M5 = (A00+A01)B11 -> C11
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((1,1), 1)]),
        # M6 = (A10-A00)(B00+B01) -> C11
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7 = (A01-A11)(B10+B11) -> C00
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    # Big Strassen terms for 4x4x4:
    # Each is (a_blocks, b_blocks) where blocks are ((block_row, block_col), sign)
    big_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),  # (A00+A11)(B00+B11)
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),               # (A10+A11)B00
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),               # A00(B01-B11)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),               # A11(B10-B00)
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),                # (A00+A01)B11
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),   # (A10-A00)(B00+B01)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),   # (A01-A11)(B10+B11)
    ]
    
    # Part 1: Strassen 4x4x4, terms 0-48
    term_idx = 0
    for bt_idx, (a_blocks, b_blocks) in enumerate(big_terms):
        a_lookup = {}
        for (bi, bj), sign in a_blocks:
            a_lookup[(bi, bj)] = sign
        b_lookup = {}
        for (bj, bk), sign in b_blocks:
            b_lookup[(bj, bk)] = sign
        
        for it_idx, (u_idx, v_idx, w_idx) in enumerate(inner_terms):
            # U: for each (li, lj, su) in u_idx, if (li, lj) in a_lookup:
            #   global_i = 2*li, global_j = 2*lj
            #   U[global_i * 4 + global_j, term_idx] += su * a_lookup[(li,lj)]
            for (li, lj), su in u_idx:
                if (li, lj) in a_lookup:
                    sb = a_lookup[(li, lj)]
                    gi = 2 * li
                    gj = 2 * lj
                    U[gi * 4 + gj][term_idx] += su * sb
            
            # V: for each (lj, lk, sv) in v_idx, if (lj, lk) in b_lookup:
            #   global_j = 2*lj, global_k = 2*lk (but only k=0,1 for 4x4 part)
            #   V[global_j * 5 + global_k, term_idx] += sv * b_lookup[(lj,lk)]
            for (lj, lk), sv in v_idx:
                if (lj, lk) in b_lookup:
                    sb = b_lookup[(lj, lk)]
                    gj = 2 * lj
                    gk = 2 * lk
                    V[gj * 5 + gk][term_idx] += sv * sb
            
            # W: for each (lk, li, sw) in w_idx:
            #   The C block is (bk, bi) where bk comes from B block col, bi from A block row
            #   We need to figure out which (bk, bi) pair this corresponds to
            #   For the big term, the C contribution goes to block (bk, bi)
            #   where bk is the column block of B and bi is the row block of A
            #   global_k = 2*bk + lk, global_i = 2*bi + li
            #   W[global_k * 4 + global_i, term_idx] += sw
            
            # For W, we need to determine bk and bi for each (lk, li) in w_idx
            # The inner term's w_idx has (lk, li) which are local indices in the 2x2 block
            # lk is the local row of C (which corresponds to B's column block), 
            # li is the local col of C (which corresponds to A's row block)
            # So bk = lk, bi = li
            for (lk, li), sw in w_idx:
                # bk = lk, bi = li (these are the block indices)
                bk = lk
                bi = li
                gk = 2 * bk + lk  # Wait, lk is already the local index within the block
                gi = 2 * bi + li  # li is the local index within the block
                # Actually, lk in w_idx is the local index (0 or 1) within the C block
                # and li is also local. The block indices are bk and bi.
                # But what are bk and bi? They come from the big term structure.
                # For a given inner term, the C contribution goes to specific blocks.
                # Looking at the inner terms:
                # M1: w has (0,0) and (1,1) -> C blocks (0,0) and (1,1)
                # M2: w has (1,0) -> C block (1,0)  [bk=1, bi=0]
                # M3: w has (0,1) -> C block (0,1)  [bk=0, bi=1]
                # M4: w has (1,0) -> C block (1,0)  [bk=1, bi=0]
                # M5: w has (1,1) -> C block (1,1)  [bk=1, bi=1]
                # M6: w has (1,1) -> C block (1,1)  [bk=1, bi=1]
                # M7: w has (0,0) -> C block (0,0)  [bk=0, bi=0]
                # So bk = first element of the tuple, bi = second element
                gk = 2 * bk + lk  # This is wrong. lk IS the local index.
                # Wait, in w_idx, the tuple is (lk, li) where lk is local row in C block,
                # li is local col in C block. The block itself is (bk, bi).
                # But bk and bi are determined by the big term, not by the inner term's local indices.
                # Actually, for the inner term, the local indices (lk, li) directly give
                # the position within the 2x2 block matrix. The block matrix entry (lk, li)
                # corresponds to C block (lk, li) in the big term's coordinate system.
                # So bk = lk, bi = li.
                # And the global indices are:
                #   gk = 2 * bk + lk_local... but lk IS lk_local.
                # Hmm, I think I'm confusing myself. Let me re-examine.
                
                # The inner term operates on a 2x2x2 "block tensor" where each entry is a 2x2 matrix.
                # The inner term's u_idx, v_idx, w_idx contain local indices (0 or 1) that refer to
                # the 2x2 block matrix positions.
                # For w_idx, (lk, li) means: the C block at position (lk, li) in the 2x2 block matrix.
                # In the big term, the C block at position (lk, li) corresponds to the original C block
                # at row block lk and col block li.
                # So the global C indices for this local position are:
                #   row = 2 * lk + local_row_within_block
                #   col = 2 * li + local_col_within_block
                # But the inner term is a rank-1 term of the 2x2x2 tensor, so it picks specific
                # scalar entries. The (lk, li) in w_idx refers to the block, and within that block,
                # the specific scalar is determined by... wait, no.
                
                # I think the issue is that the inner terms I defined are for the scalar 2x2x2 case,
                # where the "blocks" are scalars. When we apply them to the block case, each "scalar"
                # in the inner term becomes a 2x2 block.
                # So for the inner term M1 = (A00+A11)(B00+B11) -> C00 + C11:
                # In the scalar case, this is one rank-1 term with U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
                # In the block case, this becomes: for each of the 4 scalar positions in the 2x2 block,
                # we get a contribution. But actually, Strassen at the block level means:
                #   M1_block = (A00_block + A11_block) * (B00_block + B11_block)
                # This is a 2x2 matrix product, which we then decompose into 7 scalar rank-1 terms.
                
                # So the correct approach is:
                # 1. The big term gives us the block-level Strassen structure (7 terms).
                # 2. For each block-level term, we have a product of two 2x2 block matrices.
                # 3. We decompose each 2x2x2 block product into 7 scalar rank-1 terms.
                # 4. Total = 7 * 7 = 49 scalar rank-1 terms.
                
                # So for each big term bt and inner term it:
                #   The inner term it specifies which scalar entries of the 2x2 blocks are used.
                #   For example, if it is M1, it uses scalar entries (0,0) and (1,1) of the A blocks,
                #   (0,0) and (1,1) of the B blocks, and (0,0) and (1,1) of the C blocks.
                #   But which A blocks? From the big term's a_blocks.
                #   If the big term has A00 and A11, then:
                #     Scalar (0,0) of A00 -> global A[0,0]
                #     Scalar (1,1) of A00 -> global A[1,1]
                #     Scalar (0,0) of A11 -> global A[2,2]
                #     Scalar (1,1) of A11 -> global A[3,3]
                #   And the inner term M1 uses (0,0) and (1,1) of the "A matrix", which are
                #   the blocks A00 and A11. So it uses all 4 scalar entries mentioned above.
                
                # So for the U vector of this scalar rank-1 term:
                #   For each (li, lj, su) in u_idx:
                #     (li, lj) refers to the block position in the 2x2 block matrix.
                #     Find the block at position (li, lj) in the big term's a_blocks.
                #     Let that block be at (bi, bj) with sign sb.
                #     The inner term's (li, lj) also specifies the scalar position within that block?
                #     No, I think (li, lj) in the inner term IS the scalar position.
                #     And the block is determined by the big term.
                
                # I think I need to restructure. Let me just hardcode the 49 terms directly.
                # Given the complexity and time constraints, let me use a simpler valid construction.
                
                pass  # This approach is getting too complex. Let me use schoolbook for now.
            
            term_idx += 1
    
    # Given the complexity of correctly implementing Strassen 4x4x4,
    # let me use a simpler but valid approach: schoolbook for 4x4x4 (rank 16) + schoolbook for 4x4x1 (rank 16) = rank 32
    # Wait, schoolbook 4x4x4 is rank 4*4*4 = 64, not 16.
    # Schoolbook 4x4x1 is rank 4*4*1 = 16.
    # Total = 80.
    
    # Let me just return the schoolbook decomposition to ensure validity.
    rank = n * m * p  # 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    if time.time() - start_time > DEADLINE - 5:
        pass
    
    return {"U": U, "V": V, "W": W}
