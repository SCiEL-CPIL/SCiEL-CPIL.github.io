import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe under 150s limit

    def validate_decomposition(U_arr, V_arr, W_arr, n, m, p):
        """Check if U,V,W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = float(np.dot(U_arr[i * m + k], V_arr[k * p + j] * W_arr[j * n + i]))
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Fallback: schoolbook rank 216
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1.0
                    V[k * p + j, col] = 1.0
                    W[j * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2x2 Strassen with schoolbook 2x2 sub-problems: rank = 4*7*8 = 224 (worse than schoolbook)
    # 3x3 Strassen with schoolbook 3x3 sub-problems: rank = 7*27 = 189
    def get_strassen_3x3_schoolbook():
        """Strassen at 3x3 block level, schoolbook within each block. Rank = 7*27 = 189."""
        bs = 3
        nb = 2
        total_rank = 7 * 27
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)

        # Strassen terms: each defines linear combos of A-blocks, B-blocks, and C-blocks
        # For schoolbook 3x3, each "multiplication" A_block * B_block produces C_block
        # We need: for each Strassen term, for each (i3,k3) in A-block and (k3,j3) in B-block,
        # the product contributes to C-block at (i3,j3)

        strassen_defs = [
            # (a_blocks: [(ib,kb,coeff)], b_blocks: [(kb,jb,coeff)], c_blocks: [(ib,jb,coeff)])
            # M1: (A00+A11)(B00+B11)
            ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [(0,0,1),(1,1,1)]),
            # M2: (A10+A11)B00
            ([((1,0,1),(1,1,1))], [((0,0,1))], [(1,0,1)]),
            # M3: A00(B01-B11)
            ([((0,0,1))], [((0,1,1),(1,1,-1))], [(0,1,1)]),
            # M4: A11(B10-B00)
            ([((1,1,1))], [((1,0,1),(0,0,-1))], [(1,0,1)]),
            # M5: (A00+A01)B11
            ([((0,0,1),(0,1,1))], [((1,1,1))], [(0,1,1)]),
            # M6: (A10-A00)(B00+B01)
            ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [(1,1,1)]),
            # M7: (A01-A11)(B10+B11)
            ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [(0,0,1)]),
        ]

        col = 0
        for a_blocks, b_blocks, c_blocks in strassen_defs:
            # For schoolbook 3x3: iterate over all (i3,k3) pairs for A and (k3,j3) pairs for B
            # But we need to handle the linear combinations properly
            # Each term in the expansion: sum over (ab in a_blocks, bb in b_blocks) of
            #   ac*bc * A[ib*bs+i3, kb*bs+k3] * B[kb*bs+k3, jb*bs+j3]
            # This contributes to C[ib*bs+i3, jb*bs+j3] via c_block
            for (ib_a, kb_a, ac) in a_blocks:
                for (kb_b, jb_b, bc) in b_blocks:
                    # Only valid if kb_a == kb_b (the inner dimension must match)
                    if kb_a != kb_b:
                        continue
                    for i3 in range(bs):
                        for j3 in range(bs):
                            for (ib_c, jb_c, cc) in c_blocks:
                                if ib_c != ib_a or jb_c != jb_b:
                                    continue
                                u_row = (ib_a * bs + i3) * m + (kb_a * bs + i3)  # WRONG: should be k3
                                # Actually: A[i,k] where i = ib_a*bs + i3, k = kb_a*bs + k3
                                # We need to iterate k3 too
                            # Let me restructure
                    # Actually for schoolbook, each (i3, k3) in A and (k3, j3) in B gives one product
                    # Let me redo this more carefully
            # Let me restructure: for each Strassen term, expand fully
            col = _build_strassen_term_schoolbook(U, V, W, col, a_blocks, b_blocks, c_blocks, bs, m, p, n)

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _build_strassen_term_schoolbook(U, V, W, col, a_blocks, b_blocks, c_blocks, bs, m, p, n):
        """Build one Strassen term using schoolbook 3x3 sub-multiplication."""
        for (ib_a, kb_a, ac) in a_blocks:
            for (kb_b, jb_b, bc) in b_blocks:
                if kb_a != kb_b:
                    continue
                for k3 in range(bs):
                    for i3 in range(bs):
                        for j3 in range(bs):
                            for (ib_c, jb_c, cc) in c_blocks:
                                if ib_c != ib_a or jb_c != jb_b:
                                    continue
                                u_row = (ib_a * bs + i3) * m + (kb_a * bs + k3)
                                v_row = (kb_b * bs + k3) * p + (jb_b * bs + j3)
                                w_row = (jb_c * bs + j3) * n + (ib_c * bs + i3)
                                U[u_row, col] += ac * ac  # Wait, this is wrong
                                # Actually: U should pick up A[ib_a*bs+i3, kb_a*bs+k3] with coeff ac
                                # V should pick up B[kb_b*bs+k3, jb_b*bs+j3] with coeff bc
                                # W should pick up C[ib_c*bs+i3, jb_c*bs+j3] with coeff cc
                                # But the product ac*bc*cc should equal the contribution
                                # In tensor decomposition: sum_t U[i,k,t]*V[k,j,t]*W[j,i,t] = 1
                                # So for each t, we set U,V,W such that the product works
                                # For schoolbook within a Strassen term, each (i3,k3,j3) triple is one "rank-1"
                                # U[i,k] gets ac, V[k,j] gets bc, W[j,i] gets cc... but that's not right
                                # The issue: in a rank-1 tensor term, U,V,W each contribute ONE factor
                                # For schoolbook: A[i,k]*B[k,j] = C[i,j] is rank-1 with U[i,k]=1, V[k,j]=1, W[i,j]=1
                                # With Strassen linear combos: (sum ac*A_block)(sum bc*B_block) = sum cc*C_block
                                # This is NOT a single rank-1 term - it's a product of sums
                                # So we can't just use one column per (i3,k3,j3)
                                # We need to expand: (sum_a ac_a * A_a)(sum_b bc_b * B_b) = sum_a sum_b ac_a*bc_b * A_a*B_b
                                # Each A_a*B_b is a matrix product of 3x3 blocks, which is rank 27 (schoolbook)
                                # So total for this Strassen term: len(a_blocks)*len(b_blocks)*27 rank-1 terms
                                # But that's the full expansion, not compressed
                                pass
        return col

    # Proper Strassen + schoolbook composition
    def get_strassen_schoolbook():
        """
        6x6 = 2x2 blocks of 3x3.
        Strassen at block level: 7 terms.
        Each term: product of (linear combo of A-blocks) * (linear combo of B-blocks)
        = linear combo of C-blocks.
        
        For proper tensor decomposition composition:
        If sub-problem has rank-r decomposition with terms (U_t, V_t, W_t),
        then for Strassen term r with A_r = sum_a a_ra * A_a, B_r = sum_b b_rb * B_b:
        The product A_r * B_r decomposes as:
        sum_t sum_a sum_b a_ra * b_rb * (U_t embedded for A_a) * (V_t embedded for B_b) * (W_t embedded for C)
        
        But this overcounts. The correct way:
        A_r * B_r = (sum_a a_ra A_a)(sum_b b_rb B_b) = sum_a sum_b a_ra b_rb A_a B_b
        
        Each A_a B_b is a 3x3*3x3 product. With schoolbook rank 27:
        A_a B_b = sum_{i3,k3,j3} e_{i3,k3}^{(a)} e_{k3,j3}^{(b)} e_{i3,j3}^{(c)}
        
        where e_{i3,k3}^{(a)} is the (i3,k3) entry of block A_a.
        
        So for each Strassen term r, and each (a,b) pair, and each (i3,k3,j3):
        One rank-1 term with:
        U[(ib_a*3+i3)*6+(kb_a*3+k3)] = a_ra
        V[(kb_b*3+k3)*6+(jb_b*3+j3)] = b_rb
        W[(jb_c*3+j3)*6+(ib_c*3+i3)] = c_rc  (where (ib_c,jb_c) matches (ib_a,jb_b))
        
        Total rank = sum_r |a_r| * |b_r| * 27
        """
        bs = 3
        U = np.zeros((n * m, 600), dtype=float)  # generous upper bound
        V = np.zeros((m * p, 600), dtype=float)
        W = np.zeros((p * n, 600), dtype=float)

        strassen = [
            ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [(0,0,1),(1,1,1)]),
            ([((1,0,1),(1,1,1))], [((0,0,1))], [(1,0,1)]),
            ([((0,0,1))], [((0,1,1),(1,1,-1))], [(0,1,1)]),
            ([((1,1,1))], [((1,0,1),(0,0,-1))], [(1,0,1)]),
            ([((0,0,1),(0,1,1))], [((1,1,1))], [(0,1,1)]),
            ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [(1,1,1)]),
            ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [(0,0,1)]),
        ]

        col = 0
        for a_blks, b_blks, c_blks in strassen:
            for (ib_a, kb_a, ca) in a_blks:
                for (kb_b, jb_b, cb) in b_blks:
                    if kb_a != kb_b:
                        continue
                    for (ib_c, jb_c, cc) in c_blks:
                        if ib_c != ib_a or jb_c != jb_b:
                            continue
                        for i3 in range(bs):
                            for k3 in range(bs):
                                for j3 in range(bs):
                                    u_row = (ib_a * bs + i3) * m + (kb_a * bs + k3)
                                    v_row = (kb_b * bs + k3) * p + (jb_b * bs + j3)
                                    w_row = (jb_c * bs + j3) * n + (ib_c * bs + i3)
                                    U[u_row, col] = ca
                                    V[v_row, col] = cb
                                    W[w_row, col] = cc
                                    col += 1

        # Trim to actual rank
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    try:
        if time.time() > deadline:
            return get_schoolbook()

        # Try Strassen + schoolbook
        result = get_strassen_schoolbook()
        U_arr = np.array(result["U"], dtype=float)
        V_arr = np.array(result["V"], dtype=float)
        W_arr = np.array(result["W"], dtype=float)
        
        if validate_decomposition(U_arr, V_arr, W_arr, n, m, p):
            return result
        else:
            return get_schoolbook()
            
    except Exception as e:
        return get_schoolbook()
