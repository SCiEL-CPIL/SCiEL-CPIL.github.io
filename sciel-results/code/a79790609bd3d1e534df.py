import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-93 tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
      - Left part: 4x4 * 4x4 (B columns 0..3) using recursive Strassen, rank 49.
      - Right part: 4x4 * 4x3 (B columns 4..6) using optimized block decomposition, rank 44.
      - Total rank = 93.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 110

    # ------------------------------------------------------------------
    # Part 1: Recursive Strassen for 4x4 * 4x4 (Rank 49)
    # ------------------------------------------------------------------
    # Strassen rank-7 decomposition of 2x2x2 matmul tensor.
    # s2x2 list of (xvec, yvec, zvec) length 4 tuples (00,01,10,11)
    s2x2 = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,-1)),
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),
        ((0,0,0,1), (0,0,1,-1), (-1,0,1,0)),
        ((1,1,0,0), (0,0,0,1), (0,-1,0,1)),
        ((-1,0,1,0), (1,1,0,0), (1,0,0,0)),
        ((0,1,0,-1), (0,0,1,1), (-1,0,0,0)),
    ]

    # Block definitions for 4x4x4 recursive Strassen
    # x_blocks, y_blocks, out_recon
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    block_defs = [
        # M1 = (A00+A11)*(B00+B11)
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)],
         [((0,0),1), ((1,1),-1)]),
        # M2 = (A10+A11)*B00
        ([((1,0),1), ((1,1),1)], [((0,0),1)],
         [((1,0),1), ((1,1),-1)]),
        # M3 = A00*(B01-B11)
        ([((0,0),1)], [((0,1),1), ((1,1),-1)],
         [((0,1),1), ((1,1),1)]),
        # M4 = A11*(B10-B00)
        ([((1,1),1)], [((1,0),1), ((0,0),-1)],
         [((0,0),-1), ((1,0),1)]),
        # M5 = (A00+A01)*B11
        ([((0,0),1), ((0,1),1)], [((1,1),1)],
         [((0,1),-1), ((1,1),1)]),
        # M6 = (A10-A00)*(B00+B01)
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)],
         [((0,0),1)]),
        # M7 = (A01-A11)*(B10+B11)
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)],
         [((0,0),-1)]),
    ]

    out_recon = {
        (0,0): [(0,1), (3,-1), (5,1), (6,-1)],
        (1,0): [(1,1), (3,1)],
        (0,1): [(2,1), (4,1)],
        (1,1): [(0,1), (2,-1), (1,1), (4,-1)],
    }

    terms_444 = []
    for k, (x_blocks, y_blocks, _out) in enumerate(block_defs):
        # Build X and Y local 2x2 matrices mapping to global A/B entries
        X = {}
        for (br, bc), coeff in x_blocks:
            for lr in range(2):
                for lc in range(2):
                    ar, ac = br*2+lr, bc*2+lc
                    if (lr,lc) not in X: X[(lr,lc)] = {}
                    X[(lr,lc)][(ar,ac)] = X[(lr,lc)].get((ar,ac), 0) + coeff

        Y = {}
        for (br, bc), coeff in y_blocks:
            for lr in range(2):
                for lc in range(2):
                    br_, bc_ = br*2+lr, bc*2+lc
                    if (lr,lc) not in Y: Y[(lr,lc)] = {}
                    Y[(lr,lc)][(br_,bc_)] = Y[(lr,lc)].get((br_,bc_), 0) + coeff

        for t in range(7):
            xvec, yvec, zvec = s2x2[t]
            a_vec = {}
            for lr in range(2):
                for lc in range(2):
                    xc = xvec[lr*2+lc]
                    if xc == 0: continue
                    if (lr,lc) not in X: continue
                    for (ar,ac), ac2 in X[(lr,lc)].items():
                        val = xc * ac2
                        if val != 0:
                            a_vec[(ar,ac)] = a_vec.get((ar,ac), 0) + val

            b_vec = {}
            for lr in range(2):
                for lc in range(2):
                    yc = yvec[lr*2+lc]
                    if yc == 0: continue
                    if (lr,lc) not in Y: continue
                    for (br_,bc_), bc2 in Y[(lr,lc)].items():
                        val = yc * bc2
                        if val != 0:
                            b_vec[(br_,bc_)] = b_vec.get((br_,bc_), 0) + val

            c_vec = {}
            for (ob_r, ob_c), sign_list in out_recon.items():
                for (mk, sign) in sign_list:
                    if mk != k: continue
                    for lr in range(2):
                        for lc in range(2):
                            zc = zvec[lr*2+lc]
                            if zc == 0: continue
                            cr, cc = ob_r*2+lr, ob_c*2+lc
                            val = sign * zc
                            if val != 0:
                                c_vec[(cr,cc)] = c_vec.get((cr,cc), 0) + val

            if a_vec and b_vec and c_vec:
                terms_444.append((a_vec, b_vec, c_vec))

    # ------------------------------------------------------------------
    # Part 2: Optimized 4x4 * 4x3 (Rank 44)
    # ------------------------------------------------------------------
    # Split A into 4 blocks A00, A01, A10, A11 (2x2)
    # Split B into 2 blocks Btop (2x3), Bbot (2x3)
    # Ctop (2x3) = A00*Btop + A01*Bbot
    # Cbot (2x3) = A10*Btop + A11*Bbot
    
    # Rank 11 decomposition for 2x2 * 2x3
    # Uses Strassen (rank 7) for first 2 cols, Schoolbook (rank 4) for 3rd col
    def gen_m223_terms(a_offset_r, a_offset_c, b_offset_r, c_offset_r, c_offset_c):
        """
        Generate 11 terms for M[2,2,3].
        A is 2x2 at (a_offset_r, a_offset_c)
        B is 2x3 at (b_offset_r, 0..2) -- Note: B cols are global 0..2 for this subproblem?
        Actually, we are decomposing the 4x4x3 part which corresponds to B cols 4..6.
        So in the global B, these are cols 4,5,6.
        In the local 2x3 B, they are cols 0,1,2.
        Global B col = 4 + local_col.
        Global C col = 4 + local_col.
        """
        terms = []
        
        # Strassen part for cols 0,1
        # X = A (2x2), Y = B[:, 0:2] (2x2)
        # Z = C[:, 0:2] (2x2)
        for t in range(7):
            xvec, yvec, zvec = s2x2[t]
            a_vec = {}
            b_vec = {}
            c_vec = {}
            
            # Map X (2x2) to A global
            for lr in range(2):
                for lc in range(2):
                    xc = xvec[lr*2+lc]
                    if xc == 0: continue
                    ar = a_offset_r + lr
                    ac = a_offset_c + lc
                    a_vec[(ar, ac)] = a_vec.get((ar, ac), 0) + xc
            
            # Map Y (2x2) to B global (cols 0,1 -> global 4,5)
            for lr in range(2):
                for lc in range(2):
                    yc = yvec[lr*2+lc]
                    if yc == 0: continue
                    br = b_offset_r + lr
                    bc = 4 + lc  # Local col 0,1 -> Global 4,5
                    b_vec[(br, bc)] = b_vec.get((br, bc), 0) + yc
            
            # Map Z (2x2) to C global (cols 0,1 -> global 4,5)
            for lr in range(2):
                for lc in range(2):
                    zc = zvec[lr*2+lc]
                    if zc == 0: continue
                    cr = c_offset_r + lr
                    cc = 4 + lc  # Local col 0,1 -> Global 4,5
                    c_vec[(cr, cc)] = c_vec.get((cr, cc), 0) + zc
            
            if a_vec and b_vec and c_vec:
                terms.append((a_vec, b_vec, c_vec))
        
        # Schoolbook part for col 2 (global 6)
        # Z[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
        # Z[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
        
        # Term 8: A[0,0] * B[0,2] -> C[0,2]
        terms.append((
            {(a_offset_r+0, a_offset_c+0): 1},
            {(b_offset_r+0, 4+2): 1},
            {(c_offset_r+0, 4+2): 1}
        ))
        
        # Term 9: A[0,1] * B[1,2] -> C[0,2]
        terms.append((
            {(a_offset_r+0, a_offset_c+1): 1},
            {(b_offset_r+1, 4+2): 1},
            {(c_offset_r+0, 4+2): 1}
        ))
        
        # Term 10: A[1,0] * B[0,2] -> C[1,2]
        terms.append((
            {(a_offset_r+1, a_offset_c+0): 1},
            {(b_offset_r+0, 4+2): 1},
            {(c_offset_r+1, 4+2): 1}
        ))
        
        # Term 11: A[1,1] * B[1,2] -> C[1,2]
        terms.append((
            {(a_offset_r+1, a_offset_c+1): 1},
            {(b_offset_r+1, 4+2): 1},
            {(c_offset_r+1, 4+2): 1}
        ))
        
        return terms

    terms_443 = []
    # Ctop = A00*Btop + A01*Bbot
    # A00 at (0,0), Btop at (0,0) (rows 0,1), Ctop at (0,0) (rows 0,1)
    terms_443.extend(gen_m223_terms(0, 0, 0, 0, 0))
    # A01 at (0,1), Bbot at (1,0) (rows 2,3), Ctop at (0,0) (rows 0,1)
    terms_443.extend(gen_m223_terms(0, 1, 2, 0, 0))
    
    # Cbot = A10*Btop + A11*Bbot
    # A10 at (1,0), Btop at (0,0) (rows 0,1), Cbot at (1,0) (rows 2,3)
    terms_443.extend(gen_m223_terms(1, 0, 0, 2, 0))
    # A11 at (1,1), Bbot at (1,0) (rows 2,3), Cbot at (1,0) (rows 2,3)
    terms_443.extend(gen_m223_terms(1, 1, 2, 2, 0))

    all_terms = terms_444 + terms_443
    rank = len(all_terms)

    # Build U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r, (a_vec, b_vec, c_vec) in enumerate(all_terms):
        for (ar, ac), val in a_vec.items():
            U[ar * m + ac][r] = val
        for (br, bc), val in b_vec.items():
            V[br * p + bc][r] = val
        for (cr, cc), val in c_vec.items():
            W[cc * n + cr][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
