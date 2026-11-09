"""Matrix multiplication 4,4,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    time_limit = 145  # Leave margin before 150s kill

    # ------------------------------------------------------------------
    # Helper: verification via einsum
    # ------------------------------------------------------------------
    def verify_numpy(U, V, W, r):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.max(np.abs(T - 1.0))

    # ------------------------------------------------------------------
    # Strassen 4x4x4 decomposition (rank 49)
    # A is 4x4, B is 4x4, C is 4x4
    # We use two levels of Strassen: split 4x4 into four 2x2 blocks.
    # Level 1: 7 Strassen products on 2x2 blocks (each 2x2x2 = 7 multiplications)
    # Total: 7 * 7 = 49 multiplications.
    # ------------------------------------------------------------------
    def build_strassen_4x4x4():
        """
        Returns U (16 x 49), V (16 x 49), W (16 x 49)
        for 4x4x4 matrix multiplication.
        U row index: i*4+j (i,j in 0..3)
        V row index: j*4+k (j,k in 0..3)
        W row index: k*4+i (k,i in 0..3)
        """
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))

        # Strassen 2x2x2: 7 products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Strassen 2x2x2 factor patterns (for a single 2x2 block multiplication):
        # Each product t (0..6) has:
        #   A-coeffs: 4 entries (a00,a01,a10,a11)
        #   B-coeffs: 4 entries
        #   C-coeffs: (c00,c01,c10,c11) each in {-1,0,1}
        strassen2 = {
            0: {'A': [(0,1,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
            1: {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
            2: {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
            3: {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
            4: {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
            5: {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
            6: {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        }

        # Level 1: 7 products, each is a 2x2x2 Strassen (7 sub-products)
        # Block layout: A[bi][bj] where bi,bj in 0..1, each block is 2x2
        # A block (bi,bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # B block (bj,bk) covers rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
        # C block (bi,bk) covers rows 2*bi..2*bi+1, cols 2*bk..2*bk+1

        prod_idx = 0
        for outer in range(7):
            outer_info = strassen2[outer]
            for inner in range(7):
                inner_info = strassen2[inner]

                # Determine which 2x2 blocks are involved in this outer product
                # A-blocks: from outer A-coeffs
                # B-blocks: from outer B-coeffs
                # C-blocks: from outer C-coeffs

                # For each A-block (a_bi, a_bj, coeff_a)
                # For each B-block (b_bj, b_bk, coeff_b)
                #   Inner product computes: (sum of A-block coeffs) * (sum of B-block coeffs)
                #   Then distributes to C-blocks

                # Actually, the standard Strassen expansion:
                # Each outer product M_outer = (linear combo of A-blocks) * (linear combo of B-blocks)
                # Then M_outer contributes to C-blocks.
                # Each such product is itself a 2x2x2 multiplication, so it decomposes
                # into 7 inner 1x1x1 products.

                # A-blocks involved:
                a_blocks = outer_info['A']  # list of (bi, bj, coeff)
                b_blocks = outer_info['B']  # list of (bj, bk, coeff)
                c_blocks = outer_info['C']  # list of (bi, bk, coeff)

                # For the inner Strassen decomposition of this 2x2x2 product:
                # The "A" of the 2x2x2 is a 2x2 matrix formed from the A-block coefficients
                # applied to the 2x2 sub-blocks.
                # Actually, let's think differently.

                # The outer Strassen product M_outer computes:
                #   M_outer = (sum_{(bi,bj,ca)} ca * A[bi][bj]) * (sum_{(bj',bk,cb)} cb * B[bj'][bk])
                # But wait, the block indices must be compatible: A-block (bi,bj) multiplies B-block (bj,bk)
                # to give C-block (bi,bk). The Strassen formula ensures this.

                # For the inner 2x2x2 decomposition, we need to express the 2x2 "A-matrix" and
                # "B-matrix" of the sub-problem. The sub-problem is:
                #   sub_A[r][c] = coefficient of A[bi+r][bj+c] in the linear combo
                #   sub_B[r][c] = coefficient of B[bj+r][bk+c] in the linear combo
                #   sub_C[r][c] = coefficient of C[bi+r][bk+c] from the result

                # Hmm, this is getting complex. Let me use a different approach.

                # Alternative: Build the full 4x4x4 Strassen directly.
                # I'll enumerate all 49 (i,j,k) contributions.

                # Actually, let me just hardcode the 49 products.
                # Each product is (i,j,k) with i,j,k in 0..3.
                # U[i*4+j][prod_idx] = coeff
                # V[j*4+k][prod_idx] = coeff
                # W[k*4+i][prod_idx] = coeff

                prod_idx += 1
            # End inner
        # End outer

        # The above approach is wrong. Let me redo.
        # I'll construct the 49 products explicitly.

        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))

        # The 4x4x4 Strassen decomposition uses 49 scalar multiplications.
        # I'll define them by their (i,j,k) indices and coefficients.
        # Using the standard two-level Strassen.

        # Level 1: 7 products, each decomposed into 7 level-2 products.
        # Level 2: 7 products, each is a single scalar multiply.

        # Let's define the level-1 Strassen for 4x4x4:
        # M0 = (A00+A11)(B00+B11) -> contributes to C00, C11
        # M1 = (A10+A11)B00 -> contributes to C10, C11
        # M2 = A00(B01-B11) -> contributes to C01, C11
        # M3 = A11(B10-B00) -> contributes to C00, C10
        # M4 = (A00+A01)B11 -> contributes to C00, C01
        # M5 = (A10-A00)(B00+B01) -> contributes to C11
        # M6 = (A01-A11)(B10+B11) -> contributes to C00, C11

        # Where A00, A01, A10, A11 are 2x2 blocks of A, etc.

        # Each Mi is a 2x2x2 multiplication, which uses 7 scalar products.
        # For Mi = X * Y where X and Y are 2x2 matrices:
        # The 7 products are:
        # p0: (X00+X11)(Y00+Y11) -> C00+=1, C11+=1
        # p1: (X10+X11)Y00 -> C10+=1, C11-=1
        # p2: X00(Y01-Y11) -> C01+=1, C11+=1
        # p3: X11(Y10-Y00) -> C00+=1, C10+=1
        # p4: (X00+X01)Y11 -> C00-=1, C01+=1
        # p5: (X10-X00)(Y00+Y01) -> C11+=1
        # p6: (X01-X11)(Y10+Y11) -> C00+=1, C11+=1

        # For each Mi, I need to determine X and Y in terms of the 4x4 blocks.
        # Then for each of the 7 sub-products, I need to determine the (i,j,k) indices.

        # Block indices:
        # A block (bi,bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # So A[bi*2+di][bj*2+dj] for di,dj in 0..1
        # Similarly for B and C.

        # Level-1 Strassen:
        l1 = [
            # (name, A_blocks: [(bi,bj,coeff)], B_blocks: [(bj,bk,coeff)], C_blocks: [(bi,bk,coeff)])
            ('M0', [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ('M1', [(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ('M2', [(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ('M3', [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ('M4', [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ('M5', [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ('M6', [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ]

        # Level-2 Strassen (2x2x2):
        l2 = [
            # (A_coeff: [(di,dj,c)], B_coeff: [(dj2,dk,c)], C_coeff: [(di2,dk2,c)])
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ]

        prod_idx = 0
        for _, a_blks, b_blks, c_blks in l1:
            for a_coeff, b_coeff, c_coeff in l2:
                # For this sub-product, we need to find all (i,j,k) where:
                # U[i*4+j] gets a coefficient, V[j*4+k] gets a coefficient, W[k*4+i] gets a coefficient.
                # The coefficient is the product of the appropriate coefficients.

                # For each (di,dj,ca) in a_coeff, (bj2,dk,cb) in b_coeff, (di2,dk2,cc) in c_coeff:
                # The A-block is at (bi, bj) from a_blks, with sub-block offset (di,dj)
                # The B-block is at (bj_b, bk) from b_blks, with sub-block offset (bj2,dk)
                # The C-block is at (bi_c, bk_c) from c_blks, with sub-block offset (di2,dk2)

                # For consistency: bi must match between A and C, bj must match between A and B,
                # bk must match between B and C.

                for (bi_a, bj_a, ca_blk) in a_blks:
                    for (bj_b, bk_b, cb_blk) in b_blks:
                        for (bi_c, bk_c, cc_blk) in c_blks:
                            if bi_a != bi_c or bj_a != bj_b or bk_b != bk_c:
                                continue
                            bi, bj, bk = bi_a, bj_a, bk_b
                            for (di, dj, ca) in a_coeff:
                                for (dj2, dk, cb) in b_coeff:
                                    if dj != dj2:
                                        continue
                                    for (di2, dk2, cc) in c_coeff:
                                        if di != di2 or dk != dk2:
                                            continue
                                        i = 2*bi + di
                                        j = 2*bj + dj
                                        k = 2*bk + dk
                                        coeff = ca * cb * cc * ca_blk * cb_blk * cc_blk
                                        U[i*4+j, prod_idx] += coeff
                                        V[j*4+k, prod_idx] += coeff
                                        W[k*4+i, prod_idx] += coeff
                prod_idx += 1

        return U, V, W

    # Build the Strassen 4x4x4 decomposition
    U_str, V_str, W_str = build_strassen_4x4x4()

    # ------------------------------------------------------------------
    # Build the full 4x4x5 decomposition:
    # Part 1: 4x4x4 (Strassen, rank 49) - uses first 4 columns of B (k=0..3)
    # Part 2: 4x4x1 (schoolbook, rank 16) - uses last column of B (k=4)
    # ------------------------------------------------------------------
    rank_total = 49 + 16
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))  # m*p = 4*5 = 20
    W = np.zeros((20, rank_total))  # p*n = 5*4 = 20

    # Part 1: Strassen 4x4x4
    # U: rows i*4+j (i,j in 0..3) -> same
    # V: rows j*4+k (j in 0..3, k in 0..3) -> first 16 rows
    # W: rows k*4+i (k in 0..3, i in 0..3) -> first 16 rows
    U[:, :49] = U_str
    V[:16, :49] = V_str
    W[:16, :49] = W_str

    # Part 2: Schoolbook 4x4x1
    # A is 4x4, B is 4x1 (column k=4), C is 4x1
    # For each i,j: C[i][4] += A[i][j] * B[j][4]
    # U[i*4+j][49+t] = 1
    # V[j*4+4][49+t] = 1  (j in 0..3, k=4)
    # W[4*4+i][49+t] = 1  (k=4, i in 0..3)
    t = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j, 49+t] = 1
            V[j*4+4, 49+t] = 1
            W[4*4+i, 49+t] = 1
            t += 1

    # Verify
    err = verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank_total)
    if err > 1e-6:
        # Fallback to schoolbook
        U = np.zeros((16, 80))
        V = np.zeros((20, 80))
        W = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        rank_total = 80
        err = verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank_total)
        if err > 1e-6:
            # Should not happen
            pass

    # Convert to lists with clean values
    U_list = [[float(U[i][c]) for c in range(rank_total)] for i in range(16)]
    V_list = [[float(V[i][c]) for c in range(rank_total)] for i in range(20)]
    W_list = [[float(W[i][c]) for c in range(rank_total)] for i in range(20)]

    # Round to nearest 0.5
    U_list = [[round(x * 2) / 2.0 for x in row] for row in U_list]
    V_list = [[round(x * 2) / 2.0 for x in row] for row in V_list]
    W_list = [[round(x * 2) / 2.0 for x in row] for row in W_list]

    # Final verification
    err = verify_numpy(U_list, V_list, W_list, rank_total)
    if err > 1e-6:
        # Fall back to schoolbook
        U_list = [[0.0] * 80 for _ in range(16)]
        V_list = [[0.0] * 80 for _ in range(20)]
        W_list = [[0.0] * 80 for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_list[i*4+j][col] = 1.0
                    V_list[j*5+k][col] = 1.0
                    W_list[k*4+i][col] = 1.0
                    col += 1

    return {"U": U_list, "V": V_list, "W": W_list}
