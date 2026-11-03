"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 4x4 matrix multiplication
    # This achieves rank 49 (vs 64 for schoolbook)
    #
    # The 4x4 matrices are partitioned into four 2x2 blocks:
    #   A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    #   C = [C11 C12; C21 C22]
    #
    # Strassen's 7 products for 2x2:
    #   P1 = (A11+A22)(B11+B22)
    #   P2 = (A21+A22)B11
    #   P3 = A11(B21-B22)
    #   P4 = A22(B12-B11)
    #   P5 = (A11+A12)B22
    #   P6 = (A21-A11)(B11+B12)
    #   P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # For each of the 4 block positions in C, we apply Strassen,
    # giving 4 * 7 = 28 products at the top level.
    # But we can be smarter: we apply the full Strassen decomposition
    # treating the 4x4 as a single entity using the known optimal
    # or near-optimal decompositions.
    #
    # Actually, the standard approach for 4x4 is to use Strassen
    # recursively: split 4x4 into 2x2 blocks, apply Strassen's 7
    # multiplications where each "multiplication" is of 2x2 matrices.
    # Each 2x2 multiplication can itself be done with Strassen (7 scalar
    # mults) or schoolbook (4 scalar mults).
    #
    # If we use Strassen for the outer level (7 block mults) and
    # schoolbook for each inner 2x2 mult (4 scalar mults each):
    #   rank = 7 * 4 = 28
    #
    # If we use Strassen at both levels:
    #   rank = 7 * 7 = 49
    #
    # The best known for 4x4 is actually 49 with pure Strassen,
    # but there are better decompositions. Let me implement the
    # 49-rank Strassen-at-both-levels decomposition.
    #
    # Indexing convention:
    #   U has shape (n*m, rank) = (16, rank), indexed by [i*m+j, c]
    #   V has shape (m*p, rank) = (16, rank), indexed by [j*p+k, c]  
    #   W has shape (p*n, rank) = (16, rank), indexed by [k*n+i, c]
    #
    # The tensor decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = delta[i][k] (i.e., C[i,k] = sum_j A[i,j]*B[j,k])
    #
    # So T[i,j,k] = 1 if i==k, else 0.
    #
    # Let me implement the 49-rank decomposition using Strassen twice.
    #
    # Level 1: 4x4 = 2x2 blocks of 2x2
    #   Block positions: (r, s) where r,s in {0,1} for row/col block
    #   A[r*2:(r+1)*2, s*2:(s+1)*2] = A_block[r,s]
    #
    # Strassen level 1 (4x4 level):
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Each M_i is a 2x2 matrix product. Each 2x2 product uses Strassen
    # with 7 scalar multiplications.
    #
    # For M1 = (A00+A11)(B00+B11):
    #   Let X = A00+A11, Y = B00+B11
    #   X and Y are 2x2 matrices. Apply Strassen to X*Y.
    #
    # This gives 7*7 = 49 total scalar products.
    #
    # Let me build the U, V, W matrices.
    #
    # First, let me define the linear forms for each Strassen term.
    #
    # For a 2x2 matrix A = [[a00, a01], [a10, a11]], the Strassen terms are:
    #   S1 = (a00+a11)(b00+b11)  -> linear form on A: a00+a11, on B: b00+b11
    #   S2 = (a10+a11)(b00)       -> on A: a10+a11, on B: b00
    #   S3 = (a00)(b10-b11)       -> on A: a00, on B: b10-b11
    #   S4 = (a11)(b01-b00)       -> on A: a11, on B: b01-b00
    #   S5 = (a00+a01)(b11)       -> on A: a00+a01, on B: b11
    #   S6 = (a10-a00)(b00+b01)   -> on A: a10-a00, on B: b00+b01
    #   S7 = (a01-a11)(b10+b11)   -> on A: a01-a11, on B: b10+b11
    #
    # Output:
    #   c00 = S1 + S4 - S5 + S7
    #   c01 = S3 + S5
    #   c10 = S2 + S4
    #   c11 = S1 - S2 + S3 + S6

    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # i in 0..3, j in 0..3, k in 0..3
    # Block indices: bi = i//2, bj = j//2, bk = k//2 (block row/col)
    # Local indices: li = i%2, lj = j%2, lk = k%2
    
    # Strassen level 1: 7 products, each is a 2x2 * 2x2
    # For product index t in 0..6, we have:
    #   The 2x2 A-submatrix is a linear combination of A blocks
    #   The 2x2 B-submatrix is a linear combination of B blocks
    #   Each contributes to specific C blocks with specific coefficients
    
    # Level 1 Strassen terms (for 4x4):
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)(B00)
    # t=2: M3 = (A00)(B10-B11)
    # t=3: M4 = (A11)(B01-B00)
    # t=4: M5 = (A00+A01)(B11)
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    # For each level-1 term t, the A-side is a sum of A blocks:
    #   A_blocks[t] = list of (block_row, block_col, coeff)
    a_blocks = [
        [(0,0,1), (1,1,1)],   # t=0: A00+A11
        [(1,0,1), (1,1,1)],   # t=1: A10+A11
        [(0,0,1)],             # t=2: A00
        [(1,1,1)],             # t=3: A11
        [(0,0,1), (0,1,1)],   # t=4: A00+A01
        [(1,0,1), (0,0,-1)],  # t=5: A10-A00
        [(0,1,1), (1,1,-1)],  # t=6: A01-A11
    ]
    
    # B-side:
    b_blocks = [
        [(0,0,1), (1,1,1)],   # t=0: B00+B11
        [(0,0,1)],             # t=1: B00
        [(1,0,1), (1,1,-1)],  # t=2: B10-B11
        [(0,1,1), (0,0,-1)],  # t=3: B01-B00
        [(1,1,1)],             # t=4: B11
        [(0,0,1), (0,1,1)],   # t=5: B00+B01
        [(1,0,1), (1,1,1)],   # t=6: B10+B11
    ]
    
    # Level 1 output: C blocks get contributions from M_t
    # C00 = M1 + M4 - M5 + M7  -> t=0(+1), t=3(+1), t=4(-1), t=6(+1)
    # C01 = M3 + M5             -> t=2(+1), t=4(+1)
    # C10 = M2 + M4             -> t=1(+1), t=3(+1)
    # C11 = M1 - M2 + M3 + M6  -> t=0(+1), t=1(-1), t=2(+1), t=5(+1)
    c_contribs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00
        [(2, 1), (4, 1)],                     # C01
        [(1, 1), (3, 1)],                     # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
    ]
    # c_contribs[c_block] = list of (t, coeff) where c_block in 0..3 = (0,0),(0,1),(1,0),(1,1)
    
    # Now, for each level-1 term t, and each level-2 Strassen term s (0..6),
    # we have a scalar product.
    #
    # The A-side for (t,s): it's the s-th Strassen linear form applied to the
    # 2x2 matrix formed by summing A blocks per a_blocks[t].
    #
    # Let me think about this more carefully.
    #
    # For level-1 term t, the A-matrix X is a 2x2 matrix where:
    #   X = sum over (br, bc, coeff) in a_blocks[t] of A_block[br, bc]
    #
    # X is 2x2, so X = [[X00, X01], [X10, X11]]
    # where X[li, lj] = sum over (br, bc, coeff) in a_blocks[t] of A[br*2+li, bc*2+lj] * coeff
    #
    # Similarly for Y from b_blocks[t].
    #
    # Then the s-th Strassen term of X*Y is:
    #   s=0: (X00+X11)(Y00+Y11)
    #   s=1: (X10+X11)(Y00)
    #   s=2: (X00)(Y10-Y11)
    #   s=3: (X11)(Y01-Y00)
    #   s=4: (X00+X01)(Y11)
    #   s=5: (X10-X00)(Y00+Y01)
    #   s=6: (X01-X11)(Y10+Y11)
    #
    # Each is a product of a linear form on X and a linear form on Y.
    #
    # The linear form on X is a linear combination of A entries.
    # The linear form on Y is a linear combination of B entries.
    #
    # And the contribution to C: the product contributes to C blocks per
    # c_contribs[c_block], and within each 2x2 C block, the s-th Strassen
    # term contributes to specific positions:
    #   c00 gets: s0(+1), s3(+1), s4(-1), s6(+1)
    #   c01 gets: s2(+1), s4(+1)
    #   c10 gets: s1(+1), s3(+1)
    #   c11 gets: s0(+1), s1(-1), s2(+1), s5(+1)
    
    # Level 2 Strassen linear forms on X (2x2):
    # s=0: X00+X11
    # s=1: X10+X11
    # s=2: X00
    # s=3: X11
    # s=4: X00+X01
    # s=5: X10-X00
    # s=6: X01-X11
    
    x_forms = [
        {(0,0):1, (1,1):1},      # s=0
        {(1,0):1, (1,1):1},      # s=1
        {(0,0):1},                # s=2
        {(1,1):1},                # s=3
        {(0,0):1, (0,1):1},      # s=4
        {(1,0):1, (0,0):-1},     # s=5
        {(0,1):1, (1,1):-1},     # s=6
    ]
    
    # Level 2 Strassen linear forms on Y (2x2):
    y_forms = [
        {(0,0):1, (1,1):1},      # s=0
        {(0,0):1},                # s=1
        {(1,0):1, (1,1):-1},     # s=2
        {(0,1):1, (0,0):-1},     # s=3
        {(1,1):1},                # s=4
        {(0,0):1, (0,1):1},      # s=5
        {(1,0):1, (1,1):1},      # s=6
    ]
    
    # Level 2 output: within a 2x2 block, which positions get which s terms:
    # c00: s0(+1), s3(+1), s4(-1), s6(+1)
    # c01: s2(+1), s4(+1)
    # c10: s1(+1), s3(+1)
    # c11: s0(+1), s1(-1), s2(+1), s5(+1)
    local_c_contribs = [
        [(0,1), (3,1), (4,-1), (6,1)],   # local (0,0)
        [(2,1), (4,1)],                   # local (0,1)
        [(1,1), (3,1)],                   # local (1,0)
        [(0,1), (1,-1), (2,1), (5,1)],   # local (1,1)
    ]
    
    # Now build U, V, W
    col = 0
    for t in range(7):
        for s in range(7):
            # Build U column: linear form on A
            # X = sum over (br, bc, coeff) in a_blocks[t] of A_block[br,bc]
            # x_form[s] is a dict mapping (li, lj) -> coeff
            # So the full linear form on A is:
            #   sum over (br, bc, coeff_t) in a_blocks[t]:
            #     sum over (li, lj, coeff_s) in x_forms[s]:
            #       coeff_t * coeff_s * A[br*2+li, bc*2+lj]
            for br, bc, coeff_t in a_blocks[t]:
                for (li, lj), coeff_s in x_forms[s].items():
                    a_row = br * 2 + li  # i index
                    a_col = bc * 2 + lj  # j index
                    U[a_row * m + a_col][col] += coeff_t * coeff_s
            
            # Build V column: linear form on B
            for br, bc, coeff_t in b_blocks[t]:
                for (li, lj), coeff_s in y_forms[s].items():
                    b_row = br * 2 + li  # j index
                    b_col = bc * 2 + lj  # k index
                    V[b_row * p + b_col][col] += coeff_t * coeff_s
            
            # Build W column: contribution to C
            # For each C block (cb_row, cb_col) that this t contributes to:
            for cb, coeff_t_out in c_contribs[0]:  # placeholder, will fix below
                pass
            
            # Actually, let me restructure. For each (t, s), the product
            # contributes to C blocks per c_contribs, and within each block,
            # per local_c_contribs[s].
            for cb_idx, (cb_row, cb_col) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
                # Find if t contributes to this C block
                coeff_t_out = None
                for (t_idx, c) in c_contribs[cb_idx]:
                    if t_idx == t:
                        coeff_t_out = c
                        break
                if coeff_t_out is None:
                    continue
                
                # Within this 2x2 C block, which local positions get s?
                for lc_idx, (lc_row, lc_col) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
                    coeff_s_out = None
                    for (s_idx, c) in local_c_contribs[lc_idx]:
                        if s_idx == s:
                            coeff_s_out = c
                            break
                    if coeff_s_out is None:
                        continue
                    
                    # Full C position:
                    c_row = cb_row * 2 + lc_row  # i index
                    c_col = cb_col * 2 + lc_col  # k index
                    W[c_col * n + c_row][col] += coeff_t_out * coeff_s_out
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
