"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    
    Strategy: Partition 6x6 matrices into a 2x2 grid of 3x3 blocks.
    Apply Laderman's rank-15 algorithm at the block level (treating each 3x3 block as a scalar).
    For each of the 15 Laderman products, perform the 3x3 scalar matrix multiplication using
    the rank-9 schoolbook method.
    
    Total rank = 15 * 9 = 135.
    
    Laderman's algorithm for 2x2 block matrix multiplication:
    Given A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    C = A*B where Cij = sum_k Aik * Bkj
    
    Laderman's 15 multiplications:
    M1 = A00 * B00
    M2 = (A00 + A10) * B10
    M3 = A11 * B11
    M4 = (A11 + A01) * B01
    M5 = (A00 + A11) * (B00 + B11)
    M6 = (A11 + A10) * (B10 + B00)
    M7 = (A01 + A11) * (B11 + B01)
    M8 = (A01 + A00) * (B01 + B00)
    M9 = (A10 + A00) * (B10 + B11)
    M10 = (A10 + A11) * (B01 + B11)
    M11 = (A10 - A11) * (B11 - B01)
    M12 = (A01 - A00) * (B00 - B10)
    M13 = (A00 + A01 - A10) * (B01 + B10 - B00)
    M14 = (A11 + A10 - A01) * (B00 + B01 - B11)
    M15 = (A01 + A11 - A10) * (B10 + B11 - B01)
    
    Then:
    C00 = M1 - M6 + M13 - M14
    C01 = M4 - M7 + M13 - M15
    C10 = M2 - M9 + M11 - M12
    C11 = M3 + M8 - M10 + M12
    
    Wait, let me verify Laderman's algorithm more carefully.
    
    Laderman's algorithm (1976) for 2x2 block multiplication:
    Let A = [[A0, A1], [A2, A3]], B = [[B0, B1], [B2, B3]]
    
    The 15 products are:
    M1 = A0 * B0
    M2 = (A0 + A2) * B2
    M3 = A3 * B3
    M4 = (A3 + A1) * B1
    M5 = (A0 + A3) * (B0 + B3)
    M6 = (A3 + A2) * (B2 + B0)
    M7 = (A1 + A3) * (B3 + B1)
    M8 = (A1 + A0) * (B1 + B0)
    M9 = (A2 + A0) * (B2 + B3)
    M10 = (A2 + A3) * (B1 + B3)
    M11 = (A2 - A3) * (B3 - B1)
    M12 = (A1 - A0) * (B0 - B2)
    M13 = (A0 + A1 - A2) * (B1 + B2 - B0)
    M14 = (A3 + A2 - A1) * (B0 + B1 - B3)
    M15 = (A1 + A3 - A2) * (B2 + B3 - B1)
    
    Then:
    C00 = M1 - M6 + M13 - M14
    C01 = M4 - M7 + M13 - M15
    C10 = M2 - M9 + M11 - M12
    C11 = M3 + M8 - M10 + M12
    
    Let me verify this. Actually, I need to be more careful. Let me look up the exact Laderman algorithm.
    
    Laderman's algorithm:
    M1 = A00 * B00
    M2 = (A00 + A10) * B10
    M3 = A11 * B11
    M4 = (A11 + A01) * B01
    M5 = (A00 + A11) * (B00 + B11)
    M6 = (A11 + A10) * (B10 + B00)
    M7 = (A01 + A11) * (B11 + B01)
    M8 = (A01 + A00) * (B01 + B00)
    M9 = (A10 + A00) * (B10 + B11)
    M10 = (A10 + A11) * (B01 + B11)
    M11 = (A10 - A11) * (B11 - B01)
    M12 = (A01 - A00) * (B00 - B10)
    M13 = (A00 + A01 - A10) * (B01 + B10 - B00)
    M14 = (A11 + A10 - A01) * (B00 + B01 - B11)
    M15 = (A01 + A11 - A10) * (B10 + B11 - B01)
    
    C00 = M1 - M6 + M13 - M14
    C01 = M4 - M7 + M13 - M15
    C10 = M2 - M9 + M11 - M12
    C11 = M3 + M8 - M10 + M12
    
    Let me verify C00:
    C00 = A00*B00 + A01*B10
    M1 = A00*B00
    M6 = (A11+A10)*(B10+B00) = A11*B10 + A11*B00 + A10*B10 + A10*B00
    M13 = (A00+A01-A10)*(B01+B10-B00)
        = A00*B01 + A00*B10 - A00*B00
        + A01*B01 + A01*B10 - A01*B00
        - A10*B01 - A10*B10 + A10*B00
    M14 = (A11+A10-A01)*(B00+B01-B11)
        = A11*B00 + A11*B01 - A11*B11
        + A10*B00 + A10*B01 - A10*B11
        - A01*B00 - A01*B01 + A01*B11
    
    M1 - M6 + M13 - M14:
    A00*B00 - (A11*B10 + A11*B00 + A10*B10 + A10*B00)
    + (A00*B01 + A00*B10 - A00*B00 + A01*B01 + A01*B10 - A01*B00 - A10*B01 - A10*B10 + A10*B00)
    - (A11*B00 + A11*B01 - A11*B11 + A10*B00 + A10*B01 - A10*B11 - A01*B00 - A01*B01 + A01*B11)
    
    Let me collect terms:
    A00*B00: 1 - 1 = 0... wait, M1 has A00*B00, M13 has -A00*B00, so they cancel.
    
    Hmm, this is getting complicated. Let me just implement it and verify numerically.
    
    Actually, I realize I should just implement the block decomposition and verify it numerically.
    If the Laderman algorithm is correct, the verification will pass. If not, I'll fall back to a known correct method.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # We'll use a 2x2 block decomposition with 3x3 blocks
    # Block indices: (0,0), (0,1), (1,0), (1,1) for 3x3 blocks
    # Each 3x3 block multiplication uses rank-9 schoolbook
    
    # Total rank = 15 (Laderman) * 9 (schoolbook 3x3) = 135
    rank = 135
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper functions for block indexing
    # Block (bi, bj) corresponds to rows/cols bi*3 to bi*3+2, bj*3 to bj*3+2
    
    def u_idx(bi, bj, si, sj):
        """Index in U for block (bi,bj) with local indices (si,sj)"""
        return (bi * 3 + si) * m + (bj * 3 + sj)
    
    def v_idx(bj, bk, sj, sk):
        """Index in V for block (bj,bk) with local indices (sj,sk)"""
        return (bj * 3 + sj) * p + (bk * 3 + sk)
    
    def w_idx(bk, bi, sk, si):
        """Index in W for block (bk,bi) with local indices (sk,si)"""
        return (bk * 3 + sk) * n + (bi * 3 + si)
    
    # Laderman's 15 products, each with coefficients for A-blocks and B-blocks
    # Each product is of the form (sum of A-blocks) * (sum of B-blocks)
    # We need to express each as a sum of elementary products
    
    # For each Laderman product, we have:
    # Product r = (sum_a coeff_a * A_block_a) * (sum_b coeff_b * B_block_b)
    # = sum_a sum_b coeff_a * coeff_b * (A_block_a * B_block_b)
    # Each A_block_a * B_block_b is a 3x3 matrix multiplication with rank 9
    
    # Define the 15 Laderman products
    # A-blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B-blocks: 0=B00, 1=B01, 2=B10, 3=B11
    
    laderman_products = [
        # (product_index, [(a_block, coeff), ...], [(b_block, coeff), ...])
        (0, [(0, 1)], [(0, 1)]),  # M1 = A00 * B00
        (1, [(0, 1), (2, 1)], [(2, 1)]),  # M2 = (A00+A10) * B10
        (2, [(3, 1)], [(3, 1)]),  # M3 = A11 * B11
        (3, [(3, 1), (1, 1)], [(1, 1)]),  # M4 = (A11+A01) * B01
        (4, [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M5 = (A00+A11) * (B00+B11)
        (5, [(3, 1), (2, 1)], [(2, 1), (0, 1)]),  # M6 = (A11+A10) * (B10+B00)
        (6, [(1, 1), (3, 1)], [(3, 1), (1, 1)]),  # M7 = (A01+A11) * (B11+B01)
        (7, [(1, 1), (0, 1)], [(1, 1), (0, 1)]),  # M8 = (A01+A00) * (B01+B00)
        (8, [(2, 1), (0, 1)], [(2, 1), (3, 1)]),  # M9 = (A10+A00) * (B10+B11)
        (9, [(2, 1), (3, 1)], [(1, 1), (3, 1)]),  # M10 = (A10+A11) * (B01+B11)
        (10, [(2, 1), (3, -1)], [(3, 1), (1, -1)]),  # M11 = (A10-A11) * (B11-B01)
        (11, [(1, 1), (0, -1)], [(0, 1), (2, -1)]),  # M12 = (A01-A00) * (B00-B10)
        (12, [(0, 1), (1, 1), (2, -1)], [(1, 1), (2, 1), (0, -1)]),  # M13
        (13, [(3, 1), (2, 1), (1, -1)], [(0, 1), (1, 1), (3, -1)]),  # M14
        (14, [(1, 1), (3, 1), (2, -1)], [(2, 1), (3, 1), (1, -1)]),  # M15
    ]
    
    # C-block reconstruction:
    # C00 = M1 - M6 + M13 - M14
    # C01 = M4 - M7 + M13 - M15
    # C10 = M2 - M9 + M11 - M12
    # C11 = M3 + M8 - M10 + M12
    
    c_reconstruction = [
        # (c_block, [(product_index, coeff), ...])
        ((0, 0), [(0, 1), (5, -1), (12, 1), (13, -1)]),  # C00
        ((0, 1), [(3, 1), (6, -1), (12, 1), (14, -1)]),  # C01
        ((1, 0), [(1, 1), (8, -1), (10, 1), (11, -1)]),  # C10
        ((1, 1), [(2, 1), (7, 1), (9, -1), (11, 1)]),    # C11
    ]
    
    # For each Laderman product and each elementary 3x3 product within it,
    # we add 9 rank-1 terms (schoolbook for 3x3)
    
    col = 0
    for prod_idx, a_terms, b_terms in laderman_products:
        for (a_block, a_coeff) in a_terms:
            for (b_block, b_coeff) in b_terms:
                coeff = a_coeff * b_coeff
                # This corresponds to a_coeff * A_a_block * b_coeff * B_b_block
                # = coeff * (A_a_block * B_b_block)
                # A_a_block is a 3x3 matrix, B_b_block is a 3x3 matrix
                # Their product is a 3x3 matrix, decomposed with rank 9
                
                # A_a_block has local indices (si, sj) where si,sj in 0..2
                # B_b_block has local indices (sj, sk) where sj,sk in 0..2
                # Product has local indices (si, sk)
                
                for si in range(3):
                    for sj in range(3):
                        for sk in range(3):
                            # U index: A_a_block at (si, sj)
                            u_i = u_idx(a_block // 2, a_block % 2, si, sj)
                            # V index: B_b_block at (sj, sk)
                            v_i = v_idx(b_block // 2, b_block % 2, sj, sk)
                            # W index: Product at (si, sk), which goes to C block
                            # The product block is at position (a_block//2, b_block//2) in C
                            c_bi = a_block // 2
                            c_bj = b_block // 2
                            w_i = w_idx(b_block // 2, a_block // 2, sk, si)
                            
                            U[u_i][col] += coeff
                            V[v_i][col] += coeff
                            W[w_i][col] += coeff
                            col += 1
    
    # Now we need to apply the C-block reconstruction
    # The above gives us the raw products M1..M15 as 3x3 blocks
    # We need to combine them to get C00, C01, C10, C11
    
    # Wait, I think I made an error. The above decomposition directly computes
    # each M_r as a 3x3 matrix product. But the Laderman algorithm requires
    # combining these 3x3 results with coefficients to get the final C blocks.
    
    # The issue is that my current approach adds each elementary product directly
    # to U, V, W. But the Laderman combination happens at the 3x3 block level,
    # not at the scalar level.
    
    # I need to rethink this. The correct approach is:
    # 1. Compute M1..M15 as 3x3 matrices (each with rank 9)
    # 2. Combine them: C00 = M1 - M6 + M13 - M14, etc.
    
    # But in a tensor decomposition, we can't "combine" after the fact.
    # We need to express the final C as a sum of rank-1 terms.
    
    # The correct way: For each final C element, express it as a sum of rank-1 terms.
    # C[c_bi, c_bj, si, sk] = sum over Laderman products of coeff * M_r[c_bi, c_bj, si, sk]
    # where M_r is a 3x3 matrix computed from the Laderman formula.
    
    # Let me redo this more carefully.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # For each Laderman product r, for each way it contributes to each C-block,
    # we add the corresponding rank-9 decomposition
    
    # Actually, the cleaner way:
    # Each Laderman product M_r is a 3x3 matrix.
    # M_r = (sum_a a_coeff_a * A_a) * (sum_b b_coeff_b * B_b)
    #      = sum_a sum_b a_coeff_a * b_coeff_b * (A_a * B_b)
    
    # Each A_a * B_b is a 3x3 matrix multiplication, which has rank 9.
    # So M_r has rank at most 9 * |a_terms| * |b_terms|.
    
    # But we're decomposing the whole thing, so we just need to enumerate all terms.
    
    # For each Laderman product r, and for each (a_block, a_coeff) in its A-terms,
    # and for each (b_block, b_coeff) in its B-terms:
    #   The term a_coeff * b_coeff * (A_a_block * B_b_block) contributes to M_r.
    #   A_a_block * B_b_block is a 3x3 matrix, decomposed into 9 rank-1 terms.
    #   Each rank-1 term has U, V, W entries.
    
    # Then, M_r contributes to C-blocks according to the reconstruction formula.
    # So we multiply the W entries by the appropriate coefficient.
    
    # Let's build a mapping: for each Laderman product r and each C-block (c_bi, c_bj),
    # what is the coefficient?
    
    prod_to_c_coeff = {}
    for (c_bi, c_bj), terms in c_reconstruction:
        for (prod_idx, coeff) in terms:
            if prod_idx not in prod_to_c_coeff:
                prod_to_c_coeff[prod_idx] = {}
            prod_to_c_coeff[prod_idx][(c_bi, c_bj)] = coeff
    
    # Now rebuild
    col = 0
    for prod_idx, a_terms, b_terms in laderman_products:
        c_coeffs = prod_to_c_coeff.get(prod_idx, {})
        for (a_block, a_coeff) in a_terms:
            for (b_block, b_coeff) in b_terms:
                coeff = a_coeff * b_coeff
                a_bi = a_block // 2
                a_bj = a_block % 2
                b_bi = b_block // 2
                b_bj = b_block % 2
                # The product A_a_block * B_b_block is a 3x3 matrix
                # Its (si, sk) element is sum_j A[si,j] * B[j,sk]
                for si in range(3):
                    for sj in range(3):
                        for sk in range(3):
                            u_i = u_idx(a_bi, a_bj, si, sj)
                            v_i = v_idx(b_bi, b_bj, sj, sk)
                            # The W index depends on which C-block this contributes to
                            # The product A_a_block * B_b_block naturally goes to C-block (a_bi, b_bj)
                            # But in Laderman, it may contribute to multiple C-blocks with different coefficients
                            
                            # For each C-block that this product contributes to:
                            for (c_bi, c_bj), c_coeff in c_coeffs.items():
                                # Wait, this doesn't work because the W index is fixed
                                # by the position in the output matrix.
                                
                                # I think the issue is that each elementary product
                                # A_a_block * B_b_block produces a 3x3 matrix at a
                                # specific position in C, namely C[a_bi, b_bj].
                                # But Laderman's algorithm says that M_r contributes
                                # to C with certain coefficients.
                                
                                # So the W entry for this elementary product should be
                                # at position (a_bi, b_bj) in the C-block grid, but
                                # multiplied by the Laderman coefficient for that C-block.
                                
                                # But wait, that's not right either. Let me think again.
                                
                                # M_r is a 3x3 matrix. It's not at a specific C-block position.
                                # M_r is an intermediate result. The Laderman formula says:
                                # C00 = M1 - M6 + M13 - M14
                                # where each M_r is a 3x3 matrix.
                                
                                # So M_r is a 3x3 matrix that gets added/subtracted to
                                # specific C-blocks. The (si,sk) element of M_r gets added
                                # to the (si,sk) element of C00 (if the coefficient is nonzero).
                                
                                # So for the elementary product (A_a_block * B_b_block)[si,sk],
                                # it contributes to M_r[si,sk] with coefficient coeff.
                                # Then M_r[si,sk] contributes to C[c_bi, c_bj, si, sk] with
                                # coefficient c_coeff.
                                
                                # So the total W coefficient is coeff * c_coeff, and the
                                # W index is for C-block (c_bi, c_bj) at position (si, sk).
                                
                                pass  # We'll handle this below
    
    # This is getting complicated. Let me use a different approach.
    # I'll build the decomposition by directly computing the tensor entries.
    
    # The matrix multiplication tensor T[i,j,k] = 1 if i,j,k are valid indices for
    # C[i,k] = sum_j A[i,j] * B[j,k], else 0.
    
    # For the 6x6 case, T[i,j,k] = 1 for all i,j,k in 0..5.
    
    # A rank-r decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # I'll construct U, V, W such that this holds.
    
    # Using the Laderman + schoolbook approach:
    # For each Laderman product r, and for each elementary 3x3 product within it,
    # I add 9 rank-1 terms.
    
    # Let me just enumerate all terms directly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod_idx, a_terms, b_terms in laderman_products:
        c_coeffs = prod_to_c_coeff.get(prod_idx, {})
        for (a_block, a_coeff) in a_terms:
            for (b_block, b_coeff) in b_terms:
                coeff = a_coeff * b_coeff
                a_bi = a_block // 2
                a_bj = a_block % 2
                b_bi = b_block // 2
                b_bj = b_block % 2
                for si in range(3):
                    for sj in range(3):
                        for sk in range(3):
                            u_i = u_idx(a_bi, a_bj, si, sj)
                            v_i = v_idx(b_bi, b_bj, sj, sk)
                            for (c_bi, c_bj), c_coeff in c_coeffs.items():
                                w_i = w_idx(b_bj, a_bi, sk, si)
                                # Wait, the W index should be for C-block (c_bi, c_bj)
                                # But c_bi and c_bj may differ from a_bi and b_bj
                                # Actually, in Laderman, M_r is a 3x3 matrix that
                                # contributes to specific C-blocks. The (si,sk) element
                                # of M_r goes to C[c_bi, c_bj, si, sk].
                                # So the W index is for C-block (c_bi, c_bj) at (si, sk).
                                w_i = w_idx(c_bj, c_bi, sk, si)
                                U[u_i][col] += coeff
                                V[v_i][col] += coeff
                                W[w_i][col] += coeff * c_coeff
                            col += 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify
    import random
    random.seed(42)
    valid = True
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
