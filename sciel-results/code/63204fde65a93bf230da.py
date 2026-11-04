"""
Program to find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Strategy:
1. Use the known rank-49 decomposition for 4x4 * 4x4 matrix multiplication (Strassen-based).
2. Use the rank-16 schoolbook decomposition for 4x4 * 4x1 (the remaining column of the 4x9 matrix).
3. Combine these to get a rank-114 decomposition for 4x4 * 4x9.

The key insight is that the 4x9 matrix can be split into two 4x4 blocks and one 4x1 block.
We apply the efficient rank-49 algorithm to each 4x4 block and the trivial rank-16 algorithm
to the 4x1 block.
"""

import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # We'll build U, V, W matrices of shape (nm, rank), (mp, rank), (pn, rank)
    # Rank will be 49 + 49 + 16 = 114

    # First, let's define the rank-49 decomposition for 4x4 * 4x4.
    # This uses the Strassen algorithm recursively.
    # The 2x2 Strassen decomposition has rank 7.
    # For 4x4, we split into 2x2 blocks and apply Strassen.
    # This gives 7 * 7 = 49 rank-1 terms.

    # Let's define the 2x2 Strassen factors first.
    # For 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # We need to represent this as a tensor decomposition.
    # Let's build the factors for 2x2 * 2x2 first, then generalize to 4x4.

    # Actually, let's directly build the 4x4 * 4x4 factors.
    # The 4x4 matrices are split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]] where each block is 2x2

    # For each of the 7 Strassen products, we have a 2x2 * 2x2 multiplication.
    # Each of these has rank 7.
    # So total rank = 7 * 7 = 49.

    # Let's define the 2x2 Strassen factors as lists of (U_vec, V_vec, W_vec)
    # where each is a vector of length 4 (for the 4 entries of a 2x2 matrix).

    # Indexing for 2x2 matrices:
    # [0, 1, 2, 3] = [A11, A12, A21, A22]

    # Define the 7 rank-1 terms for 2x2 * 2x2.
    # Each term is (u, v, w) where u, v are vectors of length 4 (for A and B entries),
    # and w is a vector of length 4 (for C entries).

    # M1 = (A11 + A22)(B11 + B22)
    # u1 = [1, 0, 0, 1], v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1] (contributes to C11 and C22)

    # M2 = (A21 + A22)B11
    # u2 = [0, 0, 1, 1], v2 = [1, 0, 0, 0]
    # w2 = [0, 0, 1, 0] (contributes to C21)

    # M3 = A11(B12 - B22)
    # u3 = [1, 0, 0, 0], v3 = [0, 1, 0, -1]
    # w3 = [0, 1, 0, 0] (contributes to C12)

    # M4 = A22(B21 - B11)
    # u4 = [0, 0, 0, 1], v4 = [0, 0, 1, -1]
    # w4 = [1, 0, 1, 0] (contributes to C11 and C21)

    # M5 = (A11 + A12)B22
    # u5 = [1, 1, 0, 0], v5 = [0, 0, 0, 1]
    # w5 = [-1, 0, 0, 0] (contributes to C11 with -1)

    # M6 = (A21 - A11)(B11 + B12)
    # u6 = [-1, 0, 1, 0], v6 = [1, 1, 0, 0]
    # w6 = [0, 0, 0, 1] (contributes to C22)

    # M7 = (A12 - A22)(B21 + B22)
    # u7 = [0, 1, 0, -1], v7 = [0, 0, 1, 1]
    # w7 = [1, 0, 0, 0] (contributes to C11)

    # Let's verify: C11 = M1 + M4 - M5 + M7
    # w1[0] + w4[0] - w5[0] + w7[0] = 1 + 1 - (-1) + 1 = 4? That's not right.
    # Let me re-check the Strassen formulas.

    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M3

    # Let me re-derive w vectors:
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M3

    # So:
    # w1 = [1, 0, 0, 1] (M1 contributes +1 to C11 and C22)
    # w2 = [0, 0, 1, -1] (M2 contributes +1 to C21 and -1 to C22)
    # w3 = [0, 1, 0, 1] (M3 contributes +1 to C12 and C22)
    # w4 = [1, 0, 1, 0] (M4 contributes +1 to C11 and C21)
    # w5 = [-1, 1, 0, 0] (M5 contributes -1 to C11 and +1 to C12)
    # w6 = [0, 0, 0, 1] (M6 contributes +1 to C22)
    # w7 = [1, 0, 0, 0] (M7 contributes +1 to C11)

    # Let's verify C11: w1[0] - w5[0] + w4[0] + w7[0] = 1 - (-1) + 1 + 1 = 4? Still wrong.
    # I think I'm confusing the contributions. Let me be more careful.

    # Each Mi is a scalar (product of two linear combinations).
    # C11 = M1 - M5 + M4 + M7 means:
    # The coefficient of M1 in C11 is +1
    # The coefficient of M5 in C11 is -1
    # The coefficient of M4 in C11 is +1
    # The coefficient of M7 in C11 is +1

    # So for the tensor decomposition, each rank-1 term corresponds to one Mi.
    # The w vector for term i has the coefficients of Mi in each C entry.

    # Let me redefine:
    # Term 1 (M1): u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1, 0, 0, 1]
    # Term 2 (M2): u2 = [0,0,1,1], v2 = [1,0,0,0], w2 = [0, 0, 1, -1]
    # Term 3 (M3): u3 = [1,0,0,0], v3 = [0,1,0,-1], w3 = [0, 1, 0, 1]
    # Term 4 (M4): u4 = [0,0,0,1], v4 = [0,0,1,-1], w4 = [1, 0, 1, 0]
    # Term 5 (M5): u5 = [1,1,0,0], v5 = [0,0,0,1], w5 = [-1, 1, 0, 0]
    # Term 6 (M6): u6 = [-1,0,1,0], v6 = [1,1,0,0], w6 = [0, 0, 0, 1]
    # Term 7 (M7): u7 = [0,1,0,-1], v7 = [0,0,1,1], w7 = [1, 0, 0, 0]

    # Now verify C11 = M1 - M5 + M4 + M7:
    # w1[0] - w5[0] + w4[0] + w7[0] = 1 - (-1) + 1 + 1 = 4. Still 4, not 1.

    # I see the issue. The w vectors should represent the contribution to each C entry,
    # but each Mi is a scalar, so the w vector should have 1s where Mi appears.
    # But the coefficients are ±1, so w entries should be ±1.

    # Let me re-examine. C11 = M1 - M5 + M4 + M7.
    # This means:
    # - M1 appears with coefficient +1 in C11
    # - M5 appears with coefficient -1 in C11
    # - M4 appears with coefficient +1 in C11
    # - M7 appears with coefficient +1 in C11

    # So for term 1 (M1), w1[0] = +1 (contributes +1 to C11)
    # For term 5 (M5), w5[0] = -1 (contributes -1 to C11)
    # For term 4 (M4), w4[0] = +1 (contributes +1 to C11)
    # For term 7 (M7), w7[0] = +1 (contributes +1 to C11)

    # Sum: 1 + (-1) + 1 + 1 = 2. Still not 1.

    # I think I have the wrong Strassen formulas. Let me look up the correct ones.

    # Correct Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M3

    # Let's verify with a simple example.
    # A = [[1,0],[0,1]], B = [[1,0],[0,1]]
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)*1 = 1
    # M3 = 1*(0-1) = -1
    # M4 = 1*(0-1) = -1
    # M5 = (1+0)*1 = 1
    # M6 = (0-1)*(1+0) = -1
    # M7 = (0-1)*(0+1) = -1
    # C11 = 4 - 1 + (-1) + (-1) = 1 ✓
    # C12 = -1 + 1 = 0 ✓
    # C21 = 1 + (-1) = 0 ✓
    # C22 = 4 - 1 + (-1) + (-1) = 1 ✓

    # So the formulas are correct. Now let's re-derive the w vectors.
    # C11 = M1 - M5 + M4 + M7
    # So:
    # w1[0] = 1 (M1 contributes +1 to C11)
    # w5[0] = -1 (M5 contributes -1 to C11)
    # w4[0] = 1 (M4 contributes +1 to C11)
    # w7[0] = 1 (M7 contributes +1 to C11)
    # Sum: 1 + (-1) + 1 + 1 = 2. But we need 1.

    # Wait, I'm summing the w vectors, but each term is separate.
    # In the tensor decomposition, we sum over all terms.
    # For C11, we sum w_i[0] * (u_i · a) * (v_i · b) for all i.
    # But that's not how it works. Each term i has a fixed u_i, v_i, w_i.
    # The contribution to C11 from term i is w_i[0] * (u_i · a) * (v_i · b).
    # But (u_i · a) * (v_i · b) is exactly Mi.
    # So the total contribution to C11 is sum_i w_i[0] * Mi.
    # We need sum_i w_i[0] * Mi = C11 = M1 - M5 + M4 + M7.
    # So we need:
    # w1[0] = 1, w2[0] = 0, w3[0] = 0, w4[0] = 1, w5[0] = -1, w6[0] = 0, w7[0] = 1
    # Sum: 1* M1 + 0*M2 + 0*M3 + 1*M4 + (-1)*M5 + 0*M6 + 1*M7 = M1 + M4 - M5 + M7 ✓

    # So the w vectors are:
    # w1 = [1, 0, 0, 1]  (M1: +1 to C11, 0 to C12, 0 to C21, +1 to C22)
    # w2 = [0, 0, 1, -1]  (M2: 0 to C11, 0 to C12, +1 to C21, -1 to C22)
    # w3 = [0, 1, 0, 1]  (M3: 0 to C11, +1 to C12, 0 to C21, +1 to C22)
    # w4 = [1, 0, 1, 0]  (M4: +1 to C11, 0 to C12, +1 to C21, 0 to C22)
    # w5 = [-1, 1, 0, 0]  (M5: -1 to C11, +1 to C12, 0 to C21, 0 to C22)
    # w6 = [0, 0, 0, 1]  (M6: 0 to C11, 0 to C12, 0 to C21, +1 to C22)
    # w7 = [1, 0, 0, 0]  (M7: +1 to C11, 0 to C12, 0 to C21, 0 to C22)

    # Now verify C22 = M1 - M2 + M6 + M3:
    # w1[3] - w2[3] + w6[3] + w3[3] = 1 - (-1) + 1 + 1 = 4. Still 4.
    # But we need: w1[3]*M1 + w2[3]*M2 + w3[3]*M3 + w4[3]*M4 + w5[3]*M5 + w6[3]*M6 + w7[3]*M7
    # = 1*M1 + (-1)*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    # = M1 - M2 + M3 + M6 ✓

    # Great, so the w vectors are correct.

    # Now, for 4x4 * 4x4, we split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2.

    # For each of the 7 Strassen products Mi, we have a 2x2 * 2x2 multiplication.
    # Each 2x2 * 2x2 has rank 7, so each Mi has rank 7.
    # Total rank = 7 * 7 = 49.

    # Let's build the factors for 4x4 * 4x4.
    # We'll create 49 rank-1 terms.
    # For each Strassen product index s (0 to 6) and each 2x2 Strassen term index t (0 to 6):
    #   The u vector is the u_t of the 2x2 Strassen term, applied to the appropriate block of A.
    #   The v vector is the v_t of the 2x2 Strassen term, applied to the appropriate block of B.
    #   The w vector is the w_s of the 4x4 Strassen term times the w_t of the 2x2 Strassen term.

    # Let's define the 2x2 Strassen factors:
    u2x2 = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1],  # M7: A12 - A22
    ]
    v2x2 = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [0, 0, 1, -1],  # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    w2x2 = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0],  # M7
    ]

    # For 4x4, the Strassen products are:
    # M1_44 = (A11 + A22)(B11 + B22)
    # M2_44 = (A21 + A22)B11
    # M3_44 = A11(B12 - B22)
    # M4_44 = A22(B21 - B11)
    # M5_44 = (A11 + A12)B22
    # M6_44 = (A21 - A11)(B11 + B12)
    # M7_44 = (A12 - A22)(B21 + B22)

    # For each M_s_44, we need to multiply two 2x2 matrices.
    # The u vector for M_s_44 is determined by which blocks of A are used and with what signs.
    # The v vector is determined by which blocks of B are used and with what signs.

    # Let's define the block indices for A and B in 4x4:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3

    # For M1_44 = (A11 + A22)(B11 + B22):
    # u1_44 = A11 + A22, v1_44 = B11 + B22
    # For M2_44 = (A21 + A22)B11:
    # u2_44 = A21 + A22, v2_44 = B11
    # etc.

    # Now, for each M_s_44, we apply the 2x2 Strassen decomposition.
    # So for M_s_44, we have 7 rank-1 terms, each corresponding to a 2x2 Strassen term.

    # Let's build the 49 terms for 4x4 * 4x4.

    # First, define the u and v vectors for each M_s_44 (as vectors of length 16 for the 4x4 matrix).
    # But actually, we can think of it as: for each s, u_s_44 is a linear combination of A blocks,
    # and v_s_44 is a linear combination of B blocks.

    # Let's define the block combinations:
    # For M1_44: u = A11 + A22, v = B11 + B22
    # For M2_44: u = A21 + A22, v = B11
    # For M3_44: u = A11, v = B12 - B22
    # For M4_44: u = A22, v = B21 - B11
    # For M5_44: u = A11 + A12, v = B22
    # For M6_44: u = A21 - A11, v = B11 + B12
    # For M7_44: u = A12 - A22, v = B21 + B22

    # Now, for each s, and for each t (0 to 6), we have a rank-1 term.
    # The u vector for term (s,t) is: for each entry in u2x2[t], multiply by the corresponding block in u_s_44.
    # Similarly for v.

    # Let's implement this.

    def get_block_indices(block_name):
        """Return the row and col indices for a 2x2 block in a 4x4 matrix."""
        if block_name == '11':
            return [0, 1], [0, 1]
        elif block_name == '12':
            return [0, 1], [2, 3]
        elif block_name == '21':
            return [2, 3], [0, 1]
        elif block_name == '22':
            return [2, 3], [2, 3]

    # Define the u and v block combinations for each s (0 to 6)
    u_blocks = [
        [('11', 1), ('22', 1)],  # M1: A11 + A22
        [('21', 1), ('22', 1)],  # M2: A21 + A22
        [('11', 1)],             # M3: A11
        [('22', 1)],             # M4: A22
        [('11', 1), ('12', 1)],  # M5: A11 + A12
        [('21', 1), ('11', -1)],  # M6: A21 - A11
        [('12', 1), ('22', -1)],  # M7: A12 - A22
    ]
    v_blocks = [
        [('11', 1), ('22', 1)],  # M1: B11 + B22
        [('11', 1)],             # M2: B11
        [('12', 1), ('22', -1)],  # M3: B12 - B22
        [('21', 1), ('11', -1)],  # M4: B21 - B11
        [('22', 1)],             # M5: B22
        [('11', 1), ('12', 1)],  # M6: B11 + B12
        [('21', 1), ('22', 1)],  # M7: B21 + B22
    ]

    # Now build the 49 terms for 4x4 * 4x4.
    U_44 = []
    V_44 = []
    W_44 = []

    for s in range(7):
        for t in range(7):
            # u vector for term (s,t)
            u_vec = np.zeros(16)
            for block_name, coeff in u_blocks[s]:
                rows, cols = get_block_indices(block_name)
                for r in rows:
                    for c in cols:
                        # The 2x2 Strassen term t uses u2x2[t] for the 2x2 matrix.
                        # The 2x2 matrix has entries at positions (0,0), (0,1), (1,0), (1,1)
                        # which correspond to local indices 0, 1, 2, 3.
                        local_r = rows.index(r)
                        local_c = cols.index(c)
                        local_idx = local_r * 2 + local_c
                        u_vec[r * 4 + c] = coeff * u2x2[t][local_idx]

            # v vector for term (s,t)
            v_vec = np.zeros(16)
            for block_name, coeff in v_blocks[s]:
                rows, cols = get_block_indices(block_name)
                for r in rows:
                    for c in cols:
                        local_r = rows.index(r)
                        local_c = cols.index(c)
                        local_idx = local_r * 2 + local_c
                        v_vec[r * 4 + c] = coeff * v2x2[t][local_idx]

            # w vector for term (s,t)
            # The w vector is w_s (for the 4x4 Strassen) times w_t (for the 2x2 Strassen).
            # w_s has 4 entries (for C11, C12, C21, C22).
            # w_t has 4 entries (for the 2x2 output).
            # The product is a 4x4 matrix, so we need to expand it to 16 entries.

            # First, get w_s for the 4x4 Strassen term s.
            # From earlier:
            w_s = [
                [1, 0, 0, 1],  # M1
                [0, 0, 1, -1],  # M2
                [0, 1, 0, 1],  # M3
                [1, 0, 1, 0],  # M4
                [-1, 1, 0, 0],  # M5
                [0, 0, 0, 1],  # M6
                [1, 0, 0, 0],  # M7
            ][s]

            # w_t is the 2x2 Strassen w vector.
            w_t = w2x2[t]

            # The 4x4 output block for C_s is a 2x2 matrix.
            # The (i,j) entry of this 2x2 matrix is w_s[i*2+j] * (sum over k,l w_t[k*2+l] * ...)
            # Actually, the 2x2 output from the 2x2 Strassen is:
            # C_local = sum_t w_t[t] * (u_t · a_local) * (v_t · b_local)
            # But we're decomposing the 2x2 * 2x2 into rank-1 terms, so each term (s,t) contributes
            # w_s[block] * w_t[local] to the corresponding entry.

            # Let me think differently. The 4x4 Strassen term s produces a 2x2 block C_s.
            # This 2x2 block is computed as a 2x2 * 2x2 multiplication.
            # The 2x2 * 2x2 is decomposed into 7 rank-1 terms (t=0 to 6).
            # For each t, the contribution to the 2x2 output is w_t[t] * (u_t · a_s) * (v_t · b_s).
            # But a_s and b_s are the 2x2 blocks from u_s_44 and v_s_44.

            # So for term (s,t), the contribution to the 4x4 output is:
            # For each block position (i,j) in the 4x4 output (i,j in {0,1} for the block, and local (k,l) in {0,1} for the 2x2):
            #   w_vec[i*4 + k, j*4 + l] += w_s[i*2+j] * w_t[k*2+l]

            # Wait, I'm confusing myself. Let me just build the w vector directly.

            w_vec = np.zeros(16)
            for i in range(2):
                for j in range(2):
                    # Block (i,j) of the 4x4 output
                    # w_s[i*2+j] is the coefficient for this block
                    for k in range(2):
                        for l in range(2):
                            # Local position (k,l) in the 2x2 block
                            # w_t[k*2+l] is the coefficient for this local position
                            # The output position in the 4x4 matrix is:
                            row = i * 2 + k
                            col = j * 2 + l
                            w_vec[row * 4 + col] = w_s[i * 2 + j] * w_t[k * 2 + l]

            U_44.append(u_vec)
            V_44.append(v_vec)
            W_44.append(w_vec)

    # Now we have 49 terms for 4x4 * 4x4.
    # We need two such decompositions (for the two 4x4 blocks of the 4x9 matrix) and one 4x4 * 4x1.

    # The 4x9 matrix B can be split into B1 (4x4), B2 (4x4), B3 (4x1).
    # A is 4x4.
    # C = A * B = A * [B1, B2, B3] = [A*B1, A*B2, A*B3]
    # A*B1 is 4x4, A*B2 is 4x4, A*B3 is 4x1.

    # So we use:
    # - Rank-49 decomposition for A*B1 (B1 is the first 4 columns of B)
    # - Rank-49 decomposition for A*B2 (B2 is the next 4 columns of B)
    # - Rank-16 schoolbook for A*B3 (B3 is the last 1 column of B)

    # Total rank = 49 + 49 + 16 = 114.

    # Now build the full U, V, W matrices.
    # U has shape (16, 114)
    # V has shape (36, 114)
    # W has shape (36, 114)

    U_full = np.zeros((16, 114))
    V_full = np.zeros((36, 114))
    W_full = np.zeros((36, 114))

    # First 49 terms: A*B1
    for idx in range(49):
        U_full[:, idx] = U_44[idx]
        # V for B1: columns 0-3 of B
        V_full[0:4, idx] = V_44[idx]
        # W for C1: rows 0-3 of C
        W_full[0:4, idx] = W_44[idx]

    # Next 49 terms: A*B2
    for idx in range(49):
        col_idx = 49 + idx
        U_full[:, col_idx] = U_44[idx]
        # V for B2: columns 4-7 of B
        V_full[4:8, col_idx] = V_44[idx]
        # W for C2: rows 4-7 of C
        W_full[4:8, col_idx] = W_44[idx]

    # Last 16 terms: A*B3 (schoolbook for 4x4 * 4x1)
    # A is 4x4, B3 is 4x1, C3 is 4x1.
    # Schoolbook: C3[i] = sum_j A[i,j] * B3[j]
    # This has 16 rank-1 terms.
    for i in range(4):
        for j in range(4):
            term_idx = 98 + i * 4 + j
            # U: A[i,j]
            U_full[i * 4 + j, term_idx] = 1
            # V: B3[j] (column 8 of B, row j)
            V_full[8 + j, term_idx] = 1
            # W: C3[i] (row 8 of C, column i)
            W_full[8 + i, term_idx] = 1

    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
    print(f"Rank: {len(result['U'][0])}")

    # Verify the decomposition
    n, m, p = 4, 4, 9
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])

    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Check: sum_r U[a,r] * V[b,r] * W[c,r] should equal T[a,b,c]
    max_err = 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(len(U[0])))
                max_err = max(max_err, abs(val - T[a, b, c]))
    print(f"Max error: {max_err}")
