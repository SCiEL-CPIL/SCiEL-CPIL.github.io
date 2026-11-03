"""Deterministic Strassen-based decomposition for 4x4 * 4x7."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]


def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 * 4x4 using Strassen.
    Returns (U, V, W) as lists of lists, shapes (16,28), (16,28), (16,28).
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    """
    n = m = p = 4
    R = 28
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    # 4x4 split into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]  where Axy is 2x2 with rows i=2x+{0,1}, cols j=2y+{0,1}
    # B = [[B00, B01], [B10, B11]]  where Bxy is 2x2 with rows j=2x+{0,1}, cols k=2y+{0,1}
    # C = A*B: Cxy = sum_z A_xz * B_zy

    # For each block pair (x, y, z), we need A_xz * B_zy contribution to C_xy.
    # C_xy has rows i=2x+{0,1}, cols k=2y+{0,1}.
    # A_xz has rows i=2x+{0,1}, cols j=2z+{0,1}.
    # B_zy has rows j=2z+{0,1}, cols k=2y+{0,1}.

    # Strassen for 2x2*2x2 (rank 7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # We have 4 blocks of A (x,z in {0,1}^2) and 4 blocks of B (z,y in {0,1}^2).
    # For each (x,y) output block, we need sum over z of A_xz * B_zy.
    # That's 4 terms of 2x2*2x2, each needing rank 7 via Strassen = 28 total.
    # But we can share Strassen intermediates across z!
    # Actually, the standard approach: treat the whole 4x4*4x4 as one problem.
    # The 4x4*4x4 can be done with 4x4 Strassen (rank 7^2 = 49? No...).
    # Actually, bilinear complexity of 4x4*4x4 is known to be 49 (from 7^2 via Strassen on 2x2 blocks).
    # Wait, but we can do better. The known rank of 4x4*4x4 is 49 using Strassen.
    # But there might be better. Let me just use the standard Strassen composition.

    # Standard: 4x4*4x4 via 2x2 block Strassen:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each 2x2*2x2 uses rank 7, and we have 4 such products, but they share terms.
    # Total rank: we need to compute A00*B00, A00*B01, A00*B10, A00*B11,
    # A01*B00, A01*B01, A01*B10, A01*B11,
    # A10*B00, A10*B01, A10*B10, A10*B11,
    # A11*B00, A11*B01, A11*B10, A11*B11
    # That's 16 products of 2x2*2x2, each rank 7 = 112. Too many.

    # Better: use Strassen directly on the 4x4 by treating it as 2x2 blocks.
    # The 4x4*4x4 tensor decomposes as a 4x4 grid of 2x2*2x2 tensors.
    # Using Strassen on the outer 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # Each Mj is a 2x2*2x2 product, needing rank 7.
    # Total: 7 * 7 = 49. But we can do better with careful sharing.
    # For now, let's just use schoolbook for 4x4*4x4 (rank 64) and combine with 4x4*4x3.

    # Actually, let me just build it properly with the 7 Strassen terms.
    # Each Strassen term is a bilinear form on the 2x2 blocks.

    # Let me index the rank-1 terms.
    # For Strassen term s (0..6), we need to define which 2x2*2x2 products it involves.
    # M1 = (A00+A11)(B00+B11): this involves A00*B00, A00*B11, A11*B00, A11*B11
    # For each such 2x2*2x2, we use rank-7 Strassen.

    # This is getting complex. Let me just use schoolbook for the 4x4*4x4 part.
    # 4x4*4x4 schoolbook: rank 64.
    # 4x4*4x3 schoolbook: rank 48.
    # Total: 112. That's the current baseline.

    # To improve, I need Strassen. Let me implement it carefully.

    # For a 2x2*2x2 product (A2*B2 -> C2), Strassen rank-7:
    # The 16 entries of A2 are at positions (i,j) for i,j in 0,1.
    # The 16 entries of B2 are at positions (j,k) for j,k in 0,1.
    # The 16 entries of C2 are at positions (i,k) for i,k in 0,1.

    # Strassen intermediates:
    # M1 = (A2[0,0]+A2[1,1])*(B2[0,0]+B2[1,1])
    # M2 = (A2[1,0]+A2[1,1])*B2[0,0]
    # M3 = A2[0,0]*(B2[0,1]-B2[1,1])
    # M4 = A2[1,1]*(B2[1,0]-B2[0,0])
    # M5 = (A2[0,0]+A2[0,1])*B2[1,1]
    # M6 = (A2[1,0]-A2[0,0])*(B2[0,0]+B2[0,1])
    # M7 = (A2[0,1]-A2[1,1])*(B2[1,0]+B2[1,1])

    # C2[0,0] = M1+M4-M5+M7
    # C2[0,1] = M3+M5
    # C2[1,0] = M2+M4
    # C2[1,1] = M1-M2+M3+M6

    # Now for 4x4*4x4:
    # A is 4x4, split into A00, A01, A10, A11 (each 2x2)
    # B is 4x4, split into B00, B01, B10, B11 (each 2x2)
    # C = A*B, split into C00, C01, C10, C11 (each 2x2)

    # Using outer Strassen:
    # M1_out = (A00+A11)*(B00+B11)  [2x2*2x2]
    # M2_out = (A10+A11)*B00         [2x2*2x2]
    # M3_out = A00*(B01-B11)         [2x2*2x2]
    # M4_out = A11*(B10-B00)         [2x2*2x2]
    # M5_out = (A00+A01)*B11         [2x2*2x2]
    # M6_out = (A10-A00)*(B00+B01)   [2x2*2x2]
    # M7_out = (A01-A11)*(B10+B11)   [2x2*2x2]

    # C00 = M1_out + M4_out - M5_out + M7_out
    # C01 = M3_out + M5_out
    # C10 = M2_out + M4_out
    # C11 = M1_out - M2_out + M3_out + M6_out

    # Each M*_out is a 2x2*2x2 product. Using Strassen on each: rank 7 each.
    # Total: 7 * 7 = 49 rank-1 terms.

    # Let me implement this.
    # For each outer Strassen term s (0..6), it corresponds to a bilinear form
    # (linear combo of A blocks) * (linear combo of B blocks) -> 2x2 result.
    # Then for that 2x2*2x2, I use inner Strassen (rank 7).

    # Let me define the outer Strassen terms:
    # For each term, I need:
    #   A_coeff: dict from (x,z) to coefficient (for A_xz block)
    #   B_coeff: dict from (z,y) to coefficient (for B_zy block)
    #   C_target: (x,y) with sign (for C_xy)

    outer_terms = [
        # (A_coeffs, B_coeffs, [(C_block, sign), ...])
        # M1_out = (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         [((0,0), 1), ((1,1), 1)]),
        # M2_out = (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         [((1,0), 1)]),
        # M3_out = A00*(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         [((0,1), 1)]),
        # M4_out = A11*(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         [((0,0), 1), ((1,0), 1)]),
        # M5_out = (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         [((0,0), -1), ((0,1), 1)]),
        # M6_out = (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         [((1,1), 1)]),
        # M7_out = (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         [((0,0), 1)]),
    ]

    # Inner Strassen for 2x2*2x2 (rank 7):
    # For a 2x2 A_block at rows i=2x+{0,1}, cols j=2z+{0,1}
    # and 2x2 B_block at rows j=2z+{0,1}, cols k=2y+{0,1}
    # producing 2x2 C_block at rows i=2x+{0,1}, cols k=2y+{0,1}

    inner_strassen = [
        # (A_indices, B_indices, C_indices_with_signs)
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)],
         [((0,0,1), (1,1,1))]),  # contributes to C[0,0] and C[1,1]
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        ([(1,0,1),(1,1,1)], [(0,0,1)],
         [((1,0,1),)]),
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        ([(0,0,1)], [(0,1,1),(1,1,-1)],
         [((0,1,1),)]),
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        ([(1,1,1)], [(1,0,1),(0,0,-1)],
         [((0,0,1),(1,0,1))]),
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        ([(0,0,1),(0,1,1)], [(1,1,1)],
         [((0,0,-1),(0,1,1))]),
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],
         [((1,1,1),)]),
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],
         [((0,0,1),)]),
    ]

    # This is getting very complex. Let me just use schoolbook for 4x4*4x4.
    # Rank 64 for 4x4*4x4, rank 48 for 4x4*4x3, total 112.

    # Actually, let me try a different approach: just build the schoolbook
    # and return it. The structural change will be in trying to optimize.

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return U, V, W


def _build_4x4x3_schoolbook():
    """Build rank-48 schoolbook decomposition for 4x4 * 4x3.
    Returns (U, V, W) as lists of lists, shapes (16,48), (12,48), (12,48).
    """
    n, m, p = 4, 4, 3
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook baseline: rank 112
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Target tensor
    target = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target.add((i * m + j, j * p + k, k * n + i))

    def verify(U, V, W):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for (u, v, w) in target:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True

    # Build 4x4*4x4 Strassen (rank 49) + 4x4*4x3 schoolbook (rank 48) = 97
    # But we need to verify this works. Let me implement the Strassen properly.

    if time.time() > deadline - 60:
        return best

    # Implement 4x4*4x4 with Strassen (rank 49)
    # and 4x4*4x3 with schoolbook (rank 48)
    # Total rank: 97

    R_strassen = 49
    R_schoolbook_3 = 48
    R_total = R_strassen + R_schoolbook_3

    U_new = [[0.0] * R_total for _ in range(n * m)]
    V_new = [[0.0] * R_total for _ in range(m * p)]
    W_new = [[0.0] * R_total for _ in range(p * n)]

    # Part 1: 4x4*4x4 (columns 0..48)
    # Using outer Strassen on 2x2 blocks of 2x2 matrices.
    # For each outer Strassen term s (0..6), and inner Strassen term t (0..6),
    # we get a rank-1 term.

    # Outer Strassen terms for 4x4*4x4:
    # A blocks: A_xz where x in {0,1} (row block), z in {0,1} (col block)
    # B blocks: B_zy where z in {0,1} (row block), y in {0,1} (col block)
    # C blocks: C_xy where x in {0,1} (row block), y in {0,1} (col block)

    # Outer Strassen:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 (+1) and C11 (+1)
    # M2 = (A10+A11)B00 -> contributes to C10 (+1)
    # M3 = A00(B01-B11) -> contributes to C01 (+1)
    # M4 = A11(B10-B00) -> contributes to C00 (+1) and C10 (+1)
    # M5 = (A00+A01)B11 -> contributes to C00 (-1) and C01 (+1)
    # M6 = (A10-A00)(B00+B01) -> contributes to C11 (+1)
    # M7 = (A01-A11)(B10+B11) -> contributes to C00 (+1)

    outer_strassen_4x4 = [
        # (A_block_coeffs, B_block_coeffs, [(C_block, sign), ...])
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         [((0,0), 1), ((1,1), 1)]),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         [((1,0), 1)]),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         [((0,1), 1)]),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         [((0,0), 1), ((1,0), 1)]),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         [((0,0), -1), ((0,1), 1)]),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         [((1,1), 1)]),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         [((0,0), 1)]),
    ]

    # Inner Strassen for 2x2*2x2:
    # A2 is 2x2 with local indices (i_local, j_local)
    # B2 is 2x2 with local indices (j_local, k_local)
    # C2 is 2x2 with local indices (i_local, k_local)

    # M1 = (A2[0,0]+A2[1,1])*(B2[0,0]+B2[1,1])
    # M2 = (A2[1,0]+A2[1,1])*B2[0,0]
    # M3 = A2[0,0]*(B2[0,1]-B2[1,1])
    # M4 = A2[1,1]*(B2[1,0]-B2[0,0])
    # M5 = (A2[0,0]+A2[0,1])*B2[1,1]
    # M6 = (A2[1,0]-A2[0,0])*(B2[0,0]+B2[0,1])
    # M7 = (A2[0,1]-A2[1,1])*(B2[1,0]+B2[1,1])

    # C2[0,0] = M1+M4-M5+M7
    # C2[0,1] = M3+M5
    # C2[1,0] = M2+M4
    # C2[1,1] = M1-M2+M3+M6

    inner_strassen_2x2 = [
        # (A2_coeffs, B2_coeffs, [(C2_target, sign), ...])
        # A2_coeffs: list of (i_local, j_local, coeff)
        # B2_coeffs: list of (j_local, k_local, coeff)
        # C2_target: (i_local, k_local)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)],
         [((0,0), 1), ((1,1), 1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],
         [((1,0), 1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)],
         [((0,1), 1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)],
         [((0,0), 1), ((1,0), 1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)],
         [((0,0), -1), ((0,1), 1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],
         [((1,1), 1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],
         [((0,0), 1)]),
    ]

    col = 0
    for outer_s in range(7):
        A_coeffs, B_coeffs, C_targets = outer_strassen_4x4[outer_s]
        for inner_t in range(7):
            A2_coeffs, B2_coeffs, C2_targets = inner_strassen_2x2[inner_t]
            # This rank-1 term:
            # U factor: for each (i,j) in 4x4, sum over contributions
            # V factor: for each (j,k) in 4x4, sum over contributions
            # W factor: for each (k,i) in 4x4, sum over contributions

            # U[i*4+j] = product of A coefficients
            # For A_block (x,z) with coeff c, and inner A2 (i_local, j_local) with coeff d:
            # i = 2*x + i_local, j = 2*z + j_local
            # U[i*4+j] += c * d (but only if this is the first time we see this (i,j) for this term)
            # Actually, U[i*4+j] = sum over all (x, i_local, z, j_local) with i=2x+i_local, j=2z+j_local
            # of A_coeff[(x,z)] * A2_coeff[(i_local,j_local)]

            # Let me build the U, V, W vectors for this rank-1 term.
            U_vec = [0.0] * (n * m)
            V_vec = [0.0] * (m * p)
            W_vec = [0.0] * (p * n)

            # Build U_vec
            for (x, z), c in A_coeffs.items():
                for (i_local, j_local, d) in A2_coeffs:
                    i = 2 * x + i_local
                    j = 2 * z + j_local
                    U_vec[i * m + j] += c * d

            # Build V_vec
            for (z, y), c in B_coeffs.items():
                for (j_local, k_local, d) in B2_coeffs:
                    j = 2 * z + j_local
                    k = 2 * y + k_local
                    V_vec[j * p + k] += c * d

            # Build W_vec
            for (x, y), c in C_targets:
                for (i_local, k_local, d) in C2_targets:
                    i = 2 * x + i_local
                    k = 2 * y + k_local
                    W_vec[k * n + i] += c * d

            # Only add if not all zeros
            if any(abs(x) > 0 for x in U_vec + V_vec + W_vec):
                for i in range(n * m):
                    U_new[i][col] = U_vec[i]
                for j in range(m * p):
                    V_new[j][col] = V_vec[j]
                for k in range(p * n):
                    W_new[k][col] = W_vec[k]
                col += 1

    # Part 2: 4x4*4x3 schoolbook (columns 49..96)
    offset = col
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if k < 4:  # Only for the 4x4*4x4 part (k in 0..3)
                    pass  # Already handled
                # For 4x4*4x3, k in 0..2, but we need to be careful
                # Actually, the 4x4*4x3 part handles k in 0..2 for the full 4x4 A.
                # But the 4x4*4x4 part already handles k in 0..3.
                # So for the 4x4*4x3 part, I should only add terms for k in 0..2
                # that are NOT already covered by the 4x4*4x4 part.
                # Wait, that's not right. The 4x4*4x7 = 4x4*4x4 + 4x4*4x3.
                # The 4x4*4x4 handles k in 0..3.
                # The 4x4*4x3 handles k in 0..2 for the "extra" 3 columns.
                # But that's double counting!

    # I'm confusing myself. Let me restart with a cleaner approach.
    # 4x4 * 4x7 = 4x4 * 4x4 + 4x4 * 4x3
    # The 4x4*4x4 part: A is 4x4, B_left is 4x4 (cols 0..3), C_left is 4x4 (cols 0..3)
    # The 4x4*4x3 part: A is 4x4, B_right is 4x3 (cols 0..2), C_right is 4x3 (cols 0..2)
    # Total C is 4x7.

    # For the 4x4*4x4 part, I use Strassen (rank 49).
    # For the 4x4*4x3 part, I use schoolbook (rank 48).
    # Total: 97.

    # But I need to make sure the indices are correct.
    # For 4x4*4x4: U indices (i*4+j), V indices (j*4+k) for k in 0..3, W indices (k*4+i) for k in 0..3.
    # For 4x4*4x3: U indices (i*4+j), V indices (j*3+k) for k in 0..2, W indices (k*4+i) for k in 0..2.
    # But wait, the overall V has shape (m*p, rank) = (4*7, rank) = (28, rank).
    # And W has shape (p*n, rank) = (7*4, rank) = (28, rank).

    # So for the 4x4*4x4 part, V indices are j*7+k for k in 0..3.
    # And W indices are k*4+i for k in 0..3.

    # For the 4x4*4x3 part, V indices are j*7+k for k in 4..6.
    # And W indices are k*4+i for k in 4..6.

    # Let me rebuild.
    U_new = [[0.0] * R_total for _ in range(n * m)]
    V_new = [[0.0] * R_total for _ in range(m * p)]
    W_new = [[0.0] * R_total for _ in range(p * n)]

    col = 0
    # Part 1: 4x4*4x4 Strassen
    for outer_s in range(7):
        A_coeffs, B_coeffs, C_targets = outer_strassen_4x4[outer_s]
        for inner_t in range(7):
            A2_coeffs, B2_coeffs, C2_targets = inner_strassen_2x2[inner_t]
            U_vec = [0.0] * (n * m)
            V_vec = [0.0] * (m * p)
            W_vec = [0.0] * (p * n)

            for (x, z), c in A_coeffs.items():
                for (i_local, j_local, d) in A2_coeffs:
                    i = 2 * x + i_local
                    j = 2 * z + j_local
                    U_vec[i * m + j] += c * d

            for (z, y), c in B_coeffs.items():
                for (j_local, k_local, d) in B2_coeffs:
                    j = 2 * z + j_local
                    k = 2 * y + k_local
                    V_vec[j * p + k] += c * d

            for (x, y), c in C_targets:
                for (i_local, k_local, d) in C2_targets:
                    i = 2 * x + i_local
                    k = 2 * y + k_local
                    W_vec[k * n + i] += c * d

            if any(abs(x) > 0 for x in U_vec + V_vec + W_vec):
                for i in range(n * m):
                    U_new[i][col] = U_vec[i]
                for j in range(m * p):
                    V_new[j][col] = V_vec[j]
                for k in range(p * n):
                    W_new[k][col] = W_vec[k]
                col += 1

    # Part 2: 4x4*4x3 schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U_new[i * m + j][col] = 1.0
                V_new[j * p + (4 + k)][col] = 1.0
                W_new[(4 + k) * n + i][col] = 1.0
                col += 1

    # Verify
    if verify(U_new, V_new, W_new):
        best = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = R_total

    # Try to reduce rank further
    if time.time() > deadline - 10:
        return best

    # Try rank 96, 95, ...
    for trial_rank in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if time.time() > deadline - 5:
            break
        # Just truncate the last column and see if it still works
        U_t = [row[:trial_rank] for row in best["U"]]
        V_t = [row[:trial_rank] for row in best["V"]]
        W_t = [row[:trial_rank] for row in best["W"]]
        if verify(U_t, V_t, W_t):
            best = {"U": U_t, "V": V_t, "W": W_t}
            best_rank = trial_rank
            break

    return best

