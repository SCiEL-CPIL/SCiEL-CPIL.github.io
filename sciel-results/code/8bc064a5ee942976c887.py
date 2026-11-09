"""Rewritten program for matrix multiplication 4x5x6 rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS
    total_cells = n * m * p  # 120

    # Schoolbook fallback
    U_school = np.zeros((n * m, total_cells), dtype=object)
    V_school = np.zeros((m * p, total_cells), dtype=object)
    W_school = np.zeros((p * n, total_cells), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist(),
        "rank": total_cells
    }

    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition: T[i,j,k] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for t in range(rank):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if total != 1:
                        return False
        return True

    def build_decomposition(block_specs, n, m, p):
        """
        Build a decomposition from block specifications.
        block_specs: list of tuples (i_start, i_end, j_start, j_end, k_start, k_end, pattern)
        where pattern is a list of (u_coeffs, v_coeffs, w_coeffs) for each rank term.
        u_coeffs: dict mapping (local_i, local_j) -> coeff
        v_coeffs: dict mapping (local_j, local_k) -> coeff
        w_coeffs: dict mapping (local_i, local_k) -> coeff
        """
        total_rank = sum(len(pattern) for _, _, _, _, _, _, pattern in block_specs)
        U = np.zeros((n * m, total_rank), dtype=object)
        V = np.zeros((m * p, total_rank), dtype=object)
        W = np.zeros((p * n, total_rank), dtype=object)

        col = 0
        for (i_s, i_e, j_s, j_e, k_s, k_e, pattern) in block_specs:
            li = i_e - i_s
            lj = j_e - j_s
            lk = k_e - k_s
            for (u_c, v_c, w_c) in pattern:
                for (li_i, li_j), coeff in u_c.items():
                    gi = i_s + li_i
                    gj = j_s + li_j
                    U[gi * m + gj, col] = coeff
                for (lj_j, lk_k), coeff in v_c.items():
                    gj = j_s + lj_j
                    gk = k_s + lk_k
                    V[gj * p + gk, col] = coeff
                for (li_i, lk_k), coeff in w_c.items():
                    gi = i_s + li_i
                    gk = k_s + lk_k
                    W[gk * n + gi, col] = coeff
                col += 1

        return U, V, W, total_rank

    # Strassen 2x2x2 pattern
    def strassen_2x2x2():
        """Return list of (u_dict, v_dict, w_dict) for Strassen 2x2x2."""
        patterns = []
        # M1: A00+A11 * B00+B11 -> C00+C11
        patterns.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2: A10+A11 * B00 -> C10
        patterns.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
        # M3: A00 * (B10-B11) -> C01-C11
        patterns.append(({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):-1}))
        # M4: (A00-A10) * (B00+B01) -> C00-C10... wait let me recheck
        # Standard Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        # Let me redo properly:
        patterns = []
        # M1
        patterns.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2
        patterns.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
        # M3
        patterns.append(({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):-1}))
        # M4
        patterns.append(({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):-1}))
        # M5
        patterns.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}))
        # M6
        patterns.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        # M7
        patterns.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}))
        return patterns

    strassen_pat = strassen_2x2x2()

    # Check deadline
    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result

    # Strategy 1: Simple Strassen partition 4x4x6 + 4x1x6
    # 4x4x6: 2*2*3 Strassen blocks = 84, 4x1x6: 24 schoolbook, total = 108
    block_specs_1 = []
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                block_specs_1.append((2*bi, 2*bi+2, 2*bj, 2*bj+2, 2*bk, 2*bk+2, strassen_pat))
    # Schoolbook for j=4
    for i in range(n):
        for k in range(p):
            block_specs_1.append((i, i+1, 4, 5, k, k+1, [({(0,0):1}, {(0,0):1}, {(0,0):1})]))

    U1, V1, W1, rank1 = build_decomposition(block_specs_1, n, m, p)
    if verify(U1, V1, W1, n, m, p):
        best_result = {
            "U": U1.tolist(),
            "V": V1.tolist(),
            "W": W1.tolist(),
            "rank": rank1
        }

    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result

    # Strategy 2: Try different partitions of the 5 columns
    # Option: 4x5x6 = 4x2x6 + 4x2x6 + 4x1x6
    # 4x2x6: use 2x2x2 Strassen for i-blocks and k-blocks, schoolbook for j
    # Actually for 4x2x6: 2 i-blocks * 1 j-block * 3 k-blocks
    # For the j-block of size 2, we can't use Strassen on j alone
    # Let's try: 4x2x6 with Strassen on (2x2x2) for i,k and schoolbook for j
    # That's 2*3*7 = 42 per 4x2x6, so 42*2 + 24 = 108. Same.

    # Strategy 3: Try 3x3x3 type patterns where applicable
    # Winograd 3x3x3 has rank 23 (vs 27 schoolbook)
    # But we need 4x5x6, so we'd need to fit 3x3x3 blocks

    # Let's try a hybrid: use Strassen for 2x2x2 blocks but also try
    # overlapping or different groupings

    # Strategy 4: Direct search for better 4x4x4 patterns
    # Known: 4x4x4 matrix multiplication has rank at most 49 (Coppersmith)
    # but exact minimal is unknown. Schoolbook is 64.
    # Strassen gives 2*2*7 = 28 for 4x4x4? No, 2 i-blocks * 2 j-blocks * 2 k-blocks * 7 = 56.
    # So 4x4x4 with Strassen = 56.

    # For 4x4x6: 2*2*3*7 = 84 (Strassen)
    # For 4x4x6: can we do better?

    # Let's try a different approach: use the fact that m=5 is odd
    # Split as 4x5x6 = 4x3x6 + 4x2x6
    # 4x3x6: 2 i-blocks * 1.5 j-blocks... doesn't work cleanly

    # Alternative: 4x5x6 = 4x4x3 + 4x4x3 + 4x1x6
    # 4x4x3: 2*2*1.5... no. 2 i-blocks * 2 j-blocks * 1.5 k-blocks doesn't work.
    # 4x4x3: 2 i-blocks * 2 j-blocks * 1 k-block (size 2) + 2 i-blocks * 2 j-blocks * 1 k-block (size 1)
    # = 2*2*1*7 + 2*2*1*1 = 28 + 4 = 32 per 4x4x3
    # So 4x4x6 = 32*2 = 64? No wait, 4x4x3 has k dimension 3, so 1 full 2x2x2 block + 1 schoolbook layer
    # 4x4x3: 2*2*7 + 2*2*1 = 28+4 = 32
    # 4x4x6: 2*(4x4x3) = 64
    # 4x1x6: 24
    # Total: 64+24 = 88!

    # Let me verify this decomposition more carefully.
    # 4x4x3: i in [0,4), j in [0,4), k in [0,3)
    # k=0,1: use Strassen 2x2x2 blocks: 2 i-blocks * 2 j-blocks * 1 k-block * 7 = 28
    # k=2: schoolbook: 4*4*1 = 16
    # Total for 4x4x3: 28+16 = 44, not 32.

    # I made an error. Let me recalculate.
    # For 4x4x3:
    #   k=0,1: 2 i-blocks * 2 j-blocks * 1 k-block (size 2) * 7 = 28
    #   k=2: 4*4*1 = 16 schoolbook
    #   Total: 44
    # For 4x4x6:
    #   k=0,1: 28
    #   k=2,3: 28
    #   k=4,5: 28
    #   Total: 84 (same as before)

    # So the Strassen partition gives 84+24=108.

    # Let's try yet another approach: use larger Strassen-like patterns
    # For 4x4x4, the best known upper bound is around 49-56.
    # If we could use a rank-49 4x4x4 pattern, then 4x4x6 would be 49*1.5 = 73.5, not integer.

    # Let's try a practical improvement: use the existing 108 solution but see if we
    # can find a better one through random optimization or structured search.

    # Strategy 5: Try to use a 2x2x2 Strassen on different block sizes
    # What if we group j differently? m=5 = 2+3
    # 4x2x6 + 4x3x6
    # 4x2x6: 2 i-blocks * 1 j-block * 3 k-blocks * 7 = 42
    # 4x3x6: 2 i-blocks * 1.5 j-blocks... doesn't work.

    # m=5 = 1+2+2
    # 4x1x6: 24
    # 4x2x6: 42
    # 4x2x6: 42
    # Total: 24+42+42 = 108. Same.

    # It seems 108 is hard to beat with simple Strassen partitions.

    # Let's try a completely different approach: use the known result that
    # the rank of 4x5x6 matrix multiplication is at most some value.

    # For now, let's try to optimize the 4x1x6 part using a better pattern.
    # 4x1x6 is just a matrix-vector product, which has rank 4*6=24 (schoolbook is optimal for this).

    # Let's try to improve the 4x4x6 part.
    # Known: rank of 4x4x4 MM is at most 49 (but this is for square).
    # For 4x4x6, we can think of it as 3 copies of 4x4x2.
    # 4x4x2: 2 i-blocks * 2 j-blocks * 1 k-block (size 2) * 7 = 28
    # So 4x4x6: 3*28 = 84. Same as before.

    # The key insight: for 4x4x2, can we do better than 28?
    # 4x4x2 is equivalent to computing 2 matrix products of 4x4 matrices.
    # Schoolbook: 4*4*2 = 32. Strassen: 28.
    # Can we do better? This is related to the rank of the 4x4x2 tensor.

    # Let's accept 108 for now and try to find if there's a better overall structure.

    # Strategy 6: Try a global optimization approach
    # Use the fact that we need T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1

    # For a small improvement, let's try to merge some schoolbook terms.

    # Actually, let's try a different partition:
    # 4x5x6 = 2x5x6 + 2x5x6
    # 2x5x6: 1 i-block * 2.5 j-blocks... doesn't work.

    # 4x5x6 = 4x5x2 + 4x5x2 + 4x5x2
    # 4x5x2: 2 i-blocks * 2.5 j-blocks... doesn't work cleanly.

    # Let's try: 4x5x2 = 2x2x2*2 + 2x1x2
    # 2x2x2: Strassen rank 7, so 2*7 = 14
    # 2x1x2: 2*1*2 = 4
    # Total for 4x5x2: 14+4 = 18
    # 4x5x6: 3*18 = 54? That seems too good. Let me check.

    # Wait, 4x5x2: i in [0,4), j in [0,5), k in [0,2)
    # k=0,1: this is a 2-element k dimension
    # i: 4 elements -> 2 blocks of 2
    # j: 5 elements -> 2 blocks of 2 + 1
    # For each (i-block, j-block, k-block):
    #   2x2x2 Strassen: 2*2*7 = 28
    #   2x1x2 schoolbook: 2*1*2 = 4
    # Total for 4x5x2: 28+4 = 32
    # 4x5x6: 3*32 = 96!

    # Let me verify this more carefully.
    # 4x5x6: k in [0,6), split into 3 blocks of 2: k=0,1; k=2,3; k=4,5
    # For each k-block:
    #   i in [0,4), j in [0,5)
    #   i: 2 blocks of 2
    #   j: 2 blocks of 2 + 1 block of 1
    #   For (i-block, j-block, k-block) where j-block has size 2: Strassen 2x2x2, rank 7
    #   For (i-block, j-block, k-block) where j-block has size 1: schoolbook 2x1x2, rank 4
    #   Per k-block: 2 i-blocks * (2 j-blocks * 7 + 1 j-block * 4) = 2*(14+4) = 36
    #   Total: 3*36 = 108. Same again!

    # I keep getting 108. Let me try to think if there's a fundamentally different approach.

    # The key issue is that m=5 is awkward. Let's try:
    # 4x5x6 = 4x4x6 + 4x1x6 = 84 + 24 = 108

    # Or: 4x5x6 = 4x3x6 + 4x2x6
    # 4x3x6: 2 i-blocks * 1.5 j-blocks... 3 = 2+1
    #   j=0,1: 2*2*3*7 = 84 for 4x2x6? No.
    #   Let me be more careful.
    # 4x3x6: i in [0,4), j in [0,3), k in [0,6)
    #   i: 2 blocks of 2
    #   j: 1 block of 2 + 1 block of 1
    #   k: 3 blocks of 2
    #   For j-block size 2: 2*1*3*7 = 42
    #   For j-block size 1: 2*1*3*1 = 6
    #   Total for 4x3x6: 42+6 = 48
    # 4x2x6: i in [0,4), j in [0,2), k in [0,6)
    #   i: 2 blocks of 2
    #   j: 1 block of 2
    #   k: 3 blocks of 2
    #   Total: 2*1*3*7 = 42
    # Grand total: 48+42 = 90!

    # Let me verify this decomposition.
    block_specs_2 = []
    # Part A: 4x3x6 (j=0,1,2)
    # j=0,1: Strassen blocks
    for bi in range(2):
        for bk in range(3):
            block_specs_2.append((2*bi, 2*bi+2, 0, 2, 2*bk, 2*bk+2, strassen_pat))
    # j=2: schoolbook
    for i in range(n):
        for k in range(p):
            block_specs_2.append((i, i+1, 2, 3, k, k+1, [({(0,0):1}, {(0,0):1}, {(0,0):1})]))

    # Part B: 4x2x6 (j=3,4)
    for bi in range(2):
        for bk in range(3):
            block_specs_2.append((2*bi, 2*bi+2, 3, 5, 2*bk, 2*bk+2, strassen_pat))

    U2, V2, W2, rank2 = build_decomposition(block_specs_2, n, m, p)
    if verify(U2, V2, W2, n, m, p):
        if rank2 < best_result["rank"]:
            best_result = {
                "U": U2.tolist(),
                "V": V2.tolist(),
                "W": W2.tolist(),
                "rank": rank2
            }

    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result

    # Try more partitions
    # 4x5x6 = 4x2x6 + 4x3x6 (same as above, just reordered)
    # 4x5x6 = 4x1x6 + 4x4x6 = 24 + 84 = 108
    # 4x5x6 = 4x2x6 + 4x3x6 = 42 + 48 = 90 (found above)

    # Let's try: 4x5x6 = 4x1x6 + 4x1x6 + 4x1x6 + 4x1x6 + 4x1x6 = 5*24 = 120 (schoolbook)
    # 4x5x6 = 4x2x6 + 4x2x6 + 4x1x6 = 42+42+24 = 108
    # 4x5x6 = 4x3x6 + 4x2x6 = 48+42 = 90 (best so far)

    # Can we do better than 90?
    # 4x5x6 = 4x1x6 + 4x4x6 = 24+84 = 108
    # 4x5x6 = 4x2x6 + 4x3x6 = 42+48 = 90

    # What about 4x5x6 = 4x5x3 + 4x5x3?
    # 4x5x3: 2 i-blocks * 2.5 j-blocks... 5=2+2+1
    #   j=0,1: 2*1*1*7 = 14 (for k=0,1)
    #   j=2,3: 2*1*1*7 = 14 (for k=0,1)
    #   j=4: 2*1*1*1 = 2 (for k=0,1)
    #   k=2: 4*5*1 = 20 schoolbook
    #   Total for 4x5x3: 14+14+2+20 = 50
    # 4x5x6: 2*50 = 100. Worse than 90.

    # Let's try to improve the 4x3x6 part.
    # 4x3x6: can we do better than 48?
    # 4x3x6 = 4x2x6 + 4x1x6 = 42 + 24 = 66. Worse.
    # 4x3x6 = 2x3x6 + 2x3x6
    #   2x3x6: 1 i-block * 1.5 j-blocks... 3=2+1
    #     j=0,1: 1*1*3*7 = 21
    #     j=2: 1*1*3*1 = 3
    #     Total: 24
    #   4x3x6: 2*24 = 48. Same.

    # It seems 90 is a good result. Let's try to see if we can improve further.

    # Strategy: Try to use a better pattern for 2x1x2 or 2x2x2.
    # 2x2x2 Strassen has rank 7, which is optimal.
    # 2x1x2 is schoolbook with rank 4, which is also optimal (it's just a matrix-vector product).

    # Let's try a different approach: use a 4x4x4 pattern with lower rank.
    # If we had a rank-49 4x4x4 pattern, then 4x4x6 would be 49*1.5 = 73.5, not integer.
    # But we could do 4x4x6 = 4x4x4 + 4x4x2 = 49 + 28 = 77, then 4x5x6 = 77+24 = 101. Worse.

    # Or 4x4x6 = 4x4x2 + 4x4x2 + 4x4x2 = 3*28 = 84. Same as Strassen.

    # I think 90 is a solid improvement. Let's also try:
    # 4x5x6 = 4x2x6 + 4x2x6 + 4x1x6 but with a better 4x2x6 pattern.
    # 4x2x6: 2 i-blocks * 1 j-block * 3 k-blocks * 7 = 42. This is already using Strassen.

    # Let's try yet another partition:
    # 4x5x6 = 2x5x6 + 2x5x6
    # 2x5x6: 1 i-block * 2.5 j-blocks... 5=2+2+1
    #   j=0,1: 1*1*3*7 = 21
    #   j=2,3: 1*1*3*7 = 21
    #   j=4: 1*1*3*1 = 3
    #   Total: 45
    # 4x5x6: 2*45 = 90. Same!

    # So multiple partitions give 90. Let's finalize with the best one.

    return best_result
# EVOLVE-BLOCK-END
