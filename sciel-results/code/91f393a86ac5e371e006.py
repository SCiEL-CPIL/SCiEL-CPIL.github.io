import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6

    # Strategy: partition the 3x5x6 multiplication tensor into two blocks:
    # Block 1: rows 0..1 of A (2x5), all columns of B (5x6) -> 2x6 output
    # Block 2: row 2 of A (1x5), all columns of B (5x6) -> 1x6 output
    #
    # For Block 1 (2x5x6): partition the 5 columns of A into 4+1:
    #   - 2x4x6 part: use a rank-17 decomposition
    #   - 2x1x6 part: schoolbook rank 12
    #   Total for Block 1: 17 + 12 = 29
    #
    # For Block 2 (1x5x6): schoolbook rank 30
    #
    # Total rank: 29 + 30 = 59
    #
    # The rank-17 decomposition for 2x4x6 is built as follows:
    #   2x4x6 = 2x(4x6). Split 4x6 into two 4x3 blocks.
    #   Each 2x4x3 can be decomposed using Strassen-like techniques.
    #   R(2,4,3) <= 11 (known bound), so 2*11 = 22. That's not 17.
    #
    # Let me try a different approach. Use the known result R(2,4,4) = 15?
    # Actually, let's just use a clean schoolbook decomposition but with
    # Strassen for the 2x2x2 sub-blocks where possible.
    #
    # Simpler approach: Use Strassen (rank 7) for 2x2x2 blocks.
    # 3x5x6: pad to 4x8x8? No, that's complex.
    #
    # Let's use a direct approach:
    # - For the 3x5x6 tensor, use schoolbook but replace any 2x2x2 sub-block
    #   with Strassen's rank-7 decomposition.
    #
    # Actually, the cleanest improvement: use the known R(2,2,2)=7 (Strassen)
    # to reduce the rank. Partition 3x5x6 into blocks where 2x2x2 appears.
    #
    # Partition i in {0,1,2}, j in {0..4}, k in {0..5}:
    # - i={0,1}, j={0,1}, k={0,1}: 2x2x2 -> rank 7 (Strassen)
    # - i={0,1}, j={2,3}, k={0,1}: 2x2x2 -> rank 7
    # - i={0,1}, j={0,1}, k={2,3}: 2x2x2 -> rank 7
    # - i={0,1}, j={2,3}, k={2,3}: 2x2x2 -> rank 7
    # - i={0,1}, j={4}, k={0..5}: 2x1x6 -> rank 12
    # - i={0,1}, j={0..4}, k={4,5}: 2x5x2 -> rank 20
    # - i={2}, j={0..4}, k={0..5}: 1x5x6 -> rank 30
    #
    # Total: 4*7 + 12 + 20 + 30 = 28 + 62 = 90. Same as schoolbook.
    #
    # The issue is that the non-2x2x2 blocks dominate. Let me try a different
    # partition. Use Strassen for 2x2x2 and handle the rest optimally.
    #
    # Better: R(2,2,3) = 11. Use this for 2x2x3 blocks.
    # Partition: i={0,1}, j={0,1}, k={0,1,2}: 2x2x3 -> rank 11
    #            i={0,1}, j={0,1}, k={3,4,5}: 2x2x3 -> rank 11
    #            i={0,1}, j={2,3}, k={0..5}: 2x2x6 -> rank 24 (schoolbook)
    #            i={0,1}, j={4}, k={0..5}: 2x1x6 -> rank 12
    #            i={2}, j={0..4}, k={0..5}: 1x5x6 -> rank 30
    # Total: 11+11+24+12+30 = 88. Slight improvement.
    #
    # Let me try harder. R(2,3,3) = 16.
    # i={0,1}, j={0,1,2}, k={0,1,2}: 2x3x3 -> rank 16
    # i={0,1}, j={0,1,2}, k={3,4,5}: 2x3x3 -> rank 16
    # i={0,1}, j={3,4}, k={0..5}: 2x2x6 -> rank 24
    # i={2}, j={0..4}, k={0..5}: 1x5x6 -> rank 30
    # Total: 16+16+24+30 = 86.
    #
    # R(2,4,4) <= 15 (I believe this is correct).
    # i={0,1}, j={0..3}, k={0..3}: 2x4x4 -> rank 15
    # i={0,1}, j={0..3}, k={4,5}: 2x4x2 -> rank 16
    # i={0,1}, j={4}, k={0..5}: 2x1x6 -> rank 12
    # i={2}, j={0..4}, k={0..5}: 1x5x6 -> rank 30
    # Total: 15+16+12+30 = 73.
    #
    # I'm not confident about R(2,4,4)=15. Let me use a safer bound.
    # R(2,4,4) <= 2*R(2,2,4) = 2*12 = 24? Or 2*R(2,4,2)=2*16=32?
    # Actually R(2,2,4) <= 2*R(2,2,2) = 14. So R(2,4,4) <= 28.
    # That gives 28+16+12+30 = 86.
    #
    # Given the uncertainty, let me implement a clean schoolbook with
    # Strassen for 2x2x2 blocks where they fit cleanly.
    #
    # Final decision: implement schoolbook (rank 90) cleanly to ensure
    # correctness. The structural change from the seed (which had broken
    # code) should make this a valid, distinct candidate.

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
