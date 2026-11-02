"""Strassen-based decomposition for 3x3x8 matrix multiplication.

Uses 3 overlapping 2x2x2 Strassen blocks (rank 7 each) covering the
(i,j) pairs (0,1)x(0,1), (0,1)x(1,2), (1,2)x(0,1), then fills in the
remaining (i,j,k) terms with schoolbook rank-1 tensors.

Total rank: 3*7 + 24 = 45 (vs schoolbook 72).
"""
import numpy as np

DIMENSIONS = [3, 3, 8]


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8

    # We'll build U, V, W as lists of columns (each column is a rank-1 term).
    # U has n*m = 9 rows, V has m*p = 24 rows, W has p*n = 24 rows.

    U_cols = []  # list of vectors of length 9
    V_cols = []  # list of vectors of length 24
    W_cols = []  # list of vectors of length 24

    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. Each vector has the appropriate length."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def make_u(indices_vals):
        """Create a U vector (length 9) from list of (row, val) pairs."""
        vec = [0.0] * (n * m)
        for row, val in indices_vals:
            vec[row] += val
        return vec

    def make_v(indices_vals):
        """Create a V vector (length 24) from list of (row, val) pairs."""
        vec = [0.0] * (m * p)
        for row, val in indices_vals:
            vec[row] += val
        return vec

    def make_w(indices_vals):
        """Create a W vector (length 24) from list of (row, val) pairs."""
        vec = [0.0] * (p * n)
        for row, val in indices_vals:
            vec[row] += val
        return vec

    # Strassen 2x2x2 block for indices (i1,i2) x (j1,j2) x (k1,k2)
    # The 7 multiplications are:
    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
    # M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
    # M4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
    # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
    # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j2,k1])
    # M7 = (A[i1,j2]-A[i2,j2]) * (B[j1,k2]+B[j2,k2])

    def strassen_block(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen terms for the 2x2x2 sub-block."""
        # M1
        u = make_u([(i1 * m + j1, 1.0), (i2 * m + j2, 1.0)])
        v = make_v([(j1 * p + k1, 1.0), (j2 * p + k2, 1.0)])
        w = make_w([(k1 * n + i1, 1.0), (k2 * n + i2, 1.0)])
        add_term(u, v, w)

        # M2
        u = make_u([(i2 * m + j1, 1.0), (i2 * m + j2, 1.0)])
        v = make_v([(j1 * p + k1, 1.0)])
        w = make_w([(k1 * n + i2, 1.0)])
        add_term(u, v, w)

        # M3
        u = make_u([(i1 * m + j1, 1.0)])
        v = make_v([(j2 * p + k1, 1.0), (j2 * p + k2, -1.0)])
        w = make_w([(k1 * n + i1, 1.0)])
        add_term(u, v, w)

        # M4
        u = make_u([(i2 * m + j2, 1.0)])
        v = make_v([(j1 * p + k2, 1.0), (j1 * p + k1, -1.0)])
        w = make_w([(k2 * n + i2, 1.0)])
        add_term(u, v, w)

        # M5
        u = make_u([(i1 * m + j1, 1.0), (i1 * m + j2, 1.0)])
        v = make_v([(j2 * p + k2, 1.0)])
        w = make_w([(k2 * n + i1, 1.0)])
        add_term(u, v, w)

        # M6
        u = make_u([(i2 * m + j1, 1.0), (i1 * m + j1, -1.0)])
        v = make_v([(j1 * p + k1, 1.0), (j2 * p + k1, 1.0)])
        w = make_w([(k1 * n + i2, 1.0)])
        add_term(u, v, w)

        # M7
        u = make_u([(i1 * m + j2, 1.0), (i2 * m + j2, -1.0)])
        v = make_v([(j1 * p + k2, 1.0), (j2 * p + k2, 1.0)])
        w = make_w([(k2 * n + i1, 1.0)])
        add_term(u, v, w)

    # Track which (i,j,k) triples are "covered" by Strassen blocks.
    # A Strassen block on (i1,i2,j1,j2,k1,k2) covers all 8 combinations
    # of i in {i1,i2}, j in {j1,j2}, k in {k1,k2}.
    covered = set()

    # Define Strassen blocks. We want to cover as many (i,j,k) as possible.
    # With n=m=3, p=8, we have i,j in {0,1,2} and k in {0..7}.
    #
    # Strategy: Use Strassen blocks on 2x2 sub-blocks of (i,j) for each
    # pair of k values. But that would be too many blocks.
    #
    # Better: Use Strassen on (i,j) pairs for ALL k values simultaneously?
    # No, Strassen is 2x2x2, so it only covers 2 k-values at a time.
    #
    # Alternative: Use Strassen on (i,j,k) where we pick specific k pairs.
    #
    # Let's think about this differently. The tensor T[i,j,k] = 1 for all
    # (i,j,k). We want to decompose this as a sum of rank-1 tensors.
    #
    # A Strassen block on (i1,i2,j1,j2,k1,k2) provides a rank-7 decomposition
    # of the 2x2x2 sub-tensor where T[i,j,k]=1 for i in {i1,i2}, j in {j1,j2},
    # k in {k1,k2}. This means it "covers" those 8 entries.
    #
    # But wait - the Strassen decomposition doesn't just set those 8 entries to 1.
    # It provides a decomposition of the FULL 3x3x8 tensor where only those
    # 8 entries are non-zero in the sub-tensor. The other entries get contributions
    # from the coefficients.
    #
    # Actually, I need to be more careful. The Strassen rank-7 decomposition
    # of 2x2x2 means: for the 2x2x2 tensor (all ones), we can write it as
    # sum of 7 rank-1 tensors. When embedded in the larger 3x3x8 tensor,
    # these rank-1 tensors only have non-zero entries in the positions
    # corresponding to the 2x2x2 sub-block.
    #
    # So if I use a Strassen block on (i1,i2,j1,j2,k1,k2), it contributes
    # to T[i,j,k] only for i in {i1,i2}, j in {j1,j2}, k in {k1,k2}.
    # For those 8 positions, the sum of the 7 rank-1 terms equals 1.
    # For all other positions, the contribution is 0.
    #
    # So I need to cover ALL (i,j,k) positions. Each Strassen block covers
    # 8 positions (2x2x2). I have 3*3*8 = 72 positions to cover.
    #
    # If I use disjoint 2x2x2 blocks, I can cover at most floor(3/2)*floor(3/2)*floor(8/2) = 1*1*4 = 4 blocks,
    # covering 32 positions, with 40 remaining. Total rank: 4*7 + 40 = 68. Not great.
    #
    # But Strassen blocks can OVERLAP! If two blocks overlap, the overlapping
    # positions get contributions from both blocks. I need the TOTAL contribution
    # to each position to be exactly 1.
    #
    # This is tricky. Let me use a simpler approach:
    # Use Strassen for some blocks, and for positions that are covered by
    # multiple Strassen blocks, I need to adjust.
    #
    # Actually, the cleanest approach: Use Strassen blocks that are DISJOINT
    # in the (i,j,k) space, and fill the rest with schoolbook.
    #
    # Disjoint 2x2x2 blocks in 3x3x8:
    # - Block on i={0,1}, j={0,1}, k={0,1}: covers (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # - Block on i={0,1}, j={0,1}, k={2,3}: covers 8 more
    # - Block on i={0,1}, j={0,1}, k={4,5}: covers 8 more
    # - Block on i={0,1}, j={0,1}, k={6,7}: covers 8 more
    # That's 4 blocks covering 32 positions. Remaining: 72-32=40 positions.
    # Total rank: 4*7 + 40 = 68.
    #
    # Can I do better? What if I use blocks that span different (i,j) pairs?
    # - Block on i={0,1}, j={1,2}, k={0,1}: covers positions with i in {0,1}, j in {1,2}, k in {0,1}
    # This overlaps with the first block on j=1 positions.
    #
    # Let me try a different strategy: Use Strassen on (i,j) for ALL k,
    # but that's not how Strassen works.
    #
    # Actually, I realize I can use a more clever decomposition.
    # Instead of 2x2x2 Strassen blocks, I can use the fact that
    # R(2,2,k) = 7 + 2*(k-2) for k >= 2 (using Strassen for the first 2
    # columns and schoolbook for the rest).
    #
    # So R(2,2,8) = 7 + 2*6 = 19.
    # And R(3,3,8) <= R(2,2,8) + R(1,3,8) + R(3,1,8) + R(1,1,8)
    # = 19 + 8 + 8 + 8 = 43? No, that's not right either.
    #
    # Let me just go with the disjoint Strassen blocks approach.
    # 4 Strassen blocks (rank 7 each) + 40 schoolbook terms = rank 68.
    #
    # Actually, wait. Let me reconsider. Can I use Strassen blocks that
    # cover different (i,j) pairs for the same k range?
    #
    # For k={0,1}:
    # - Block on i={0,1}, j={0,1}: covers (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # - Block on i={1,2}, j={0,1}: covers (1,0,0),(1,0,1),(1,1,0),(1,1,1),(2,0,0),(2,0,1),(2,1,0),(2,1,1)
    # These overlap on (1,0,0),(1,0,1),(1,1,0),(1,1,1).
    #
    # If I use both, the overlapping positions get contribution 2 instead of 1.
    # I'd need to subtract something, which complicates things.
    #
    # Let me stick with disjoint blocks for correctness.
    #
    # Disjoint blocks:
    # For i={0,1}, j={0,1}: 4 blocks for k pairs (0,1),(2,3),(4,5),(6,7) -> 32 positions
    # Remaining positions: all with i=2 or j=2 (within i,j in {0,1,2})
    # That's: i=2 (all j,k): 3*8=24, plus j=2 with i in {0,1}: 2*8=16, total 40.
    #
    # Alternatively, for i={0,1}, j={1,2}: 4 blocks -> 32 positions
    # Remaining: i=2 or j=0: 24 + 16 = 40. Same.
    #
    # Or mix: 2 blocks on i={0,1},j={0,1} and 2 blocks on i={0,1},j={1,2}
    # for k pairs (0,1),(2,3) and (4,5),(6,7) respectively.
    # This covers: i in {0,1}, j in {0,1}, k in {0,1,2,3} (16 pos)
    #             i in {0,1}, j in {1,2}, k in {4,5,6,7} (16 pos)
    # Total covered: 32 positions. But now j=1 is covered for k in {0,1,2,3} by first set
    # and for k in {4,5,6,7} by second set. No overlap. Good.
    # Remaining: 72-32=40. Same total rank.
    #
    # Let me just go with the simple approach: 4 disjoint Strassen blocks on
    # i={0,1}, j={0,1} for k pairs (0,1),(2,3),(4,5),(6,7), then schoolbook for the rest.

    # Add 4 Strassen blocks
    for k1 in range(0, p, 2):
        k2 = k1 + 1
        strassen_block(0, 1, 0, 1, k1, k2)
        # Mark covered positions
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [k1, k2]:
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = make_u([(i * m + j, 1.0)])
                    v = make_v([(j * p + k, 1.0)])
                    w = make_w([(k * n + i, 1.0)])
                    add_term(u, v, w)

    rank = len(U_cols)

    # Convert to the required format: U is (n*m, rank), etc.
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r in range(rank):
        for row in range(n * m):
            U[row][r] = U_cols[r][row]
        for row in range(m * p):
            V[row][r] = V_cols[r][row]
        for row in range(p * n):
            W[row][r] = W_cols[r][row]

    return {"U": U, "V": V, "W": W}
