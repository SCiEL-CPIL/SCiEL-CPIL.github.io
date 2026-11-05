"""Exact rank-81 decomposition of M_{3,4,7} via three disjoint 2x2x2 Strassen blocks."""
DIMENSIONS = [3, 4, 7]
# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Guaranteed correct rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Four disjoint 2x2x2 Strassen blocks:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 3: i in {1,2}, j in {0,1}, k in {4,5}
    # Block 4: i in {1,2}, j in {2,3}, k in {6,7} -- wait, p=7 so k only goes to 6
    # Let me reconsider. p=7, so k in {0,...,6}.
    # Block 4: i in {1,2}, j in {2,3}, k in {5,6} -- but j in {2,3} is already used in block 2
    # Need disjoint U, V, W index sets.
    # U indices: i*m+j. For n=3,m=4: U has 12 rows.
    # V indices: j*p+k. For m=4,p=7: V has 28 rows.
    # W indices: k*n+i. For p=7,n=3: W has 21 rows.
    #
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    #   U: {0,1,4,5}, V: {0,1,7,8}, W: {0,1,3,4}
    # Block 2: i={0,1}, j={2,3}, k={2,3}
    #   U: {2,3,6,7}, V: {14,15,21,22}, W: {6,7,9,10}
    # Block 3: i={1,2}, j={0,1}, k={4,5}
    #   U: {4,5,8,9}, V: {4,5,11,12}, W: {12,13,15,16}
    # Block 4: i={1,2}, j={2,3}, k={5,6}
    #   U: {6,7,10,11}, V: {18,19,25,26}, W: {15,16,18,19}
    #
    # Check U overlap: Block1 uses {0,1,4,5}, Block3 uses {4,5,8,9} -> overlap at {4,5}!
    # So blocks 1 and 3 share U indices. That's a problem for disjointness.
    #
    # Let me reconsider. The key insight from the working solution is that the blocks
    # must have disjoint U, V, AND W index sets.
    #
    # Block 1: i={0,1}, j={0,1}, k={0,1}: U={0,1,4,5}, V={0,1,7,8}, W={0,1,3,4}
    # Block 2: i={0,1}, j={2,3}, k={2,3}: U={2,3,6,7}, V={14,15,21,22}, W={6,7,9,10}
    # Block 3: i={1,2}, j={0,1}, k={4,5}: U={4,5,8,9}, V={4,5,11,12}, W={12,13,15,16}
    #
    # U overlap between 1 and 3: {4,5}. Not disjoint!
    #
    # But the current program claims rank 81 with three blocks and it passes verification.
    # Let me re-examine. Maybe "disjoint" in the original comment was aspirational or
    # the blocks can share some indices as long as the tensor identity still holds.
    #
    # Actually, looking more carefully: if two blocks share a U row, V row, or W row,
    # the contributions from both blocks add up in that shared coordinate. As long as
    # the total sum equals the correct value for each (i,j,k), it's fine. The Strassen
    # identity is linear, so overlapping blocks just add their contributions.
    #
    # So the real question is: which (i,j,k) triples are covered by which blocks?
    # Each 2x2x2 Strassen block covers exactly 8 (i,j,k) triples with rank 7 instead of 8.
    # If blocks overlap in (i,j,k) space, we get double-counting issues.
    #
    # Block 1 covers: i in {0,1}, j in {0,1}, k in {0,1} -> 8 triples
    # Block 2 covers: i in {0,1}, j in {2,3}, k in {2,3} -> 8 triples
    # Block 3 covers: i in {1,2}, j in {0,1}, k in {4,5} -> 8 triples
    #
    # These (i,j,k) sets are disjoint! No triple is in more than one block.
    # So we can add a 4th block covering another 8 disjoint triples.
    #
    # Remaining triples not covered: total 84 - 24 = 60 triples.
    # Let's find a 2x2x2 block in the remaining.
    # Block 4: i={1,2}, j={2,3}, k={5,6}: 8 triples
    #   Check no overlap with blocks 1,2,3:
    #   Block 1: i={0,1},j={0,1},k={0,1} - no overlap (j differs)
    #   Block 2: i={0,1},j={2,3},k={2,3} - k differs ({5,6} vs {2,3}), no overlap
    #   Block 3: i={1,2},j={0,1},k={4,5} - j differs ({2,3} vs {0,1}), no overlap
    #   Great! Block 4 is disjoint in (i,j,k) space.
    #
    # So 4 blocks: rank = 4*7 + (84 - 4*8) = 28 + 52 = 80.
    # Wait: 84 - 32 = 52 remaining, plus 28 from blocks = 80.
    #
    # Can we add a 5th? i={1,2}, j={2,3}, k={6,7} - but p=7, max k=6. So k={5,6} or {6,?}
    # k must be two consecutive values. k={6,7} invalid. k={5,6} already used.
    # What about i={0,1}, j={2,3}, k={4,5}? 
    #   Block 2: i={0,1},j={2,3},k={2,3} - k differs, OK
    #   Block 3: i={1,2},j={0,1},k={4,5} - j differs, OK
    #   Block 4: i={1,2},j={2,3},k={5,6} - i overlaps at {1}, j overlaps at {2,3}, k overlaps at {5}
    #   Triple (1,2,5) is in both block 3? No, block 3 has j={0,1}. 
    #   Triple (1,2,5): i=1 in {0,1} yes, j=2 in {2,3} yes, k=5 in {4,5} yes -> in block 5
    #   Triple (1,2,5): i=1 in {1,2} yes, j=2 in {2,3} yes, k=5 in {5,6} yes -> in block 4!
    #   OVERLAP! So block 5 at (i={0,1},j={2,3},k={4,5}) overlaps with block 4.
    #
    # Try i={0,1}, j={0,1}, k={2,3}:
    #   Block 1: k={0,1} vs {2,3} - no overlap
    #   Block 2: j={2,3} vs {0,1} - no overlap  
    #   Block 3: i overlaps at {1}, j={0,1} same, k={4,5} vs {2,3} - no k overlap
    #   Block 4: i overlaps at {1}, j={2,3} vs {0,1} - no overlap
    #   No (i,j,k) overlap! 
    #
    # Block 5: i={0,1}, j={0,1}, k={2,3}
    #   5 blocks: rank = 5*7 + (84 - 5*8) = 35 + 44 = 79.
    #
    # Can we add a 6th? i={0,1}, j={0,1}, k={4,5}:
    #   Block 3: i={1,2},j={0,1},k={4,5} - i overlaps at {1}, j same, k same -> OVERLAP!
    #
    # i={0,1}, j={2,3}, k={4,5}: overlaps with block 4 as shown.
    # i={1,2}, j={0,1}, k={2,3}:
    #   Block 1: i overlaps at {1}, j same, k={0,1} vs {2,3} - no overlap
    #   Block 3: i same, j same, k={4,5} vs {2,3} - no overlap
    #   Block 5: i overlaps at {1}, j same, k={2,3} same -> OVERLAP at (1,0,2),(1,1,2),(1,0,3),(1,1,3)
    #
    # i={1,2}, j={2,3}, k={0,1}:
    #   Block 1: i overlaps at {1}, j={0,1} vs {2,3} - no overlap
    #   Block 2: i overlaps at {1}, j same, k same -> OVERLAP!
    #
    # i={1,2}, j={2,3}, k={4,5}:
    #   Block 3: i same, j={0,1} vs {2,3} - no overlap
    #   Block 4: i same, j same, k={5,6} vs {4,5} - k overlaps at {5}
    #   Triple (1,2,5): in block 4 (i=1 in {1,2}, j=2 in {2,3}, k=5 in {5,6}) and in block 6 (i=1 in {1,2}, j=2 in {2,3}, k=5 in {4,5}) -> OVERLAP
    #
    # i={0,1}, j={0,1}, k={6,7}: k=7 invalid.
    # i={0,1}, j={2,3}, k={6,7}: k=7 invalid.
    # i={1,2}, j={0,1}, k={6,7}: k=7 invalid.
    # i={1,2}, j={2,3}, k={6,7}: k=7 invalid.
    #
    # So max 5 disjoint 2x2x2 blocks -> rank 79.
    #
    # Wait, let me double-check block 5 vs block 1 more carefully.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 5: i in {0,1}, j in {0,1}, k in {2,3}
    # (i,j,k) sets: Block1 = {0,1}x{0,1}x{0,1}, Block5 = {0,1}x{0,1}x{2,3}
    # Disjoint in k, so no (i,j,k) overlap. Correct.
    #
    # And block 5 vs block 3:
    # Block 3: i in {1,2}, j in {0,1}, k in {4,5}
    # Block 5: i in {0,1}, j in {0,1}, k in {2,3}
    # No overlap in i (only {1} common) AND no overlap in k. So no (i,j,k) overlap. Correct.
    #
    # Total: 5 blocks, 40 (i,j,k) covered, 44 remaining.
    # Rank = 5*7 + 44 = 79.

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Guaranteed correct rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    rank = 79
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_start, i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        U[i0*m+j0][col_start] = 1
        U[i1*m+j1][col_start] = 1
        V[j0*p+k0][col_start] = 1
        V[j1*p+k1][col_start] = 1
        W[k0*n+i0][col_start] = 1
        W[k1*n+i1][col_start] = 1

        U[i1*m+j0][col_start+1] = 1
        U[i1*m+j1][col_start+1] = 1
        V[j0*p+k0][col_start+1] = 1
        W[k0*n+i1][col_start+1] = 1
        W[k1*n+i1][col_start+1] = -1

        U[i0*m+j0][col_start+2] = 1
        V[j0*p+k1][col_start+2] = 1
        V[j1*p+k1][col_start+2] = -1
        W[k1*n+i0][col_start+2] = 1
        W[k1*n+i1][col_start+2] = 1

        U[i1*m+j1][col_start+3] = 1
        V[j1*p+k0][col_start+3] = 1
        V[j0*p+k0][col_start+3] = -1
        W[k0*n+i0][col_start+3] = 1
        W[k0*n+i1][col_start+3] = 1

        U[i0*m+j0][col_start+4] = 1
        U[i0*m+j1][col_start+4] = 1
        V[j1*p+k1][col_start+4] = 1
        W[k0*n+i0][col_start+4] = -1
        W[k1*n+i0][col_start+4] = 1

        U[i1*m+j0][col_start+5] = 1
        U[i0*m+j0][col_start+5] = -1
        V[j0*p+k0][col_start+5] = 1
        V[j0*p+k1][col_start+5] = 1
        W[k1*n+i1][col_start+5] = 1

        U[i0*m+j1][col_start+6] = 1
        U[i1*m+j1][col_start+6] = -1
        V[j1*p+k0][col_start+6] = 1
        V[j1*p+k1][col_start+6] = 1
        W[k0*n+i0][col_start+6] = 1

        return col_start + 7

    col = add_strassen_block(0, 0, 0, 0)       # i={0,1}, j={0,1}, k={0,1}
    col = add_strassen_block(col, 0, 2, 2)       # i={0,1}, j={2,3}, k={2,3}
    col = add_strassen_block(col, 1, 0, 4)       # i={1,2}, j={0,1}, k={4,5}
    col = add_strassen_block(col, 1, 2, 5)       # i={1,2}, j={2,3}, k={5,6}
    col = add_strassen_block(col, 0, 0, 2)       # i={0,1}, j={0,1}, k={2,3}

    covered = set()
    for i0, j0, k0 in [(0,0,0), (0,2,2), (1,0,4), (1,2,5), (0,0,2)]:
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    if col != rank:
        return _schoolbook(n, m, p)

    if time.time() > deadline:
        return _schoolbook(n, m, p)
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
