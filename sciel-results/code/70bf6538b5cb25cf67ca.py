"""Exact rank-78 decomposition of M_{3,4,7} via 6 Strassen blocks (i={0,1}) + 36 schoolbook."""
DIMENSIONS = [3, 4, 7]
# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
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

    # Design: 6 Strassen blocks (rank 42) covering i in {0,1}, all j in {0..3}, k in {0..5}.
    # That's 2*4*6 = 48 entries covered. Remaining 36 entries get schoolbook terms.
    # Total rank = 42 + 36 = 78.
    #
    # We can also try: 3 blocks i={0,1} j={0,1} + 3 blocks i={1,2} j={2,3}
    # This covers: i=0 j={0,1} k={0..5} (12), i=1 j={0,1} k={0..5} (12),
    #              i=1 j={2,3} k={0..5} (12), i=2 j={2,3} k={0..5} (12) = 48 entries.
    # Same coverage, same rank. No improvement.
    #
    # Try 8 blocks: 4 with i={0,1} and 4 with i={1,2}, on disjoint (j,k) sub-cubes.
    # i={0,1}: j={0,1} k={0,1},{2,3} (2 blocks, rank 14, covers 16)
    # i={0,1}: j={2,3} k={0,1},{2,3} (2 blocks, rank 14, covers 16)
    # i={1,2}: j={0,1} k={4,5} (1 block, rank 7, covers 8)
    # i={1,2}: j={2,3} k={4,5} (1 block, rank 7, covers 8)
    # Total: 6 blocks, rank 42, covers 48. Same.
    #
    # What about 7 blocks? 3 with i={0,1} j={0,1} k={0,1},{2,3},{4,5} (rank 21, covers 24)
    # 3 with i={1,2} j={2,3} k={0,1},{2,3},{4,5} (rank 21, covers 24)
    # 1 extra block i={0,1} j={2,3} k={6,6}? No, need k0 and k0+1 distinct.
    # Can't do a 2x2x2 block with k={6,6}.
    #
    # So 78 is the floor for this family. Let's just build it correctly.

    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_start, i0, j0, k0):
        """Add a 7-term Strassen block for sub-cube [i0:i0+2, j0:j0+2, k0:k0+2]."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        c = col_start

        # t0: A00*B00 + A11*B11 -> C00 += 1, C11 += 1
        U[i0*m+j0][c] = 1
        U[i1*m+j1][c] = 1
        V[j0*p+k0][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = 1
        W[k1*n+i1][c] = 1

        # t1: (A10+A11)*(B00+B10) -> C10 += 1, C11 += -1
        U[i1*m+j0][c+1] = 1
        U[i1*m+j1][c+1] = 1
        V[j0*p+k0][c+1] = 1
        V[j1*p+k0][c+1] = 1
        W[k0*n+i1][c+1] = 1
        W[k1*n+i1][c+1] = -1

        # t2: A00*(B01-B11) -> C01 += 1, C11 += 1
        U[i0*m+j0][c+2] = 1
        V[j0*p+k1][c+2] = 1
        V[j1*p+k1][c+2] = -1
        W[k1*n+i0][c+2] = 1
        W[k1*n+i1][c+2] = 1

        # t3: A11*(B10-B00) -> C10 += 1, C00 += 1
        U[i1*m+j1][c+3] = 1
        V[j1*p+k0][c+3] = 1
        V[j0*p+k0][c+3] = -1
        W[k0*n+i0][c+3] = 1
        W[k0*n+i1][c+3] = 1

        # t4: (A00+A01)*B11 -> C11 += -1, C10 += 1
        U[i0*m+j0][c+4] = 1
        U[i0*m+j1][c+4] = 1
        V[j1*p+k1][c+4] = 1
        W[k0*n+i0][c+4] = -1
        W[k1*n+i0][c+4] = 1

        # t5: (A10-A00)*(B00+B01) -> C00 += 1, C01 += 1
        U[i1*m+j0][c+5] = 1
        U[i0*m+j0][c+5] = -1
        V[j0*p+k0][c+5] = 1
        V[j0*p+k1][c+5] = 1
        W[k0*n+i1][c+5] = 1
        W[k1*n+i1][c+5] = 1

        # t6: (A01-A11)*(B10+B11) -> C00 += 1
        U[i0*m+j1][c+6] = 1
        U[i1*m+j1][c+6] = -1
        V[j1*p+k0][c+6] = 1
        V[j1*p+k1][c+6] = 1
        W[k0*n+i0][c+6] = 1

        return c + 7

    # 6 Strassen blocks: i={0,1}, j-pairs {0,1} and {2,3}, k-pairs {0,1},{2,3},{4,5}
    col = add_strassen_block(0, 0, 0, 0)
    col = add_strassen_block(col, 0, 0, 2)
    col = add_strassen_block(col, 0, 0, 4)
    col = add_strassen_block(col, 0, 2, 0)
    col = add_strassen_block(col, 0, 2, 2)
    col = add_strassen_block(col, 0, 2, 4)

    # Track covered (i,j,k) indices
    covered = set()
    for i0, j0, k0 in [(0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4)]:
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))

    # Schoolbook for remaining 36 entries
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
