import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    # Start with 4 non-overlapping Strassen blocks (rank 71)
    # Block positions: (i0, j0, k0) where block covers i∈{i0,i0+1}, j∈{j0,j0+1}, k∈{k0,k0+1}
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]

    # Total rank: 4*7 (Strassen) + (75 - 4*8) schoolbook = 28 + 43 = 71
    total_rank = 4 * 7 + (n * m * p - 4 * 8)

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0

    # Strassen decomposition for a 2x2x2 block
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    for (i0, j0, k0) in blocks:
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

        # M1 = (a00+a11)(b00+b11)
        c = col
        U[i0*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1   # C00
        W[k1*n + i1][c] += 1   # C11
        col += 1

        # M2 = (a10+a11)b00
        c = col
        U[i1*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + i1][c] += 1   # C10
        W[k1*n + i1][c] += -1  # C11
        col += 1

        # M3 = a00(b01-b11)
        c = col
        U[i0*m + j0][c] += 1
        V[j0*p + k1][c] += 1
        V[j1*p + k1][c] += -1
        W[k1*n + i0][c] += 1   # C01
        W[k1*n + i1][c] += 1   # C11
        col += 1

        # M4 = a11(b10-b00)
        c = col
        U[i1*m + j1][c] += 1
        V[j1*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1   # C00
        W[k0*n + i1][c] += 1   # C10
        col += 1

        # M5 = (a00+a01)b11
        c = col
        U[i0*m + j0][c] += 1
        U[i0*m + j1][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += -1  # C00
        W[k1*n + i0][c] += 1   # C01
        col += 1

        # M6 = (a10-a00)(b00+b01)
        c = col
        U[i1*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + k1][c] += 1
        W[k1*n + i1][c] += 1   # C11
        col += 1

        # M7 = (a01-a11)(b10+b11)
        c = col
        U[i0*m + j1][c] += 1
        U[i1*m + j1][c] += -1
        V[j1*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1   # C00
        col += 1

    # Mark covered entries
    covered = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))

    # Remaining schoolbook entries
    remaining = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    remaining.append((i, j, k))

    # Now try to find rank-1 terms that can replace pairs of schoolbook terms
    # A rank-1 term (u, v, w) with u supported on {i1*m+j1, i2*m+j2},
    # v on {j1*p+k1, j2*p+k2}, w on {k1*n+i1, k2*n+i2}
    # can replace the two schoolbook terms if the coefficients work out.
    #
    # For two schoolbook terms (i1,j1,k1) and (i2,j2,k2) to be replaced by one
    # rank-1 term, we need:
    #   u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = 1
    #   u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1
    #   and all other products = 0
    #
    # This requires that the supports don't create cross-terms. Specifically:
    #   u[i1*m+j1]*v[j2*p+k1]*w[k1*n+i2] = 0
    #   u[i2*m+j2]*v[j1*p+k2]*w[k2*n+i1] = 0
    # etc.
    #
    # This is only possible if the "cross" indices are zero, which means
    # the supports must be structured so that cross products vanish.
    #
    # A simpler approach: try to find a rank-1 term that has support on
    # multiple U indices, multiple V indices, and multiple W indices, and
    # verify that it exactly reproduces the desired entries.
    #
    # Let me try a different tactic: use the fact that some remaining entries
    # share structure. For example, entries with the same j but different i,k
    # or same i but different j,k, etc.
    #
    # Actually, let me try a more systematic approach:
    # For each pair of remaining entries, check if they can be merged into
    # a single rank-1 term. The condition is that the 2x2x2 sub-tensor
    # formed by these two entries (and their "cross" terms) has rank 1.
    #
    # Two entries (i1,j1,k1) and (i2,j2,k2) can be merged iff:
    #   The tensor with 1s at these two positions and 0s elsewhere has rank 1.
    #   This happens iff (i1,j1,k1) = (i2,j2,k2) (same entry) or
    #   the two entries are "aligned" in a specific way.
    #
    # Actually, two distinct rank-1 tensors sum to a rank-1 tensor only if
    # they are proportional. Since both have coefficient 1, they must be
    # identical. So simple pair merging of schoolbook terms doesn't work.
    #
    # The Strassen trick works because it introduces NEW terms with coefficients
    # other than 1, and the combination of 7 new terms equals the sum of 8
    # old terms.
    #
    # Let me try to find similar patterns in the remaining 43 entries.
    # I'll search for 3-4 entry patterns that can be replaced by fewer terms.
    #
    # For now, let me just add the schoolbook terms and verify.
    # The rank is 71, which is our current best.

    for (i, j, k) in remaining:
        U[i*m + j][col] += 1
        V[j*p + k][col] += 1
        W[k*n + i][col] += 1
        col += 1

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    tensor = np.zeros((n, m, p))
    for r in range(U_arr.shape[1]):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        for i in range(n):
            for j in range(m):
                if u[i*m+j] != 0:
                    for k in range(p):
                        if v[j*p+k] != 0:
                            if w[k*n+i] != 0:
                                tensor[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]

    expected = np.ones((n, m, p))
    if not np.allclose(tensor, expected, atol=1e-10):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
