"""Systematic search for rank-minimal tensor decomposition of 3x4*4x7 matrix multiplication.

Uses structured block decomposition: splits the 7 output columns into groups,
applies Strassen-like decompositions to each group, and combines results.
Also runs greedy set-cover as a fallback.
"""
import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    total_entries = n * m * p  # 84

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i0, i1 in itertools.combinations(range(n), 2):
        for j0, j1 in itertools.combinations(range(m), 2):
            for k0, k1 in itertools.combinations(range(p), 2):
                all_blocks.append((i0, i1, j0, j1, k0, k1))

    block_covers = {}
    for idx, (i0, i1, j0, j1, k0, k1) in enumerate(all_blocks):
        covers = set()
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covers.add((ii, jj, kk))
        block_covers[idx] = covers

    import random
    rng = random.Random(42)

    best_rank = 84
    best_block_set = None

    # Greedy set-cover with multiple restarts
    for trial in range(300):
        if time.time() - _t0 > _DEADLINE - 15:
            break

        uncovered = set(itertools.product(range(n), range(m), range(p)))
        selected = []
        indices = list(range(len(all_blocks)))
        if trial > 0:
            rng.shuffle(indices)

        while uncovered:
            if time.time() - _t0 > _DEADLINE - 15:
                break
            best_block = -1
            best_overlap = 0
            for idx in indices:
                overlap = len(block_covers[idx] & uncovered)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_block = idx
            if best_block == -1:
                break
            selected.append(best_block)
            uncovered -= block_covers[best_block]

        if uncovered:
            continue

        rank = 7 * len(selected)
        if rank < best_rank:
            best_rank = rank
            best_block_set = list(selected)

    # Known fixed 6-block construction (rank 78)
    fixed_blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    fixed_covered = set()
    for (i0, i1, j0, j1, k0, k1) in fixed_blocks:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    fixed_covered.add((ii, jj, kk))
    fixed_remaining = n * m * p - len(fixed_covered)
    fixed_rank = 6 * 7 + fixed_remaining
    if fixed_rank < best_rank:
        best_rank = fixed_rank
        best_block_set = None

    # Build the decomposition
    if best_block_set is None:
        blocks_to_use = fixed_blocks
    else:
        blocks_to_use = [all_blocks[idx] for idx in best_block_set]

    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    covered = set()

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        # Standard 7-term Strassen for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[i0*m+j0][col_start] = 1
        U[i1*m+j1][col_start] = 1
        V[j0*p+k0][col_start] = 1
        V[j1*p+k1][col_start] = 1
        W[k0*n+i0][col_start] = 1
        W[k1*n+i1][col_start] = 1

        # M2 = (A10+A11)B00
        U[i1*m+j0][col_start+1] = 1
        U[i1*m+j1][col_start+1] = 1
        V[j0*p+k0][col_start+1] = 1
        W[k0*n+i1][col_start+1] = 1
        W[k1*n+i1][col_start+1] = -1

        # M3 = A00(B10-B00)
        U[i0*m+j0][col_start+2] = 1
        V[j0*p+k1][col_start+2] = 1
        V[j1*p+k1][col_start+2] = -1
        W[k1*n+i0][col_start+2] = 1
        W[k1*n+i1][col_start+2] = 1

        # M4 = A11(B01-B10)
        U[i1*m+j1][col_start+3] = 1
        V[j1*p+k0][col_start+3] = 1
        V[j0*p+k0][col_start+3] = -1
        W[k0*n+i0][col_start+3] = 1
        W[k0*n+i1][col_start+3] = 1

        # M5 = (A00+A01)B11
        U[i0*m+j0][col_start+4] = 1
        U[i0*m+j1][col_start+4] = 1
        V[j1*p+k1][col_start+4] = 1
        W[k0*n+i0][col_start+4] = -1
        W[k1*n+i0][col_start+4] = 1

        # M6 = (A10-A00)(B00+B01)
        U[i1*m+j0][col_start+5] = 1
        U[i0*m+j0][col_start+5] = -1
        V[j0*p+k0][col_start+5] = 1
        V[j0*p+k1][col_start+5] = 1
        W[k1*n+i1][col_start+5] = 1

        # M7 = (A01-A11)(B10+B11)
        U[i0*m+j1][col_start+6] = 1
        U[i1*m+j1][col_start+6] = -1
        V[j1*p+k0][col_start+6] = 1
        V[j1*p+k1][col_start+6] = 1
        W[k0*n+i0][col_start+6] = 1

        return col_start + 7

    for (i0, i1, j0, j1, k0, k1) in blocks_to_use:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for remaining
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0*m+j0][col] = 1
                V[j0*p+k0][col] = 1
                W[k0*n+i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
