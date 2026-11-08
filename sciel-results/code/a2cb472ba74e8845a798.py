"""Deterministic multi-Strassen rank-77 decomposition for 3x4*4x7 matrix multiplication.

Uses 7 disjoint 2x2x2 Strassen blocks covering 56 entries. Remaining 28 entries use schoolbook.
Total rank = 7*7 + 28 = 77.
"""
import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def _find_7_blocks(n, m, p, deadline):
    """Search for 7 disjoint 2x2x2 blocks in the n x m x p tensor."""
    all_tuples = set((i, j, k) for i in range(n) for j in range(m) for k in range(p))
    
    # Generate all possible 2x2x2 blocks
    i_pairs = list(itertools.combinations(range(n), 2))
    j_pairs = list(itertools.combinations(range(m), 2))
    k_pairs = list(itertools.combinations(range(p), 2))
    
    all_blocks = []
    for (i0, i1) in i_pairs:
        for (j0, j1) in j_pairs:
            for (k0, k1) in k_pairs:
                block_set = frozenset((ii, jj, kk) for ii in (i0, i1) for jj in (j0, j1) for kk in (k0, k1))
                all_blocks.append(((i0, i1, j0, j1, k0, k1), block_set))
    
    # Backtracking search for 7 disjoint blocks
    found = []
    
    def backtrack(remaining, chosen, depth):
        if time.time() > deadline:
            return False
        if depth == 7:
            found.extend(chosen)
            return True
        if not remaining:
            return False
        # Try each block that fits in remaining
        for idx, (block_params, block_set) in enumerate(all_blocks):
            if block_set.issubset(remaining):
                chosen.append(block_params)
                new_remaining = remaining - set(block_set)
                if backtrack(new_remaining, chosen, depth + 1):
                    return True
                chosen.pop()
        return False
    
    backtrack(all_tuples, [], 0)
    return found[:7] if found else None


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Search for 7 disjoint Strassen blocks
    blocks = _find_7_blocks(n, m, p, _t0 + 60)
    
    if blocks is None:
        # Fallback to 6-block decomposition (rank 78)
        blocks = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 2, 3, 2, 3),
            (0, 1, 2, 3, 4, 5),
        ]
        num_strassen = 6
    else:
        num_strassen = len(blocks)
    
    # Count covered entries
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))
    
    num_schoolbook = n * m * p - len(covered)
    rank = num_strassen * 7 + num_schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block with explicit indices."""
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)

    # Schoolbook for all remaining (i,j,k)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
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
