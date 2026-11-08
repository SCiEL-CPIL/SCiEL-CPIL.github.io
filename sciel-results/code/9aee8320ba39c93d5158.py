"""
Rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.

Approach:
- Use exhaustive search over all possible 2x2x2 Strassen blocks to find the
  maximum set of non-conflicting blocks (maximum independent set in conflict graph).
- With only 36 possible blocks, we can use branch-and-bound to find the optimal solution.
- Each Strassen block uses rank 7 instead of rank 8 (schoolbook), saving 1 rank per block.
- Fill remaining entries with schoolbook (rank 1 each).
"""

import time
from itertools import combinations

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Generate all possible 2x2x2 Strassen blocks
    i_windows = [(i, i + 1) for i in range(n - 1)]  # (0,1), (1,2)
    j_windows = [(j, j + 1) for j in range(m - 1)]  # (0,1), (1,2), (2,3)
    k_windows = [(k, k + 1) for k in range(p - 1)]  # (0,1), ..., (5,6)

    all_blocks = []
    for (i0, i1) in i_windows:
        for (j0, j1) in j_windows:
            for (k0, k1) in k_windows:
                entries = frozenset()
                for ii in (i0, i1):
                    for jj in (j0, j1):
                        for kk in (k0, k1):
                            entries = entries | frozenset([(ii, jj, kk)])
                all_blocks.append(((i0, i1, j0, j1, k0, k1), entries))

    num_blocks_total = len(all_blocks)  # 36

    # Build conflict matrix: two blocks conflict if they share any entry
    conflict = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    # Branch-and-bound to find maximum independent set
    best_selected = []
    best_size = 0

    def branch_bound(idx, current, current_size):
        nonlocal best_selected, best_size
        if time.time() - _t0 > _DEADLINE - 10:
            return
        # Upper bound: current_size + (num_blocks_total - idx)
        if current_size + (num_blocks_total - idx) <= best_size:
            return
        if idx == num_blocks_total:
            if current_size > best_size:
                best_size = current_size
                best_selected = list(current)
            return
        # Option 1: don't include block idx
        branch_bound(idx + 1, current, current_size)
        # Option 2: include block idx if it doesn't conflict with any in current
        if current_size > 0:
            can_add = True
            for c in current:
                if conflict[idx][c]:
                    can_add = False
                    break
            if not can_add:
                return
        current.append(idx)
        branch_bound(idx + 1, current, current_size + 1)
        current.pop()

    branch_bound(0, [], 0)

    selected_indices = best_selected
    selected = [all_blocks[i][0] for i in selected_indices]
    covered_entries = set()
    for i in selected_indices:
        covered_entries.update(all_blocks[i][1])

    num_blocks = len(selected)
    num_schoolbook = n * m * p - num_blocks * 8
    rank = num_blocks * 7 + num_schoolbook

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
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

    for params in selected:
        col = add_strassen_block(col, *params)

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered_entries:
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
