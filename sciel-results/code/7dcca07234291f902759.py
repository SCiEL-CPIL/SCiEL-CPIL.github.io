"""Optimized rank decomposition for 3x4*4x7 matrix multiplication.

Uses exhaustive search over all maximal packings of disjoint 2x2x2 Strassen
blocks to find the minimum-rank decomposition. For 3x4x7, the structure is
small enough to enumerate all valid packings and pick the one with minimum
total rank (7 * num_blocks + num_uncovered).
"""
import time
import itertools
from functools import lru_cache

_DIMS = (3, 4, 7)
_DEADLINE = 145.0  # Safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = _DIMS

    # All possible 2x2x2 blocks
    all_blocks = [(i0, j0, k0) for i0 in range(n - 1) for j0 in range(m - 1) for k0 in range(p - 1)]
    num_blocks_total = len(all_blocks)  # 2*3*6 = 36

    # Represent each block by the set of (i,j,k) entries it covers
    def block_entries(b):
        i0, j0, k0 = b
        return frozenset((i, j, k) for i in range(i0, i0 + 2)
                         for j in range(j0, j0 + 2)
                         for k in range(k0, k0 + 2))

    # Precompute entry sets
    block_sets = [block_entries(b) for b in all_blocks]

    # Check disjointness via bitmask or set intersection
    # For speed, use a compact representation: each entry (i,j,k) -> index
    # Total entries: 3*4*7 = 84
    entry_to_idx = {}
    idx_to_entry = []
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entry_to_idx[(i, j, k)] = idx
                idx_to_entry.append((i, j, k))
                idx += 1
    total_entries = n * m * p  # 84

    # Use Python int as bitmask (84 bits fits easily)
    block_masks = []
    for bs in block_sets:
        mask = 0
        for (i, j, k) in bs:
            mask |= (1 << entry_to_idx[(i, j, k)])
        block_masks.append(mask)

    def are_disjoint(b1, b2):
        return (block_masks[b1] & block_masks[b2]) == 0

    # Find all maximal packings via backtracking
    # A packing is a set of mutually disjoint blocks
    # We want to find the packing that minimizes: 7*len(packing) + (total_entries - union_size)
    # = 7*len(packing) + 84 - union_size
    # Since union_size = 8 * len(packing) (each block covers 8 entries, all disjoint),
    # rank = 7*len(packing) + 84 - 8*len(packing) = 84 - len(packing)
    # So we just want to MAXIMIZE the number of blocks in the packing!

    best_packing = []
    best_size = 0

    def backtrack(start, current, current_mask):
        nonlocal best_packing, best_size
        if time.time() - _t0 > _DEADLINE:
            return

        if len(current) > best_size:
            best_size = len(current)
            best_packing = current[:]

        # Upper bound: can we beat best_size?
        # At most (num_blocks_total - start) more blocks, but constrained by disjointness
        # Simple bound: remaining entries / 8
        remaining_mask = ((1 << total_entries) - 1) & ~current_mask
        remaining_count = bin(remaining_mask).count('1')
        if len(current) + remaining_count // 8 <= best_size:
            return

        for i in range(start, num_blocks_total):
            if current_mask & block_masks[i]:
                continue
            if time.time() - _t0 > _DEADLINE:
                return
            current.append(i)
            backtrack(i + 1, current, current_mask | block_masks[i])
            current.pop()

    backtrack(0, [], 0)

    # If exhaustive search didn't finish, fall back to greedy
    if time.time() - _t0 > _DEADLINE - 5.0 and best_size < 3:
        # Greedy fallback
        selected = []
        for i in range(num_blocks_total):
            if all(are_disjoint(i, s) for s in selected):
                selected.append(i)
        if len(selected) > best_size:
            best_size = len(selected)
            best_packing = selected

    # Build the decomposition from best_packing
    num_strassen = len(best_packing)
    rank = 84 - num_strassen  # 7*num_strassen + (84 - 8*num_strassen)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
        U[i0 * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0 + 1) * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = -1
        col += 1
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M4: (A[i0+1,j0+1]) * (B[j0+1,k0]-B[j0,k0])
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0 + 1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0 * m + (j0 + 1)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = -1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for bi in best_packing:
        add_strassen(*all_blocks[bi])

    # Add schoolbook terms for uncovered entries
    covered = set()
    for bi in best_packing:
        i0, j0, k0 = all_blocks[bi]
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    assert col == rank, f"Rank mismatch: {col} vs {rank}"

    # Verification
    if time.time() - _t0 < _DEADLINE - 5.0:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    if val != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): {val}")

    return {"U": U, "V": V, "W": W}
