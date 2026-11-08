import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices, frozenset(indices)))

    num_blocks_total = len(all_blocks)

    # Pre-compute compatibility: two blocks are compatible if their index sets are disjoint
    idx_to_bit = {}
    bit_to_idx = {}
    bit = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_to_bit[(i, j, k)] = bit
                bit_to_idx[bit] = (i, j, k)
                bit += 1

    block_bitmasks = []
    for params, indices, _ in all_blocks:
        mask = 0
        for idx in indices:
            mask |= (1 << idx_to_bit[idx])
        block_bitmasks.append(mask)

    # Precompute conflict lists: conflicts[a] = list of block indices that intersect block a
    conflicts = [[] for _ in range(num_blocks_total)]
    for a in range(num_blocks_total):
        for b in range(a + 1, num_blocks_total):
            if (block_bitmasks[a] & block_bitmasks[b]) != 0:
                conflicts[a].append(b)
                conflicts[b].append(a)

    # Degrees
    degrees = [len(conflicts[a]) for a in range(num_blocks_total)]

    best_num_blocks = 0
    best_packing = []

    # Greedy with multiple restarts and different ordering strategies
    num_trials = 5000
    for t in range(num_trials):
        if time.time() > deadline - 15.0:
            break

        # Different ordering strategies
        if t % 4 == 0:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        elif t % 4 == 1:
            # Most constrained first
            order = sorted(range(num_blocks_total), key=lambda x: degrees[x])
            random.shuffle(order)
        elif t % 4 == 2:
            # Random with bias towards low-degree
            order = list(range(num_blocks_total))
            weights = [1.0 / (degrees[x] + 1) for x in range(num_blocks_total)]
            total_w = sum(weights)
            normalized = [w / total_w for w in weights]
            for i in range(len(order) - 1):
                r = random.random()
                cumsum = 0
                for j in range(i, len(order)):
                    cumsum += normalized[order[j]]
                    if cumsum >= r:
                        order[i], order[j] = order[j], order[i]
                        break
        else:
            # Reverse sorted by degree (least constrained first)
            order = sorted(range(num_blocks_total), key=lambda x: -degrees[x])
            random.shuffle(order)

        used_mask = 0
        current_packing = []

        for idx in order:
            if (block_bitmasks[idx] & used_mask) == 0:
                current_packing.append(idx)
                used_mask |= block_bitmasks[idx]

        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
            if best_num_blocks >= 13:
                break

    # Local improvement: try to swap blocks to improve packing
    if time.time() < deadline - 20.0 and best_num_blocks < 13:
        improved = True
        while improved:
            improved = False
            current_set = set(best_packing)
            used_mask = 0
            for idx in best_packing:
                used_mask |= block_bitmasks[idx]

            # Try adding blocks that don't conflict
            added = False
            for j in range(num_blocks_total):
                if time.time() > deadline - 15.0:
                    break
                if j in current_set:
                    continue
                if (block_bitmasks[j] & used_mask) == 0:
                    best_packing.append(j)
                    current_set.add(j)
                    used_mask |= block_bitmasks[j]
                    added = True
                    improved = True

            if added:
                continue

            # Try 1-for-1 swaps
            for i in range(len(best_packing)):
                if time.time() > deadline - 15.0:
                    break
                for j in range(num_blocks_total):
                    if j in current_set:
                        continue
                    # Check if j conflicts only with best_packing[i]
                    conflict_count = 0
                    for k in range(len(best_packing)):
                        if (block_bitmasks[j] & block_bitmasks[best_packing[k]]) != 0:
                            conflict_count += 1
                    if conflict_count == 1 and (block_bitmasks[j] & block_bitmasks[best_packing[i]]) != 0:
                        # Check if removing i and adding j still allows all others
                        new_used = 0
                        for k in range(len(best_packing)):
                            if k != i:
                                new_used |= block_bitmasks[best_packing[k]]
                        if (block_bitmasks[j] & new_used) == 0:
                            best_packing[i] = j
                            current_set.discard(j)
                            current_set.add(j)
                            used_mask = new_used | block_bitmasks[j]
                            improved = True
                            break

    # Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for block_idx in best_packing:
        params, indices, _ = all_blocks[block_idx]
        i1, i2, j1, j2, k1, k2 = params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered |= set(indices)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
