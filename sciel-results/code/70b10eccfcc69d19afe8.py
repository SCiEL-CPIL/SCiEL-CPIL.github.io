"""Covering design C(24,7,3): fast vectorized greedy + efficient local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s

    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    def block_tset_ids(block):
        return np.array([tset_to_id[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)

    # Precompute: for each pair (a,b) with a<b, list of tset IDs containing {a,b}
    # For t=3: tset {a,b,c} has pair (a,b) -> c, so for each pair we can find which
    # tsets contain it by checking all c != a,b
    pair_to_tsets = {}
    for ts in all_tsets:
        for i in range(t):
            for j in range(i+1, t):
                pair = tuple(sorted((ts[i], ts[j])))
                pair_to_tsets.setdefault(pair, []).append(tset_to_id[ts])
    for pair in pair_to_tsets:
        pair_to_tsets[pair] = np.array(pair_to_tsets[pair], dtype=np.int32)

    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks = []

        while uncovered.any():
            unc_idx = np.where(uncovered)[0]
            start_id = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            start_ts = all_tsets[start_id]

            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                candidates = [p for p in range(v) if p not in block_set]
                # For each candidate, compute gain = number of newly covered tsets
                # A tset {x,y,z} is covered by adding p to block if:
                #   p is in the tset, and the other 2 elements are already in block
                # So gain = sum over (t-1)-subsets S of block: 
                #   count of uncovered tsets containing S and p
                # For t=3: for each pair (a,b) in block, count uncovered tsets {a,b,p}
                
                best_gain = -1
                best_point = -1
                subs = list(itertools.combinations(block, t-1))
                
                for p in candidates:
                    gain = 0
                    for sub in subs:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if uncovered[tid]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p

                if best_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for tid in block_tset_ids(block):
                uncovered[tid] = False

        return blocks

    # Phase 1: Greedy restarts
    best_blocks = None
    best_count = float('inf')

    max_restarts = 80
    for seed in range(max_restarts):
        if time.time() > deadline - 55:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Try a few more seeds with same structure
            for extra_seed in range(3):
                if time.time() > deadline - 55:
                    break
                b2 = greedy_construct(seed * 1000 + extra_seed)
                if len(b2) < best_count:
                    best_count = len(b2)
                    best_blocks = b2

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # Phase 2: Local search
    def compute_counts(blocks):
        counts = np.zeros(n_tsets, dtype=np.int32)
        for block in blocks:
            for tid in block_tset_ids(block):
                counts[tid] += 1
        return counts

    current_blocks = list(best_blocks)
    counts = compute_counts(current_blocks)

    def get_block_tset_set(block):
        return set(block_tset_ids(block).tolist())

    def is_redundant(block, counts):
        bts = get_block_tset_set(block)
        for tid in bts:
            if counts[tid] == 1:
                return False
        return True

    def try_remove_redundant(blocks, counts):
        new_blocks = []
        changed = False
        for b in blocks:
            if is_redundant(b, counts):
                for tid in get_block_tset_set(b):
                    counts[tid] -= 1
                changed = True
            else:
                new_blocks.append(b)
        return new_blocks, changed

    def try_swap_2_for_1(blocks, counts):
        n = len(blocks)
        if n < 2:
            return blocks, counts, False

        for i in range(n):
            if time.time() > deadline - 15:
                return blocks, counts, False
            for j in range(i+1, n):
                if time.time() > deadline - 15:
                    return blocks, counts, False

                b1 = blocks[i]
                b2 = blocks[j]
                ts1 = get_block_tset_set(b1)
                ts2 = get_block_tset_set(b2)
                all_ts = ts1 | ts2

                # Find tsets that would become uncovered
                u_need = []
                for tid in all_ts:
                    c = counts[tid]
                    in_b1 = 1 if tid in ts1 else 0
                    in_b2 = 1 if tid in ts2 else 0
                    if c == in_b1 + in_b2:
                        u_need.append(all_tsets[tid])

                if not u_need:
                    # Remove both
                    for tid in ts1:
                        counts[tid] -= 1
                    for tid in ts2:
                        counts[tid] -= 1
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    return new_blocks, counts, True

                # Find required points
                p_need = set()
                for ts in u_need:
                    p_need.update(ts)

                if len(p_need) <= k:
                    # Try to find a single block covering all u_need
                    # Must contain all points in p_need
                    required = list(p_need)
                    remaining_slots = k - len(required)
                    remaining_points = [p for p in range(v) if p not in p_need]

                    # Try several random completions
                    best_block = None
                    for attempt in range(8):
                        rng = random.Random(i * 10000 + j * 100 + attempt)
                        new_block = list(required)
                        for _ in range(remaining_slots):
                            p = rng.choice(remaining_points)
                            if p not in new_block:
                                new_block.append(p)
                        new_block.sort()
                        # Verify it covers all u_need
                        nb_ts = set(block_tset_ids(new_block).tolist())
                        if all(tid in nb_ts for tid in [tset_to_id[ts] for ts in u_need]):
                            # Check it's valid (k distinct elements)
                            if len(set(new_block)) == k:
                                best_block = new_block
                                break

                    if best_block is not None:
                        # Remove b1, b2; add best_block
                        for tid in ts1:
                            counts[tid] -= 1
                        for tid in ts2:
                            counts[tid] -= 1
                        for tid in get_block_tset_set(best_block):
                            counts[tid] += 1
                        new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                        new_blocks.append(best_block)
                        return new_blocks, counts, True

        return blocks, counts, False

    def try_swap_1_for_1(blocks, counts):
        n = len(blocks)
        if n < 1:
            return blocks, counts, False

        for i in range(n):
            if time.time() > deadline - 15:
                return blocks, counts, False

            b = blocks[i]
            ts_b = get_block_tset_set(b)

            # Find tsets only covered by this block
            u_need = []
            for tid in ts_b:
                if counts[tid] == 1:
                    u_need.append(all_tsets[tid])

            if not u_need:
                for tid in ts_b:
                    counts[tid] -= 1
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                return new_blocks, counts, True

            p_need = set()
            for ts in u_need:
                p_need.update(ts)

            if len(p_need) <= k:
                required = list(p_need)
                remaining_slots = k - len(required)
                remaining_points = [p for p in range(v) if p not in p_need]

                best_block = None
                for attempt in range(8):
                    rng = random.Random(i * 7919 + attempt)
                    new_block = list(required)
                    for _ in range(remaining_slots):
                        p = rng.choice(remaining_points)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    nb_ts = set(block_tset_ids(new_block).tolist())
                    if all(tset_to_id[ts] in nb_ts for ts in u_need):
                        if len(set(new_block)) == k:
                            best_block = new_block
                            break

                if best_block is not None and best_block != b:
                    for tid in ts_b:
                        counts[tid] -= 1
                    for tid in get_block_tset_set(best_block):
                        counts[tid] += 1
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(best_block)
                    return new_blocks, counts, True

        return blocks, counts, False

    # Iterative local search
    improved = True
    max_rounds = 300
    for _ in range(max_rounds):
        if time.time() > deadline - 30:
            break
        if not improved:
            break
        improved = False

        current_blocks, counts, rem = try_remove_redundant(current_blocks, counts)
        if rem:
            improved = True

        current_blocks, counts, swp = try_swap_2_for_1(current_blocks, counts)
        if swp:
            improved = True

        current_blocks, counts, swp = try_swap_1_for_1(current_blocks, counts)
        if swp:
            improved = True

    # SA local search for further improvement
    def sa_local_search(blocks, counts, max_iters=3000):
        blocks = list(blocks)
        counts = counts.copy()
        current_len = len(blocks)
        best_blocks = list(blocks)
        best_counts = counts.copy()
        best_len = current_len

        T_start = 6.0
        T_end = 0.03
        alpha = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        rng = random.Random(123)

        for it in range(max_iters):
            if time.time() > deadline - 8:
                break
            T *= alpha

            move_type = rng.random()

            if move_type < 0.3:
                new_blocks, new_counts, rem = try_remove_redundant(blocks, counts)
                if rem:
                    blocks = new_blocks
                    counts = new_counts
                    current_len = len(blocks)
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(blocks)
                        best_counts = counts.copy()
            elif move_type < 0.6:
                new_blocks, new_counts, swp = try_swap_2_for_1(blocks, counts)
                if swp:
                    blocks = new_blocks
                    counts = new_counts
                    current_len = len(blocks)
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(blocks)
                        best_counts = counts.copy()
            else:
                new_blocks, new_counts, swp = try_swap_1_for_1(blocks, counts)
                if swp:
                    blocks = new_blocks
                    counts = new_counts
                    current_len = len(blocks)
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(blocks)
                        best_counts = counts.copy()

        return best_blocks, best_counts

    if time.time() < deadline - 10:
        sa_blocks, sa_counts = sa_local_search(current_blocks, counts, max_iters=3000)
        if len(sa_blocks) < len(current_blocks):
            current_blocks = sa_blocks
            counts = sa_counts

    # Final verification
    covered = set()
    for block in current_blocks:
        for tid in block_tset_ids(block):
            covered.add(tid)

    if len(covered) < n_tsets:
        # Fallback to best_blocks
        return best_blocks

    return current_blocks
