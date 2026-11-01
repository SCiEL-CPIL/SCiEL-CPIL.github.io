import numpy as np
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 120s

    points = list(range(v))

    # ---------- Precomputation ----------
    # All t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_idx = {sub: i for i, sub in enumerate(t_subsets)}

    # All k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_b = len(k_blocks)

    # Map each point to a bit in a 64-bit int (v <= 64)
    point_bit = {p: (1 << p) for p in points}

    # For each block: list of t-subset indices it covers, and a 64-bit mask of its points
    block_cover = []   # list of np.array of int32 indices
    block_mask = np.zeros(n_b, dtype=np.uint64)
    for bi, block in enumerate(k_blocks):
        m = 0
        for p in block:
            m |= point_bit[p]
        block_mask[bi] = m
        subs = list(itertools.combinations(block, t))
        idx = np.array([t_idx[s] for s in subs], dtype=np.int32)
        block_cover.append(idx)

    k_choose_t = len(block_cover[0])

    # For each t-subset: which blocks cover it (as numpy int32 array)
    tsub_blocks = [[] for _ in range(n_t)]
    for bi, inds in enumerate(block_cover):
        for s in inds:
            tsub_blocks[s].append(bi)
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]

    # For fast "does block b cover subset s" via bitmask:
    # subset_mask[s] = 64-bit mask of the t points in t_subsets[s]
    subset_mask = np.zeros(n_t, dtype=np.uint64)
    for i, sub in enumerate(t_subsets):
        m = 0
        for p in sub:
            m |= point_bit[p]
        subset_mask[i] = m

    best_blocks = None
    best_len = float('inf')

    def time_ok():
        return time.time() < deadline

    def greedy():
        """Greedy set cover. Returns (selected_indices, completed)."""
        gains = np.full(n_b, k_choose_t, dtype=np.int32)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        unc_count = n_t
        while unc_count > 0:
            if not time_ok():
                return selected, False
            b = int(np.argmax(gains))
            if gains[b] == 0:
                return selected, False
            selected.append(b)
            inds = block_cover[b]
            # unique: subsets in inds that are still uncovered
            unique = inds[uncovered[inds]]
            if len(unique) > 0:
                unc_count -= len(unique)
                uncovered[unique] = False
                # decrement gains for all blocks covering each newly-covered subset
                # build flat list of block-indices to decrement
                to_dec = []
                for s in unique:
                    to_dec.append(tsub_blocks_np[s])
                if to_dec:
                    flat = np.concatenate(to_dec)
                    np.add.at(gains, flat, -1)
        return selected, True

    def compute_cover_count(selected):
        cc = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cc[block_cover[b]] += 1
        return cc

    def local_search(selected):
        """Redundancy removal + 1-for-1 swaps + 2-for-1 merges + 2-for-2 pair swaps."""
        cc = compute_cover_count(selected)
        improved = True
        while improved and time_ok():
            improved = False

            # --- Redundancy removal ---
            redundant = []
            for b in selected:
                if np.min(cc[block_cover[b]]) > 1:
                    redundant.append(b)
            if redundant:
                for b in redundant:
                    selected.remove(b)
                    cc[block_cover[b]] -= 1
                improved = True
                continue

            # --- Compute unique subsets per block ---
            uniq = {}  # block -> list of subset indices unique to it
            for b in selected:
                subs = block_cover[b]
                u = [s for s in subs if cc[s] == 1]
                uniq[b] = u

            # --- 1-for-1 swap ---
            min_u = min(len(u) for u in uniq.values()) if uniq else 0
            if min_u > 0:
                cands_out = [b for b, u in uniq.items() if len(u) == min_u]
                random.shuffle(cands_out)
                swap_done = False
                for b_out in cands_out:
                    if not time_ok():
                        break
                    s_uniq = uniq[b_out]
                    if not s_uniq:
                        continue
                    first_s = s_uniq[0]
                    cands_in = tsub_blocks_np[first_s]
                    exclude = set(selected)
                    # Precompute masks of unique subsets for fast check
                    s_masks = [subset_mask[s] for s in s_uniq]
                    valid = []
                    for c in cands_in:
                        c = int(c)
                        if c in exclude:
                            continue
                        cm = block_mask[c]
                        ok = True
                        for sm in s_masks:
                            if (cm & sm) != sm:
                                ok = False
                                break
                        if ok:
                            valid.append(c)
                    if valid:
                        b_in = random.choice(valid)
                        selected.remove(b_out)
                        selected.append(b_in)
                        cc[block_cover[b_out]] -= 1
                        cc[block_cover[b_in]] += 1
                        swap_done = True
                        improved = True
                        break
                if swap_done:
                    continue

            # --- 2-for-1 merge: find two blocks whose union of unique subsets is covered by one block ---
            # Only try if |selected| > 2 and time permits
            if len(selected) > 2 and time_ok():
                # For each pair of blocks, check if their combined unique subsets can be covered by a single new block
                # Limit to pairs where both have small unique sets
                small = [b for b in selected if len(uniq[b]) <= 8]
                random.shuffle(small)
                merge_done = False
                for i in range(len(small)):
                    if not time_ok():
                        break
                    for j in range(i + 1, len(small)):
                        b1, b2 = small[i], small[j]
                        u1 = uniq[b1]
                        u2 = uniq[b2]
                        combined = set(u1) | set(u2)
                        if len(combined) == 0:
                            continue
                        # Find a block c (not b1, b2) that covers all subsets in combined
                        # Use first subset to get candidates
                        first_s = next(iter(combined))
                        cands = tsub_blocks_np[first_s]
                        exclude = {b1, b2}
                        c_masks = [subset_mask[s] for s in combined]
                        found = False
                        for c in cands:
                            c = int(c)
                            if c in exclude:
                                continue
                            cm = block_mask[c]
                            ok = True
                            for sm in c_masks:
                                if (cm & sm) != sm:
                                    ok = False
                                    break
                            if ok:
                                # Perform merge: remove b1, b2, add c
                                selected.remove(b1)
                                selected.remove(b2)
                                selected.append(c)
                                cc[block_cover[b1]] -= 1
                                cc[block_cover[b2]] -= 1
                                cc[block_cover[c]] += 1
                                merge_done = True
                                found = True
                                break
                        if found:
                            improved = True
                            break
                    if merge_done:
                        break
                if merge_done:
                    continue

            # --- 2-for-2 pair swap: replace (b1, b2) with (c1, c2) ---
            # This is more expensive; only try a few random pairs
            if len(selected) >= 4 and time_ok():
                # Pick a few random pairs of blocks with small unique sets
                small = [b for b in selected if len(uniq[b]) <= 6]
                if len(small) >= 2:
                    random.shuffle(small)
                    pair_swap_done = False
                    for i in range(min(5, len(small) - 1)):
                        if not time_ok():
                            break
                        b1, b2 = small[i], small[i + 1]
                        u1 = uniq[b1]
                        u2 = uniq[b2]
                        if not u1 or not u2:
                            continue
                        # We need c1 covering u1 and c2 covering u2 (or cross)
                        # Try: c1 covers u1, c2 covers u2
                        cands1 = tsub_blocks_np[u1[0]]
                        cands2 = tsub_blocks_np[u2[0]]
                        exclude = {b1, b2}
                        u1_masks = [subset_mask[s] for s in u1]
                        u2_masks = [subset_mask[s] for s in u2]
                        found = False
                        for c1 in cands1:
                            if found:
                                break
                            c1 = int(c1)
                            if c1 in exclude:
                                continue
                            cm1 = block_mask[c1]
                            ok1 = all((cm1 & sm) == sm for sm in u1_masks)
                            if not ok1:
                                continue
                            for c2 in cands2:
                                c2 = int(c2)
                                if c2 in exclude or c2 == c1:
                                    continue
                                cm2 = block_mask[c2]
                                ok2 = all((cm2 & sm) == sm for sm in u2_masks)
                                if ok2:
                                    selected.remove(b1)
                                    selected.remove(b2)
                                    selected.append(c1)
                                    selected.append(c2)
                                    cc[block_cover[b1]] -= 1
                                    cc[block_cover[b2]] -= 1
                                    cc[block_cover[c1]] += 1
                                    cc[block_cover[c2]] += 1
                                    pair_swap_done = True
                                    found = True
                                    break
                        if pair_swap_done:
                            improved = True
                            break
                    if pair_swap_done:
                        continue

            # No improvement this round
            break

        return selected

    # ---------- Main loop: multiple restarts ----------
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
    si = 0
    while time_ok():
        seed = seeds[si % len(seeds)]
        si += 1
        random.seed(seed)
        np.random.seed(seed)

        selected, completed = greedy()
        if not completed:
            continue

        # Local search
        selected = local_search(selected)

        cur_len = len(selected)
        if cur_len < best_len:
            best_len = cur_len
            best_blocks = [list(k_blocks[i]) for i in selected]

        # If we hit a good solution, keep trying to improve
        # Stop if time is low
        if time.time() > deadline - 5.0:
            break

    # Fallback
    if best_blocks is None:
        selected, completed = greedy()
        best_blocks = [list(k_blocks[i]) for i in selected]

    return best_blocks
