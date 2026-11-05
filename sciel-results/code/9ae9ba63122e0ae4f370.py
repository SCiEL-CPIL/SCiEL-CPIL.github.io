"""Covering design C(v,k,t) via vectorized greedy + local improvement."""
import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute for each point, the list of t-subsets it participates in
    # point_t_subs[x] = list of indices of t-subsets containing x
    point_t_subs = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_t_subs[x].append(i)

    # Convert to numpy arrays for vectorized ops
    point_t_subs_arr = [np.array(point_t_subs[x], dtype=np.int32) for x in range(v)]

    def build_greedy(seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # uncovered: boolean array, True means still uncovered
        uncovered = np.ones(num_t, dtype=bool)
        # cover_count: how many times each t-subset is covered
        cover_count = np.zeros(num_t, dtype=np.int32)
        blocks = []

        while uncovered.any():
            if time.time() > deadline:
                break

            # Pick the first uncovered t-subset (or a random one for diversity)
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            # Use the one with lowest frequency (rare) for better coverage
            # Actually for greedy, we want to cover the hardest ones first
            # Let's pick the one with the fewest possible extensions
            # For simplicity, pick the first uncovered
            idx = uncovered_indices[0]
            block = list(all_t_subsets[idx])

            while len(block) < k:
                block_set = set(block)
                remaining = [x for x in range(v) if x not in block_set]
                if not remaining:
                    break

                # For each candidate point, compute how many new t-subsets it would cover
                # A t-subset is covered by adding x if it contains x and (t-1) elements from block
                # and is currently uncovered

                gains = np.zeros(len(remaining), dtype=np.int32)
                for j, x in enumerate(remaining):
                    # t-subsets containing x
                    x_subs = point_t_subs_arr[x]
                    if len(x_subs) == 0:
                        continue
                    # Need t-1 other elements from block
                    # For each (t-1)-subset of block, check if combined with x gives an uncovered t-subset
                    block_arr = np.array(block, dtype=np.int32)
                    # All (t-1) subsets of block
                    if t - 1 <= len(block):
                        for combo in itertools.combinations(block, t - 1):
                            combo_set = set(combo)
                            # Check if any t-subset in x_subs has all elements in combo_set + {x}
                            # i.e., the t-subset minus {x} equals some (t-1)-subset of block
                            # More efficiently: for each x_sub in x_subs, check if x_sub - {x} subset of block
                            for sub_idx in x_subs:
                                ts = all_t_subsets[sub_idx]
                                # Check if ts contains x
                                # and ts - {x} is subset of block
                                # and ts is uncovered
                                if uncovered[sub_idx]:
                                    ts_set = set(ts)
                                    ts_set.discard(x)
                                    if ts_set.issubset(combo_set):
                                        gains[j] += 1

                # Pick the point with max gain, break ties randomly
                max_gain = gains.max()
                if max_gain == 0:
                    # No point adds coverage, just add a random point
                    block.append(rng.choice(remaining))
                else:
                    candidates = [remaining[j] for j in range(len(remaining)) if gains[j] == max_gain]
                    block.append(rng.choice(candidates))

            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)

            # Update uncovered and cover_count
            for sub in itertools.combinations(block, t):
                si = t_idx[sub]
                cover_count[si] += 1
                if cover_count[si] == 1:
                    uncovered[si] = False

        return blocks

    def remove_redundant(blocks, cover_count, uncovered):
        """Remove blocks that are not essential (their removal doesn't uncover any t-subset)."""
        if time.time() > deadline:
            return blocks
        essential = [True] * len(blocks)
        # Rebuild cover_count from scratch for accuracy
        cc = np.zeros(num_t, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cc[t_idx[sub]] += 1

        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            for bi in range(len(blocks)):
                if not essential[bi]:
                    continue
                # Temporarily remove block bi
                for sub in itertools.combinations(blocks[bi], t):
                    cc[t_idx[sub]] -= 1
                # Check if any t-subset now has count 0
                if np.any(cc == 0):
                    # Restore
                    for sub in itertools.combinations(blocks[bi], t):
                        cc[t_idx[sub]] += 1
                else:
                    # Block is redundant, keep it removed
                    essential[bi] = False
                    removed_any = True
                    break

        return [blocks[i] for i in range(len(blocks)) if essential[i]]

    def local_search(blocks, cover_count):
        """Try to swap points in blocks to cover more t-subsets."""
        if time.time() > deadline:
            return blocks

        cc = np.zeros(num_t, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cc[t_idx[sub]] += 1

        best_blocks = list(blocks)
        best_cc = cc.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for bi in range(len(best_blocks)):
                if time.time() > deadline:
                    return best_blocks
                for pi in range(len(best_blocks[bi])):
                    old_pt = best_blocks[bi][pi]
                    for new_pt in range(v):
                        if new_pt == old_pt:
                            continue
                        if new_pt in best_blocks[bi]:
                            continue
                        # Try swapping
                        new_block = list(best_blocks[bi])
                        new_block[pi] = new_pt
                        new_block.sort()
                        new_block = tuple(new_block)

                        # Compute change in coverage
                        old_subs = list(itertools.combinations(best_blocks[bi], t))
                        new_subs = list(itertools.combinations(new_block, t))

                        gain = 0
                        for sub in new_subs:
                            si = t_idx[sub]
                            if cc[si] == 0:
                                gain += 1
                        loss = 0
                        for sub in old_subs:
                            si = t_idx[sub]
                            if cc[si] == 1:
                                loss += 1

                        if gain > loss:
                            # Apply the swap
                            for sub in old_subs:
                                cc[t_idx[sub]] -= 1
                            for sub in new_subs:
                                cc[t_idx[sub]] += 1
                            best_blocks[bi] = new_block
                            best_cc = cc
                            improved = True
                            break

        return best_blocks

    # Main: try multiple greedy restarts with different seeds, keep the best
    best_blocks = None
    best_count = float('inf')

    # First, a deterministic greedy
    blocks = build_greedy(seed=42)
    cc = np.zeros(num_t, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cc[t_idx[sub]] += 1

    if np.all(cc >= 1):
        blocks = remove_redundant(blocks, cc, None)
        blocks = local_search(blocks, cc)
        best_blocks = blocks
        best_count = len(blocks)

    # Try more restarts if time allows
    for attempt in range(1, 50):
        if time.time() > deadline - 5:
            break
        seed = 42 + attempt
        blocks = build_greedy(seed=seed)
        cc = np.zeros(num_t, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cc[t_idx[sub]] += 1

        if np.all(cc >= 1):
            blocks = remove_redundant(blocks, cc, None)
            blocks = local_search(blocks, cc)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)

    if best_blocks is None:
        # Fallback: return the first greedy result even if incomplete (shouldn't happen)
        best_blocks = build_greedy(seed=0)

    return [list(b) for b in best_blocks]
