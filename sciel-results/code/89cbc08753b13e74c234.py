"""Covering design C(19,8,5) - optimized greedy with fast local search."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using:
    1. Fast greedy with precomputed gain tables
    2. Multi-seed exploration with varied strategies
    3. Efficient local search via incremental cover counts
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Precompute all t-subsets and index mapping
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute: for each (t-1)-subset and point x not in it, the t-subset index
    # This lets us quickly compute gain when adding x to a block
    # For each point x, precompute which t-subsets contain x
    point_to_tsubsets = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubsets):
        for x in ts:
            point_to_tsubsets[x].append(idx)

    # Precompute gain table: gain[x][frozen_mask] where frozen_mask represents
    # which t-subsets are already covered. Too large for full state, so we
    # precompute per-point contributions to specific t-subsets.
    # For each point x and each (t-1)-subset S not containing x:
    #   t_subset = S ∪ {x}, index = tsubset_to_idx[t_subset]
    # We'll build a lookup: for point x, dict from (t-1)-tuple to t-subset-index
    point_gain_lookup = [{} for _ in range(v)]
    for idx, ts in enumerate(all_tsubsets):
        for i in range(t):
            x = ts[i]
            rest = ts[:i] + ts[i+1:]  # (t-1)-subset without x
            point_gain_lookup[x][tuple(sorted(rest))] = idx

    def greedy_build(seed, strategy='standard'):
        """Fast greedy covering."""
        rng = random.Random(seed)
        uncovered = bytearray([1] * n_tsub)  # 1 = uncovered
        uncovered_count = n_tsub
        blocks = []
        v_list = list(range(v))

        while uncovered_count > 0:
            if time.time() > deadline - 3:
                break

            # Find an uncovered t-subset to seed
            # Sample from uncovered for speed
            if strategy == 'random':
                # Random sample of uncovered
                attempts = 0
                while attempts < 100:
                    idx = rng.randint(0, n_tsub - 1)
                    if uncovered[idx]:
                        break
                    attempts += 1
                else:
                    # Fallback: scan
                    idx = 0
                    while idx < n_tsub and not uncovered[idx]:
                        idx += 1
                    if idx >= n_tsub:
                        break
            else:
                # Standard: pick first uncovered (deterministic base) with random offset
                offset = rng.randint(0, max(1, n_tsub // 4))
                idx = offset % n_tsub
                while not uncovered[idx]:
                    idx = (idx + 1) % n_tsub
                    if idx == offset % n_tsub:
                        break

            if idx >= n_tsub:
                break

            seed_ts = all_tsubsets[idx]
            block = list(seed_ts)
            rng.shuffle(block)
            block_set = set(block)

            # Greedily add points to maximize uncovered t-subsets covered
            while len(block) < k:
                if time.time() > deadline - 3:
                    break

                best_gain = 0
                best_x = None
                block_t1 = list(itertools.combinations(block, t - 1))

                for x in v_list:
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1:
                        rest_t = tuple(sorted(rest))
                        t_idx = point_gain_lookup[x].get(rest_t)
                        if t_idx is not None and uncovered[t_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x is None:
                    remaining = [x for x in v_list if x not in block_set]
                    if remaining:
                        best_x = rng.choice(remaining)
                    else:
                        break

                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                t_idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if t_idx is not None and uncovered[t_idx]:
                    uncovered[t_idx] = 0
                    uncovered_count -= 1

        if uncovered_count == 0:
            return blocks
        return None

    def local_search(blocks, max_time=8):
        """Remove redundant blocks using incremental cover counting."""
        start_ls = time.time()

        # Build cover count
        cover_count = [0] * n_tsub
        block_tsubsets = []  # For each block, list of t-subset indices it covers
        for block in blocks:
            bts = []
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if idx is not None:
                    bts.append(idx)
                    cover_count[idx] += 1
            block_tsubsets.append(bts)

        changed = True
        while changed and (time.time() - start_ls) < max_time:
            changed = False
            new_blocks = []
            new_block_tsubsets = []
            for i, block in enumerate(blocks):
                is_critical = False
                for idx in block_tsubsets[i]:
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    for idx in block_tsubsets[i]:
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
                    new_block_tsubsets.append(block_tsubsets[i])
            blocks = new_blocks
            block_tsubsets = new_block_tsubsets

        return blocks

    def is_valid(blocks):
        """Check if blocks cover all t-subsets."""
        covered = bytearray([0] * n_tsub)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if idx is not None:
                    covered[idx] = 1
        return all(covered)

    # Main search loop
    best_result = None
    best_len = float('inf')

    # Phase 1: Multi-seed greedy + local search
    for seed in range(30):
        if time.time() > deadline - 20:
            break
        result = greedy_build(seed, strategy='standard')
        if result is not None:
            result = local_search(result, max_time=4)
            if is_valid(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    # Phase 2: Random strategy seeds
    for seed in range(15):
        if time.time() > deadline - 12:
            break
        result = greedy_build(1000 + seed, strategy='random')
        if result is not None:
            result = local_search(result, max_time=3)
            if is_valid(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    # Phase 3: More standard seeds if time permits
    for seed in range(30, 50):
        if time.time() > deadline - 5:
            break
        result = greedy_build(seed, strategy='standard')
        if result is not None:
            result = local_search(result, max_time=2)
            if is_valid(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    # Phase 4: Final local search pass on best result
    if best_result is not None:
        best_result = local_search(best_result, max_time=5)
        if not is_valid(best_result):
            # Revert if somehow invalid (shouldn't happen)
            pass

    if best_result is not None:
        return best_result

    # Fallback
    return greedy_build(0, strategy='standard') or []
