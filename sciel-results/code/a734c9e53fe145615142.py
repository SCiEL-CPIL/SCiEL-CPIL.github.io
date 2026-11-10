"""Bitset-accelerated greedy covering with aggressive local search for C(24,7,3)."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin below 150s

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}

    # For each t-subset index, store the set of points (as frozenset for quick issubset)
    t_sets = [frozenset(ts) for ts in all_t]

    # Precompute for each point x, the list of t-subset indices that contain x
    point_to_tsubsets = defaultdict(list)
    for i, ts in enumerate(all_t):
        for x in ts:
            point_to_tsubsets[x].append(i)

    # Precompute for each (t-1)-subset, the list of t-subsets that contain it
    # This helps in gain computation
    t_minus_1 = t - 1
    ext_map = defaultdict(list)
    for i, ts in enumerate(all_t):
        for pos in range(t):
            sub = ts[:pos] + ts[pos+1:]
            ext_map[sub].append(i)

    def compute_gain(block_set, x, uncovered):
        """Compute how many new t-subsets are covered by adding point x to block."""
        gain = 0
        # For each (t-1)-subset A of block, the t-subset A∪{x} is newly covered
        block_list = sorted(block_set)
        for A in itertools.combinations(block_list, t_minus_1):
            ts = tuple(sorted(A + (x,)))
            if ts in t_index:
                if uncovered[t_index[ts]]:
                    gain += 1
        return gain

    def greedy_cover(rng, time_limit):
        """Greedy set cover: repeatedly pick block covering most uncovered t-subsets."""
        uncovered = bytearray([1]) * n_t  # 1 = uncovered
        blocks = []
        remaining = n_t

        while remaining > 0:
            if time.time() > time_limit:
                break

            # Find the uncovered t-subset with the fewest "options" (hardest to cover)
            # Or just pick the first uncovered one
            # For speed, pick first uncovered
            seed_idx = -1
            for i in range(n_t):
                if uncovered[i]:
                    seed_idx = i
                    break
            if seed_idx == -1:
                break

            # Start with the seed t-subset's points
            block = set(all_t[seed_idx])

            # Greedily add points to maximize coverage
            while len(block) < k:
                if time.time() > time_limit:
                    break
                best_gain = -1
                best_x = None
                for x in range(v):
                    if x in block:
                        continue
                    gain = compute_gain(block, x, uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
                        if rng.random() < 0.3:
                            best_x = x
                if best_x is None:
                    break
                block.add(best_x)

            block = sorted(block)
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                i = t_index[ts]
                if uncovered[i]:
                    uncovered[i] = 0
                    remaining -= 1

        return blocks

    def is_valid(blocks):
        """Check if blocks cover all t-subsets."""
        covered = bytearray([0]) * n_t
        for block in blocks:
            block_set = frozenset(block)
            for ts in itertools.combinations(block, t):
                covered[t_index[ts]] = 1
        for i in range(n_t):
            if not covered[i]:
                return False
        return True

    def coverage_count(blocks):
        """Return number of covered t-subsets."""
        covered = bytearray([0]) * n_t
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered[t_index[ts]] = 1
        return sum(covered)

    def remove_redundant(blocks, time_limit):
        """Remove blocks that are not needed."""
        if not blocks:
            return blocks

        n_blocks = len(blocks)
        # For each t-subset, count how many blocks cover it
        cover_count = [0] * n_t
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[t_index[ts]] += 1

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    break
                # Check if any t-subset is only covered by block i
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    idx = t_index[ts]
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts in itertools.combinations(blocks[i], t):
                        cover_count[t_index[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break  # Restart scan

        return blocks

    def local_search(blocks, time_limit):
        """Try to improve by removing redundant blocks and merging."""
        if not blocks:
            return blocks

        best_blocks = blocks[:]

        # Phase 1: Remove redundant blocks
        best_blocks = remove_redundant(best_blocks, time_limit)

        if time.time() > time_limit:
            return best_blocks

        # Phase 2: Try to replace pairs of blocks with a single block
        improved = True
        while improved and time.time() < time_limit - 5:
            improved = False
            n = len(best_blocks)
            for i in range(n):
                if time.time() > time_limit - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit - 5:
                        break

                    # Find t-subsets that are ONLY covered by block i or block j
                    required = set()
                    bi_set = set(best_blocks[i])
                    bj_set = set(best_blocks[j])

                    for ts in itertools.combinations(best_blocks[i], t):
                        ts_f = frozenset(ts)
                        # Check if covered by any other block
                        found = False
                        for idx, blk in enumerate(best_blocks):
                            if idx == i or idx == j:
                                continue
                            if ts_f.issubset(frozenset(blk)):
                                found = True
                                break
                        if not found:
                            required.add(ts)

                    for ts in itertools.combinations(best_blocks[j], t):
                        ts_f = frozenset(ts)
                        found = False
                        for idx, blk in enumerate(best_blocks):
                            if idx == i or idx == j:
                                continue
                            if ts_f.issubset(frozenset(blk)):
                                found = True
                                break
                        if not found:
                            required.add(ts)

                    if not required:
                        # Both blocks are redundant - should have been caught by remove_redundant
                        # But try removing both
                        candidate = best_blocks[:i] + best_blocks[i+1:j] + best_blocks[j+1:]
                        if is_valid(candidate):
                            best_blocks = candidate
                            improved = True
                            break

                    # Get required points
                    req_points = set()
                    for ts in required:
                        req_points.update(ts)

                    if len(req_points) > k:
                        continue

                    # Try to build a block containing req_points
                    new_block = set(req_points)
                    # Try a few random completions
                    for attempt in range(10):
                        if time.time() > time_limit - 5:
                            break
                        trial = set(new_block)
                        while len(trial) < k:
                            candidates = [x for x in range(v) if x not in trial]
                            if not candidates:
                                break
                            x = random.Random(hash((i, j, attempt, len(trial)))).choice(candidates)
                            trial.add(x)

                        # Check if trial covers all required t-subsets
                        ok = True
                        for ts in required:
                            if not frozenset(ts).issubset(frozenset(trial)):
                                ok = False
                                break
                        if ok:
                            candidate = best_blocks[:i] + best_blocks[i+1:j] + best_blocks[j+1:] + [sorted(trial)]
                            if is_valid(candidate):
                                best_blocks = candidate
                                improved = True
                                break

                    if improved:
                        break

        return best_blocks

    # Main: run multiple randomized greedy, keep best
    best_result = None
    best_len = float('inf')

    # Run several randomized greedy attempts
    num_trials = 8
    for trial in range(num_trials):
        if time.time() > deadline - 25:
            break
        rng = random.Random(trial * 7919 + 42)
        result = greedy_cover(rng, deadline - 20)
        if is_valid(result):
            if len(result) < best_len:
                best_len = len(result)
                best_result = result

    # Local search on best
    if best_result is not None and time.time() < deadline - 10:
        improved = local_search(best_result, deadline - 5)
        if is_valid(improved):
            best_result = improved

    if best_result is None:
        # Fallback: pure greedy with fixed seed
        rng = random.Random(0)
        best_result = greedy_cover(rng, deadline)

    return best_result
