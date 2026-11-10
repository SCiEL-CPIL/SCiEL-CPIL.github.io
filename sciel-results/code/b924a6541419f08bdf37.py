"""Covering design C(19,8,5) - multi-strategy greedy with local search."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using:
    1. Fast greedy with bitmask-based uncovered tracking
    2. Multi-seed exploration
    3. Local search to remove redundant blocks
    4. Block replacement / perturbation
    """
    start_time = time.time()
    deadline = start_time + 112  # safety margin

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute for each point x and each t-subset index: which t-subsets contain x
    # Actually, let's precompute the incidence: for each k-block candidate, which t-subsets it covers
    # That's too many. Instead, precompute for each point x, the list of t-subsets containing x.
    # Better: for greedy, when we add point x to a block B, the new t-subsets covered are
    # those formed by (t-1)-subsets of B plus x.

    # Precompute: for each point x, list of t-subset indices that contain x
    point_to_tsubsets = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubsets):
        for x in ts:
            point_to_tsubsets[x].append(idx)

    def greedy_build(seed):
        """Greedy covering with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = [True] * n_tsub
        uncovered_count = n_tsub
        blocks = []

        while uncovered_count > 0:
            if time.time() > deadline - 3:
                break

            # Pick a random uncovered t-subset to seed the block
            # Find uncovered indices efficiently
            uncovered_indices = [i for i in range(n_tsub) if uncovered[i]]
            if not uncovered_indices:
                break
            chosen = rng.choice(uncovered_indices)
            seed_ts = all_tsubsets[chosen]

            # Build block starting from seed_ts
            block = list(seed_ts)
            rng.shuffle(block)

            while len(block) < k:
                if time.time() > deadline - 3:
                    break

                # Find point x not in block that covers most uncovered t-subsets
                best_gain = -1
                best_x = None
                block_set = set(block)
                # Precompute (t-1)-subsets of current block for gain computation
                block_t1_subsets = list(itertools.combinations(block, t - 1))

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subsets:
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x is None or best_gain <= 0:
                    # Try any point to complete the block
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = rng.choice(remaining)
                    else:
                        break

                block.append(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if idx is not None and uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1

        if uncovered_count == 0:
            return blocks
        return None

    def local_search(blocks, max_time=8):
        """Remove redundant blocks iteratively."""
        start_ls = time.time()

        # Build cover counts
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                ts = tuple(sorted(ts))
                cover_count[ts] += 1

        changed = True
        while changed and (time.time() - start_ls) < max_time:
            changed = False
            new_blocks = []
            for block in blocks:
                is_critical = False
                for ts in itertools.combinations(block, t):
                    ts = tuple(sorted(ts))
                    if cover_count[ts] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    for ts in itertools.combinations(block, t):
                        ts = tuple(sorted(ts))
                        cover_count[ts] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks

        return blocks

    def validate(blocks):
        """Check covering validity."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(tuple(sorted(ts)))
        return len(covered) == n_tsub

    def try_replace(blocks, seed, max_time=5):
        """Try to replace a block with a better one or remove it."""
        start = time.time()
        rng = random.Random(seed)
        n = len(blocks)
        if n == 0:
            return blocks

        # Try removing each block
        for i in range(n):
            if time.time() > start + max_time:
                break
            candidate = blocks[:i] + blocks[i+1:]
            if validate(candidate):
                return candidate

        # Try replacing a block
        for i in range(n):
            if time.time() > start + max_time:
                break
            # Remove block i, find what's uncovered
            without = blocks[:i] + blocks[i+1:]
            covered = set()
            for block in without:
                for ts in itertools.combinations(block, t):
                    covered.add(tuple(sorted(ts)))

            uncovered = [ts for ts in all_tsubsets if ts not in covered]
            if not uncovered:
                return without

            # Try to find a single block that covers all uncovered (unlikely) or most
            best_block = None
            best_gain = -1
            for ts in uncovered[:50]:
                block = list(ts)
                while len(block) < k:
                    block_set = set(block)
                    best_x = None
                    best_x_gain = -1
                    block_t1 = list(itertools.combinations(block, t-1))
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = sum(1 for rest in block_t1
                                   if tuple(sorted(rest + (x,))) in set(uncovered))
                        if gain > best_x_gain:
                            best_x_gain = gain
                            best_x = x
                    if best_x is None:
                        remaining = [x for x in range(v) if x not in block_set]
                        if not remaining:
                            break
                        best_x = rng.choice(remaining)
                    block.append(best_x)
                block.sort()
                gain = sum(1 for ts2 in itertools.combinations(block, t)
                           if tuple(sorted(ts2)) in set(uncovered))
                if gain > best_gain:
                    best_gain = gain
                    best_block = block

            if best_block is not None:
                new_blocks = without + [best_block]
                if validate(new_blocks) and len(new_blocks) < n:
                    return new_blocks

        return blocks

    # Main search loop
    best_result = None
    best_len = float('inf')

    # Phase 1: Multi-seed greedy + local search
    for seed in range(30):
        if time.time() > deadline - 20:
            break
        result = greedy_build(seed)
        if result is not None:
            result = local_search(result, max_time=6)
            if validate(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    # Phase 2: Try to improve best_result with replacement
    if best_result is not None:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            improved = try_replace(best_result, seed=100 + attempt, max_time=3)
            if improved is not None and len(improved) < best_len:
                best_len = len(improved)
                best_result = improved
                # Re-run local search
                best_result = local_search(best_result, max_time=4)
                if validate(best_result) and len(best_result) < best_len:
                    best_len = len(best_result)

    # Phase 3: More greedy seeds if time permits
    for seed in range(30, 60):
        if time.time() > deadline - 5:
            break
        result = greedy_build(seed)
        if result is not None:
            result = local_search(result, max_time=3)
            if validate(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    if best_result is not None:
        return best_result

    # Fallback
    return greedy_build(0) or []
