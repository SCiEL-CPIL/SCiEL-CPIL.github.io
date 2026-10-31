"""Covering design C(19,7,4) using structured cyclic family + fast local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

_DEADLINE = 150.0
_TIME_LIMIT = 115.0  # safety margin


def _all_tsubsets(v, t):
    """Return list of all t-subsets as tuples."""
    return list(itertools.combinations(range(v), t))


def _block_covers(block, t):
    """Return set of t-subsets covered by a block."""
    return set(itertools.combinations(sorted(block), t))


def _count_covered(blocks, t, v):
    """Count how many t-subsets are covered by the set of blocks.
    Returns (covered_count, covered_set)."""
    covered = set()
    for b in blocks:
        covered.update(itertools.combinations(sorted(b), t))
    return len(covered), covered


def _greedy_construct(v, k, t, seed=0, time_limit=10.0):
    """Greedy covering: repeatedly pick the block that covers the most uncovered t-subsets."""
    rng = random.Random(seed)
    all_ts = _all_tsubsets(v, t)
    total = len(all_ts)
    uncovered = set(all_ts)
    blocks = []
    start = time.time()

    while uncovered:
        if time.time() - start > time_limit:
            break

        # Find the block that covers the most uncovered t-subsets
        best_block = None
        best_gain = -1
        # Try random candidate blocks
        points = list(range(v))
        rng.shuffle(points)
        # Try a limited number of random k-subsets
        for _ in range(200):
            block = tuple(sorted(rng.sample(points, k)))
            gain = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block

        # Also try a greedy extension from a random seed point
        seed_pt = rng.choice(points)
        block = [seed_pt]
        while len(block) < k:
            best_x = None
            best_g = -1
            for x in points:
                if x in block:
                    continue
                g = sum(1 for rest in itertools.combinations(block, t - 1)
                        if tuple(sorted(rest + (x,))) in uncovered)
                if g > best_g:
                    best_g = g
                    best_x = x
            if best_x is not None:
                block.append(best_x)
            else:
                break
        block = tuple(sorted(block))
        gain = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
        if gain > best_gain:
            best_gain = gain
            best_block = block

        if best_block is None or best_gain <= 0:
            break

        blocks.append(list(best_block))
        for ts in itertools.combinations(sorted(best_block), t):
            uncovered.discard(ts)

    return blocks, len(uncovered)


def _cyclic_family(v, k, base_block, time_limit=5.0):
    """Generate blocks by cyclically shifting a base block modulo v."""
    blocks = []
    seen = set()
    for shift in range(v):
        block = tuple(sorted((x + shift) % v for x in base_block))
        if block not in seen:
            seen.add(block)
            blocks.append(list(block))
    return blocks


def _local_search(blocks, v, k, t, time_limit=60.0, seed=0):
    """Local search: try to replace or add blocks to reduce uncovered t-subsets."""
    rng = random.Random(seed)
    all_ts = _all_tsubsets(v, t)
    covered = set()
    for b in blocks:
        covered.update(itertools.combinations(sorted(b), t))

    uncovered = set(all_ts) - covered
    start = time.time()
    best_blocks = list(blocks)
    best_uncovered = len(uncovered)

    while uncovered and time.time() - start < time_limit:
        # Try to find a block that covers the most uncovered t-subsets
        best_gain = 0
        best_block = None

        # Random candidate blocks
        for _ in range(500):
            block = tuple(sorted(rng.sample(range(v), k)))
            gain = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block

        # Greedy extension from hardest uncovered subset
        if uncovered:
            hard_ts = rng.choice(list(uncovered))
            block = list(hard_ts)
            remaining = [x for x in range(v) if x not in block]
            rng.shuffle(remaining)
            for x in remaining:
                if len(block) >= k:
                    break
                block.append(x)
            block = tuple(sorted(block))
            gain = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_gain <= 0 or best_block is None:
            break

        # Add the block
        blocks.append(list(best_block))
        for ts in itertools.combinations(sorted(best_block), t):
            uncovered.discard(ts)

        if len(uncovered) < best_uncovered:
            best_uncovered = len(uncovered)
            best_blocks = list(blocks)

    # Try to remove redundant blocks
    if time.time() - start < time_limit:
        new_blocks = []
        for b in best_blocks:
            # Check if this block is needed
            test_blocks = [x for x in best_blocks if x != b]
            cov = set()
            for tb in test_blocks:
                cov.update(itertools.combinations(sorted(tb), t))
            if len(cov) >= len(all_ts):
                continue  # redundant, skip
            new_blocks.append(b)
        if len(new_blocks) < len(best_blocks):
            best_blocks = new_blocks

    return best_blocks


def construct(v, k, t):
    """Construct a covering design: every t-subset of {0..v-1} lies in some block of size k."""
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    # Phase 1: Generate initial blocks using multiple strategies
    all_blocks = []
    seen = set()

    def add_block(b):
        b = tuple(sorted(b))
        if b not in seen:
            seen.add(b)
            all_blocks.append(list(b))

    # 1. Cyclic family from random base blocks
    rng = random.Random(42)
    for _ in range(20):
        base = tuple(sorted(rng.sample(range(v), k)))
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            add_block(block)
        if time.time() > deadline - 80:
            break

    # 2. Greedy construction from multiple seeds
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        blocks, unc = _greedy_construct(v, k, t, seed=seed, time_limit=5.0)
        for b in blocks:
            add_block(b)

    # 3. Random blocks
    for _ in range(200):
        if time.time() > deadline - 50:
            break
        block = tuple(sorted(rng.sample(range(v), k)))
        add_block(block)

    # Phase 2: Local search to improve coverage
    if time.time() < deadline - 30:
        remaining_time = min(60.0, deadline - time.time() - 10)
        result = _local_search(all_blocks, v, k, t, time_limit=remaining_time, seed=0)
        all_blocks = result

    # Phase 3: Final check and cleanup - remove redundant blocks
    if time.time() < deadline - 5:
        all_ts = _all_tsubsets(v, t)
        # Remove redundant blocks
        cleaned = []
        for i, b in enumerate(all_blocks):
            # Check if removing this block still covers everything
            test = all_blocks[:i] + all_blocks[i+1:]
            cov = set()
            for tb in test:
                cov.update(itertools.combinations(sorted(tb), t))
            if len(cov) >= len(all_ts):
                continue
            cleaned.append(b)
        if len(cleaned) < len(all_blocks):
            all_blocks = cleaned

    # Ensure validity: if not all t-subsets are covered, add blocks greedily
    if time.time() < deadline - 1:
        all_ts = _all_tsubsets(v, t)
        covered = set()
        for b in all_blocks:
            covered.update(itertools.combinations(sorted(b), t))
        uncovered = set(all_ts) - covered

        if uncovered:
            # Add blocks to cover remaining
            while uncovered and time.time() < deadline - 0.5:
                # Find block covering most uncovered
                best_gain = 0
                best_block = None
                for _ in range(100):
                    block = tuple(sorted(random.sample(range(v), k)))
                    gain = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block

                if best_gain <= 0 or best_block is None:
                    # Fallback: just pick any block containing an uncovered t-subset
                    ts = next(iter(uncovered))
                    block = list(ts)
                    for x in range(v):
                        if x not in block and len(block) < k:
                            block.append(x)
                    best_block = tuple(sorted(block))
                    break

                all_blocks.append(list(best_block))
                for ts in itertools.combinations(sorted(best_block), t):
                    uncovered.discard(ts)

    return all_blocks
# EVOLVE-BLOCK-END
