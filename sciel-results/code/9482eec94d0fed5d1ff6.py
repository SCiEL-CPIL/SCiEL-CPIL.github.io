"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _greedy_cover(v, k, t, rng=None):
    """Greedy set-cover: open a block on the first uncovered t-subset, then add
    the point that covers the most uncovered t-subsets until the block has k points."""
    if rng is None:
        rng = random.Random()
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Pick a random uncovered t-subset as seed (randomization helps)
        seed = rng.choice(list(uncovered))
        block = list(seed)
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            candidates = [x for x in range(v) if x not in block]
            block.append(max(candidates, key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search: remove a block, then greedily
    re-cover any newly-uncovered t-subsets with minimal new blocks. Accept if
    net reduction. Also try swap: replace one block with a better one."""
    blocks = [list(b) for b in blocks]
    n_blocks = len(blocks)
    # Precompute coverage: for each t-subset, which block indices cover it
    all_tsubs = list(itertools.combinations(range(v), t))
    # Map t-subset -> frozenset of block indices that contain it
    coverage = {}
    for bi, block in enumerate(blocks):
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            if ts not in coverage:
                coverage[ts] = set()
            coverage[ts].add(bi)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each block
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Count t-subsets that become uncovered
            newly_uncovered = []
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                cov = coverage.get(ts, set())
                if len(cov) == 1 and bi in cov:
                    newly_uncovered.append(ts)
            if not newly_uncovered:
                # Safe removal
                blocks.pop(bi)
                # Update coverage
                for ts in itertools.combinations(blocks[bi], t):
                    ts = tuple(sorted(ts))
                    if ts in coverage:
                        coverage[ts].discard(bi)
                        if not coverage[ts]:
                            del coverage[ts]
                # Re-index: all blocks after bi shift down by 1
                for ts in coverage:
                    new_set = set()
                    for idx in coverage[ts]:
                        if idx < bi:
                            new_set.add(idx)
                        else:
                            new_set.add(idx - 1)
                    coverage[ts] = new_set
                improved = True
                break
        if improved:
            continue
        # Try swap: remove block bi, add a new block that covers more
        # This is more expensive; only do it if we have time
        if time.time() >= deadline:
            break
        # For now, just do removals. Swap logic would be complex.
        break
    return blocks


def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with randomization, multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # First greedy run
    blocks = _greedy_cover(v, k, t, rng)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try a few more greedy restarts if time permits
    restarts = 0
    while time.time() < deadline - 30 and restarts < 5:
        blocks = _greedy_cover(v, k, t, rng)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count
    if time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, deadline - 2)

    return best_blocks
# EVOLVE-BLOCK-END
