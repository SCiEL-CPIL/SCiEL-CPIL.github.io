"""Max-SAT style local search over block-selection for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _generate_pool(v, k, n_blocks, rng):
    """Generate a pool of random k-subsets of {0,...,v-1}."""
    pool = set()
    points = list(range(v))
    attempts = 0
    max_attempts = n_blocks * 50
    while len(pool) < n_blocks and attempts < max_attempts:
        attempts += 1
        block = tuple(sorted(rng.sample(points, k)))
        pool.add(block)
    return list(pool)


def _count_uncovered(selected, pool, v, t):
    """Return set of uncovered t-subsets given selected block indices."""
    covered = set()
    for bi in selected:
        block = pool[bi]
        for sub in itertools.combinations(block, t):
            covered.add(sub)
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs - covered


def _walksat(v, k, t, pool, deadline, rng):
    """WalkSAT local search: flip block selections to maximize coverage."""
    n_pool = len(pool)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Precompute: for each block, which t-subsets it covers (as indices into all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    block_tsubs = []
    for block in pool:
        covered = []
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx.get(sub)
            if idx is not None:
                covered.append(idx)
        block_tsubs.append(covered)
    
    # Initial state: select blocks greedily (by coverage of random uncovered t-subsets)
    uncovered_count = [1] * n_tsubs  # 1 = uncovered
    selected = set()
    
    # Greedy initialization: repeatedly pick the block covering most uncovered t-subsets
    while any(uncovered_count):
        if time.time() > deadline:
            break
        best_block = -1
        best_gain = 0
        for bi in range(n_pool):
            if bi in selected:
                continue
            gain = sum(1 for idx in block_tsubs[bi] if uncovered_count[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        if best_block == -1:
            break
        selected.add(best_block)
        for idx in block_tsubs[best_block]:
            if uncovered_count[idx] == 0:
                uncovered_count[idx] = 1
    
    # WalkSAT: random flips with acceptance if coverage doesn't decrease
    total_steps = 0
    max_steps = 500000
    while total_steps < max_steps:
        if time.time() > deadline:
            break
        total_steps += 1
        
        # Pick a random block to flip
        bi = rng.randint(0, n_pool - 1)
        
        # Compute effect of flipping
        if bi in selected:
            # Removing: count how many t-subsets become uncovered
            newly_uncovered = 0
            for idx in block_tsubs[bi]:
                if uncovered_count[idx] == 1:
                    # Check if any other selected block covers it
                    covered_by_other = False
                    for other_bi in selected:
                        if other_bi == bi:
                            continue
                        if idx in block_tsubs[other_bi]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        newly_uncovered += 1
            # Accept removal if it doesn't increase uncovered count
            if newly_uncovered == 0:
                selected.remove(bi)
                for idx in block_tsubs[bi]:
                    uncovered_count[idx] = 0
        else:
            # Adding: count how many t-subsets become covered
            newly_covered = 0
            for idx in block_tsubs[bi]:
                if uncovered_count[idx] == 0:
                    newly_covered += 1
            # Accept addition if it improves coverage
            if newly_covered > 0:
                selected.add(bi)
                for idx in block_tsubs[bi]:
                    uncovered_count[idx] = 1
    
    # Convert selected indices to blocks
    return [list(pool[bi]) for bi in sorted(selected)]


def _delete_redundant(blocks, v, t, deadline):
    """Remove blocks whose removal still leaves all t-subsets covered."""
    if not blocks:
        return blocks
    
    # Build coverage count
    cover_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1
    
    # Iteratively remove removable blocks
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_blocks = []
        for block in blocks:
            removable = True
            for sub in itertools.combinations(block, t):
                if cover_count[sub] <= 1:
                    removable = False
                    break
            if removable:
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                changed = True
            else:
                new_blocks.append(block)
        blocks = new_blocks
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = random.Random(42)
    
    # Generate a pool of candidate blocks
    # Pool size: large enough for good coverage, but not too large for WalkSAT
    pool_size = min(5000, int(v ** 2 * 5))
    pool = _generate_pool(v, k, pool_size, rng)
    
    # Run WalkSAT local search
    result = _walksat(v, k, t, pool, deadline - 10.0, rng)
    
    # If WalkSAT didn't finish or returned empty, fall back to greedy
    if not result:
        # Simple greedy fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time.time() < deadline - 5.0:
            seed = rng.choice(list(uncovered))
            block = list(seed)
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)
                if best_gain <= 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                    else:
                        break
                else:
                    block.append(rng.choice(best_candidates))
            block.sort()
            blocks.append(block)
            covered = set(itertools.combinations(block, t))
            uncovered.difference_update(covered)
        result = blocks
    
    # Deletion phase
    if result and time.time() < deadline - 2.0:
        result = _delete_redundant(result, v, t, deadline - 2.0)
    
    return result
# EVOLVE-BLOCK-END
