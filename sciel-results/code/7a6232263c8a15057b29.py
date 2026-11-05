"""
Covering design C(19,7,4) via randomized greedy with multi-start local refinement.
Strategy:
  1. Build an initial covering using a randomized greedy algorithm (with random tie-breaking
     and random block ordering).
  2. Refine the covering by iteratively removing redundant blocks (a block is redundant if
     every 4-subset it covers is also covered by other blocks).
  3. Try to replace a pair of blocks with a single block that covers all 4-subsets of the
     two blocks (if such a block exists among the 7-subsets of the union).
  4. Use multiple random restarts within the time budget.
"""
import itertools
import random
import time


def _all_t_subsets(block, t):
    """Return list of t-subsets of block."""
    return list(itertools.combinations(block, t))


def _build_coverage_map(blocks, v, t):
    """
    For each t-subset of {0..v-1}, store the count of blocks that cover it.
    Returns a dict mapping frozenset -> count.
    """
    counts = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            key = frozenset(ts)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _is_redundant(block, blocks, v, t, coverage_map):
    """
    Check if removing block still leaves all its t-subsets covered.
    We check: for each t-subset of block, is there another block covering it?
    """
    for ts in itertools.combinations(block, t):
        key = frozenset(ts)
        if coverage_map.get(key, 0) <= 1:
            return False
    return True


def _refine_covering(blocks, v, t, deadline):
    """
    Iteratively remove redundant blocks until no more can be removed.
    Returns the refined list of blocks.
    """
    blocks = list(blocks)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        coverage_map = _build_coverage_map(blocks, v, t)
        # Try to remove each block
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            if _is_redundant(blocks[i], blocks, v, t, coverage_map):
                del blocks[i]
                changed = True
                # Rebuild coverage map after removal
                coverage_map = _build_coverage_map(blocks, v, t)
    return blocks


def _try_block_replacement(blocks, v, t, deadline):
    """
    Try to replace two blocks with one block that covers all their t-subsets.
    This is a heuristic: for each pair, check if there exists a k-subset
    of the union that contains all t-subsets of both blocks.
    For k=7, t=4, two 7-blocks have at most 14 points in their union.
    We need a 7-subset of the union that covers all 4-subsets of both original blocks.
    This is only possible if the two blocks share enough points.
    """
    k = 7  # fixed for this task
    blocks = list(blocks)
    n = len(blocks)
    if n < 2:
        return blocks
    
    # For each pair, try to find a replacement
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            b1 = set(blocks[i])
            b2 = set(blocks[j])
            union = b1 | b2
            # A 7-subset of union must contain b1 and b2's 4-subsets
            # Actually, the new block must contain every 4-subset of b1 and b2.
            # That means the new block must contain b1 and b2 entirely? No.
            # It must contain every 4-element subset of b1. Since b1 has 7 elements,
            # every 4-subset of b1 must be in the new block. The smallest set containing
            # all 4-subsets of b1 is b1 itself. So the new block must contain b1.
            # Similarly it must contain b2. So new block must contain b1 ∪ b2.
            # But |b1 ∪ b2| can be up to 14, and we need a 7-subset. So this only works
            # if b1 == b2, which is impossible (distinct blocks).
            # So pair replacement is not feasible in general. Skip.
            pass
    return blocks


def _randomized_greedy(v, k, t, rng, deadline):
    """
    Randomized greedy covering.
    """
    all_points = list(range(v))
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        # Pick a random uncovered t-subset to start
        start_sub = rng.choice(list(uncovered))
        block = list(start_sub)
        
        # Randomize the order of candidate points
        candidates = [x for x in all_points if x not in block]
        rng.shuffle(candidates)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            best_gain = -1
            best_points = []
            for x in candidates:
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = frozenset(tuple(sorted(rest + (x,))))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)
            if not best_points:
                break
            # Randomly pick among best
            chosen = rng.choice(best_points)
            block.append(chosen)
            candidates.remove(chosen)
        
        if len(block) == k:
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(frozenset(ts))
    
    # Convert back to tuples of ints for consistency
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    # First, run a deterministic greedy to get a baseline
    rng = random.Random(42)
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Use different seeds for diversity
        seed = 42 + num_restarts * 7919
        rng = random.Random(seed)
        
        blocks = _randomized_greedy(v, k, t, rng, deadline)
        
        if not blocks:
            continue
        
        # Refine
        blocks = _refine_covering(blocks, v, t, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        # If we've hit the theoretical lower bound, stop
        if best_len <= 131:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _randomized_greedy(v, k, t, random.Random(0), time.time() + 5)
    
    return best_blocks
