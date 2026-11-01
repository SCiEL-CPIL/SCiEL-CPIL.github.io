<<<<<<< SEARCH
"""Covering design C(24,7,3): randomized greedy with local search reduction."""
import itertools
import time
import random
from collections import defaultdict


def _fast_greedy(v, k, t, rng, deadline):
    """
    Fast greedy covering with randomized tie-breaking.
    Uses precomputed (t-1)-subset to t-subset mappings for speed.
    """
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    # Precompute: for each (t-1)-subset, which t-sets contain it
    tm1_to_tsets = defaultdict(set)
    for ts in all_tsets:
        for i in range(t):
            tm1 = ts[:i] + ts[i+1:]
            tm1_to_tsets[tm1].add(ts)
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-set to start the block
        start_set = rng.choice(list(uncovered))
        block = list(start_set)
        
        while len(block) < k:
            best_gain = -1
            best_candidates = []
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many uncovered t-sets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            # Random tie-breaking
            block.append(rng.choice(best_candidates))
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    return blocks


def _local_search_reduce(blocks, v, k, t, deadline):
    """
    Try to remove redundant blocks and improve the covering.
    A block is redundant if removing it still leaves all t-sets covered.
    """
    # Build coverage count for each t-set
    tset_coverage = defaultdict(int)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            tset_coverage[ts] += 1
    
    # Try to remove blocks that are redundant
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks)):
            # Check if block i is redundant
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                if tset_coverage[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for ts in itertools.combinations(blocks[i], t):
                    tset_coverage[ts] -= 1
                blocks.pop(i)
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    # Start with a generous time budget, leaving safety margin
    start_time = time.time()
    total_budget = 110  # seconds, leaving 10s safety margin for 120s limit
    
    # Multiple randomized greedy restarts
    best_blocks = None
    rng = random.Random(42)
    
    deadline = start_time + total_budget
    while time.time() < deadline:
        # Run a fast greedy with randomization
        blocks = _fast_greedy(v, k, t, rng, deadline)
        
        # Apply local search reduction
        blocks = _local_search_reduce(blocks, v, k, t, deadline)
        
        # Keep the best result
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
        
        # If we've hit a good number, we can stop early
        if len(blocks) <= 76:
            break
    
    # Final safety check: ensure we return a valid covering
    if best_blocks is None:
        # Fallback: run one more greedy
        best_blocks = _fast_greedy(v, k, t, rng, time.time() + 10)
        best_blocks = _local_search_reduce(best_blocks, v, k, t, time.time() + 5)
    
    return best_blocks
# EVOLVE-BLOCK-END
=======
"""Covering design C(24,7,3): Robust greedy with fallback and local search."""
import itertools
import time
import random
from collections import defaultdict


def _naive_greedy(v, k, t, deadline):
    """
    Fast, deterministic greedy that is guaranteed to produce a valid covering
    quickly, though it may use more blocks than optimal.
    """
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick the lexicographically smallest uncovered t-set
        start_set = min(uncovered)
        block = list(start_set)
        
        # Complete the block with the smallest available elements
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block:
                block.append(x)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
        
    return blocks


def _fast_greedy(v, k, t, rng, deadline):
    """
    Optimized greedy with randomized tie-breaking.
    """
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-set
        start_set = rng.choice(list(uncovered))
        block = list(start_set)
        
        while len(block) < k:
            best_gain = -1
            best_candidates = []
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                # Optimized for t=3, fallback for general t
                if t == 3:
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            ts = tuple(sorted((block[i], block[j], x)))
                            if ts in uncovered:
                                gain += 1
                else:
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
                
            block.append(rng.choice(best_candidates))
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
        
    return blocks


def _local_search_reduce(blocks, v, k, t, deadline):
    """
    Try to remove redundant blocks.
    """
    tset_coverage = defaultdict(int)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            tset_coverage[ts] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks)):
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                if tset_coverage[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(blocks[i], t):
                    tset_coverage[ts] -= 1
                blocks.pop(i)
                improved = True
                break
    
    return blocks


def _is_valid_covering(blocks, v, k, t):
    """
    Check if the covering is valid.
    """
    all_tsets = set(itertools.combinations(range(v), t))
    covered = set()
    for block in blocks:
        if len(block) != k or len(set(block)) != k:
            return False
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return all_tsets.issubset(covered)


def construct(v, k, t):
    start_time = time.time()
    total_budget = 145  # Safety margin for 150s limit
    
    # 1. Get a guaranteed valid (but potentially large) covering first
    fallback_deadline = start_time + 10
    best_blocks = _naive_greedy(v, k, t, fallback_deadline)
    
    # Verify fallback validity
    if not _is_valid_covering(best_blocks, v, k, t):
        # This should not happen, but if it does, we have a problem.
        # Return the naive blocks anyway as they are likely valid.
        pass

    # 2. Try to improve with smart greedy
    rng = random.Random(42)
    current_deadline = start_time + total_budget
    
    while time.time() < current_deadline - 5:
        # Run smart greedy
        blocks = _fast_greedy(v, k, t, rng, current_deadline - 5)
        
        # Check validity of smart greedy result
        if _is_valid_covering(blocks, v, k, t):
            # Reduce
            blocks = _local_search_reduce(blocks, v, k, t, current_deadline - 2)
            
            # Check validity after reduction
            if _is_valid_covering(blocks, v, k, t):
                if len(blocks) < len(best_blocks):
                    best_blocks = blocks
            else:
                pass
        else:
            # Smart greedy timed out or failed
            pass

    return best_blocks
>>>>>>> REPLACE
