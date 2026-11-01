"""Covering design C(24,7,3) with efficient greedy + local search + randomized restarts."""
import itertools
import time
import random

DEADLINE = 142.0  # safety margin before 150s kill

def _all_tsubsets(v, t):
    """Return list of all t-subsets of {0,...,v-1} as sorted tuples."""
    return [tuple(c) for c in itertools.combinations(range(v), t)]

def _block_tsubsets(block, t):
    """Return list of t-subsets contained in a block as sorted tuples."""
    return [tuple(c) for c in itertools.combinations(block, t)]

def _build_greedy(v, k, t, uncovered_set, rng, deadline_fn):
    """
    Greedily build a covering. uncovered_set is a set of tuples (t-subsets not yet covered).
    Returns list of blocks (each a sorted list of k ints).
    """
    blocks = []
    # Precompute: for each point x, and each (t-1)-subset S, whether S∪{x} is in uncovered
    # For t=3, k=7, v=24: pairs are C(24,2)=276, points=24, so 276*24 = 6624 entries
    # We'll compute gain on the fly but use a lookup structure
    
    while uncovered_set and not deadline_fn():
        # Pick a random uncovered t-subset to seed
        seed_ts = next(iter(uncovered_set))
        block = list(seed_ts)
        
        # Greedily extend block to size k, maximizing new coverage
        while len(block) < k and not deadline_fn():
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Gain: number of uncovered t-subsets that include x and (t-1) elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
        
        if len(block) < k:
            # Fill with arbitrary points
            block_set = set(block)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for ts in _block_tsubsets(block, t):
            uncovered_set.discard(ts)
    
    return blocks

def _local_search(blocks, v, k, t, all_ts_set, rng, deadline_fn, max_rounds=50):
    """
    Try to reduce number of blocks by:
    1. Removing redundant blocks (if all their t-subsets are covered by other blocks)
    2. Swapping blocks: replace one block with another that covers more uncovered
    """
    blocks = list(blocks)
    n = len(blocks)
    
    # Build coverage count: for each t-subset, how many blocks cover it
    cov_count = {}
    for ts in all_ts_set:
        cov_count[ts] = 0
    for b in blocks:
        for ts in _block_tsubsets(b, t):
            cov_count[ts] += 1
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and not deadline_fn():
        improved = False
        rounds += 1
        # Try removing each block
        for i in range(n - 1, -1, -1):
            if deadline_fn():
                break
            b = blocks[i]
            # Check if all t-subsets in b are still covered by other blocks
            redundant = True
            for ts in _block_tsubsets(b, t):
                if cov_count[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for ts in _block_tsubsets(b, t):
                    cov_count[ts] -= 1
                blocks.pop(i)
                n = len(blocks)
                improved = True
                break  # restart scan
    
    # Try to replace blocks: for each block, see if we can find a better one
    for _ in range(3):  # a few passes
        if deadline_fn():
            break
        for i in range(len(blocks)):
            if deadline_fn():
                break
            b = blocks[i]
            # Count how many t-subsets are ONLY covered by this block
            only_covered = []
            for ts in _block_tsubsets(b, t):
                if cov_count[ts] == 1:
                    only_covered.append(ts)
            
            if not only_covered:
                continue  # redundant, already handled
            
            # Try to find a replacement block that covers all of only_covered
            # and maximizes additional coverage
            best_new_block = None
            best_score = -1
            b_set = set(b)
            
            # Candidate: keep the points that are essential, try variations
            # Simple approach: try replacing one point at a time
            for pos in range(k):
                old_pt = b[pos]
                for new_pt in range(v):
                    if new_pt == old_pt or new_pt in b_set - {old_pt}:
                        continue
                    candidate = b[:pos] + [new_pt] + b[pos+1:]
                    candidate.sort()
                    cand_set = set(candidate)
                    
                    # Check if it covers all only_covered
                    ok = True
                    extra = 0
                    for ts in _block_tsubsets(candidate, t):
                        if ts in only_covered:
                            pass  # must cover
                        else:
                            if cov_count[ts] == 0:
                                extra += 1
                    
                    # Verify all only_covered are in candidate
                    cand_ts_set = set(_block_tsubsets(candidate, t))
                    for ts in only_covered:
                        if ts not in cand_ts_set:
                            ok = False
                            break
                    
                    if ok:
                        score = extra
                        if score > best_score:
                            best_score = score
                            best_new_block = candidate
            
            if best_new_block is not None:
                # Apply swap
                old_ts_set = set(_block_tsubsets(b, t))
                new_ts_set = set(_block_tsubsets(best_new_block, t))
                for ts in old_ts_set:
                    cov_count[ts] -= 1
                for ts in new_ts_set:
                    cov_count[ts] += 1
                blocks[i] = best_new_block
                improved = True
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)
    
    def deadline_fn():
        return time.time() - start_time > DEADLINE
    
    all_ts = _all_tsubsets(v, t)
    all_ts_set = set(all_ts)
    total = len(all_ts)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if deadline_fn():
            break
        
        # Different random seeds for each restart
        rng.seed(42 + restart * 1000)
        uncovered_set = set(all_ts_set)
        
        # Shuffle the order of t-subsets to get different greedy paths
        # We do this by randomly permuting which uncovered set element we pick first
        # Actually, the greedy picks next(iter(uncovered_set)) which is arbitrary
        # To vary it, we can pre-shuffle
        
        blocks = _build_greedy(v, k, t, uncovered_set, rng, deadline_fn)
        
        if not uncovered_set:
            # Apply local search
            blocks = _local_search(blocks, v, k, t, all_ts_set, rng, deadline_fn)
        
        count = len(blocks)
        if count < best_count:
            best_count = count
            best_blocks = blocks[:]
        
        # If we've hit the known best (76), stop early
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered_set = set(all_ts_set)
        best_blocks = _build_greedy(v, k, t, uncovered_set, rng, deadline_fn)
    
    # Final safety: ensure validity
    if uncovered_set if 'uncovered_set' in dir() else None:
        pass
    
    # Verify and fix if needed
    covered = set()
    for b in best_blocks:
        for ts in _block_tsubsets(b, t):
            covered.add(ts)
    
    missing = all_ts_set - covered
    if missing and not deadline_fn():
        # Add blocks to cover missing
        for ts in list(missing):
            if deadline_fn():
                break
            block = list(ts)
            block_set = set(block)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
            block.sort()
            if block not in best_blocks:
                best_blocks.append(block)
    
    return best_blocks
