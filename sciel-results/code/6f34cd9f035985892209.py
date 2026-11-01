"""Improved covering design C(24,7,3) with deadline guard and stronger greedy."""
import itertools
import time
import random

# Deadline guard: kill at 145s to stay within 150s wall-clock limit
DEADLINE = 145.0

def _all_tsubsets(v, t):
    """Return list of all t-subsets of {0,...,v-1} as frozensets."""
    return [frozenset(c) for c in itertools.combinations(range(v), t)]

def _block_tsubsets(block, t):
    """Return list of t-subsets contained in a block."""
    return [frozenset(c) for c in itertools.combinations(block, t)]

def _count_uncovered_in_block(block, uncovered_set, t):
    """Count how many uncovered t-subsets are covered by a block."""
    cnt = 0
    for c in itertools.combinations(block, t):
        if frozenset(c) in uncovered_set:
            cnt += 1
    return cnt

def construct(v, k, t):
    start_time = time.time()
    
    # All t-subsets
    all_ts = _all_tsubsets(v, t)
    total = len(all_ts)
    uncovered = set(all_ts)
    blocks = []
    
    # Precompute for speed: map each (t-1)-subset to the list of points that extend it
    # For t=3, k=7: for each pair (a,b), list all c such that {a,b,c} is a valid t-subset
    # We'll use this to compute gain efficiently
    
    # For greedy block construction: given current partial block B, for candidate point x,
    # gain = number of uncovered t-subsets that contain x and (t-1) elements from B
    # For t=3: gain(x) = number of uncovered pairs {a,b} in B such that {a,b,x} is uncovered
    
    def time_check():
        return time.time() - start_time > DEADLINE - 5.0
    
    # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    # Strategy: start from an uncovered t-subset, greedily extend to k elements
    
    while uncovered and not time_check():
        # Pick a random uncovered t-subset to start (or the first one for determinism)
        # Using random can help escape local optima
        seed_ts = next(iter(uncovered))
        block = list(seed_ts)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k and not time_check():
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: number of uncovered t-subsets that would be covered
                # by adding x to the block
                # A t-subset {x} ∪ S where S is a (t-1)-subset of block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = frozenset(rest + (x,))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            # Fill remaining with arbitrary points not in block
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets from uncovered
        covered = set(_block_tsubsets(block, t))
        uncovered -= covered
    
    # If we ran out of time, we may not have covered everything.
    # The evaluator checks validity, so we must ensure all t-subsets are covered.
    # If uncovered is non-empty, we need to add more blocks.
    # But we're at the deadline... Let's do a final fast pass if there's time.
    
    if uncovered and not time_check():
        # Fast greedy for remaining uncovered
        while uncovered and not time_check():
            seed_ts = next(iter(uncovered))
            block = list(seed_ts)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = frozenset(rest + (x,))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
            block.sort()
            blocks.append(block)
            covered = set(_block_tsubsets(block, t))
            uncovered -= covered
    
    # Final safety: if still uncovered (shouldn't happen if we have time),
    # add blocks to cover remaining t-subsets. This is a last resort.
    if uncovered:
        # For each uncovered t-subset, create a block containing it
        for ts in uncovered:
            block = list(ts)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
            block.sort()
            if block not in blocks:
                blocks.append(block)
    
    return blocks
