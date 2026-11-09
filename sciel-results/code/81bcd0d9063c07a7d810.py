"""
Rewritten covering design search for C(v, k, t) with focus on speed and
effective local search.

Approach:
1. Fast greedy seed with precomputed t-subset indices.
2. Aggressive block-removal pass (iterative, until stable).
3. Local search with block swaps: try replacing a block with a random
   k-subset that maintains coverage, allowing removal of redundant blocks.
4. Multiple randomized restarts with diverse seeds.
5. Strict wall-clock deadline guard.
"""

import time
import itertools
import random
import sys

_DEADLINE = None
_TIME_LIMIT = 120
_MARGIN = 5


def _check_time():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    return time.time() >= _DEADLINE


def _build_index(v, k, t):
    """Precompute all t-subsets and map them to indices."""
    all_ts = list(itertools.combinations(range(v), t))
    ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}
    return all_ts, ts_to_idx


def _greedy_cover(v, k, t, rng, time_limit_fn=None, all_ts=None, ts_to_idx=None):
    """Fast greedy covering with precomputed indices."""
    if all_ts is None or ts_to_idx is None:
        all_ts, ts_to_idx = _build_index(v, k, t)
    
    uncovered_count = [1] * len(all_ts)
    blocks = []
    
    while any(c > 0 for c in uncovered_count):
        if time_limit_fn is not None and time_limit_fn():
            break
        
        # Find most "constrained" uncovered t-subset (arbitrary pick for speed)
        # For diversity, pick a random uncovered one
        uncovered_indices = [i for i, c in enumerate(uncovered_count) if c > 0]
        if not uncovered_indices:
            break
        seed_idx = rng.choice(uncovered_indices)
        seed_subset = list(all_ts[seed_idx])
        
        block = list(seed_subset)
        block_set = set(block)
        
        while len(block) < k:
            if time_limit_fn is not None and time_limit_fn():
                break
            
            # Precompute (t-1)-subsets of current block
            if t > 1:
                block_t1 = list(itertools.combinations(block, t - 1))
            else:
                block_t1 = [()]
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_t1:
                    combo = tuple(sorted(rest + (x,)))
                    idx = ts_to_idx.get(combo)
                    if idx is not None and uncovered_count[idx] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain == len(block_t1):
                        break
            
            if best_x == -1:
                unused = [x for x in range(v) if x not in block_set]
                if unused:
                    best_x = rng.choice(unused)
                else:
                    break
            
            block.append(best_x)
        
        if len(block) == k:
            block.sort()
            blocks.append(block)
            # Update uncovered counts
            for ts in itertools.combinations(block, t):
                idx = ts_to_idx.get(ts)
                if idx is not None:
                    uncovered_count[idx] = max(0, uncovered_count[idx] - 1)
    
    return blocks


def _remove_redundant_blocks(v, k, t, blocks, all_ts, ts_to_idx, time_limit_fn=None):
    """Iteratively remove redundant blocks."""
    if not blocks:
        return blocks
    
    # Count coverage for each t-subset
    cover_count = [0] * len(all_ts)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = ts_to_idx.get(ts)
            if idx is not None:
                cover_count[idx] += 1
    
    remaining = list(blocks)
    changed = True
    while changed:
        changed = False
        if time_limit_fn is not None and time_limit_fn():
            break
        for i in range(len(remaining)):
            if time_limit_fn is not None and time_limit_fn():
                break
            b = remaining[i]
            removable = True
            for ts in itertools.combinations(b, t):
                idx = ts_to_idx.get(ts)
                if idx is not None and cover_count[idx] <= 1:
                    removable = False
                    break
            if removable:
                for ts in itertools.combinations(b, t):
                    idx = ts_to_idx.get(ts)
                    if idx is not None:
                        cover_count[idx] -= 1
                remaining.pop(i)
                changed = True
                break
    
    return remaining, cover_count


def _local_search(v, k, t, blocks, all_ts, ts_to_idx, rng, time_limit_fn=None, max_iters=5000):
    """Local search: try block swaps and removals."""
    if not blocks:
        return blocks
    
    # First remove redundant
    blocks, cover_count = _remove_redundant_blocks(v, k, t, blocks, all_ts, ts_to_idx, time_limit_fn)
    
    current = list(blocks)
    current_set = set(tuple(sorted(b)) for b in current)
    
    for _ in range(max_iters):
        if time_limit_fn is not None and time_limit_fn():
            break
        
        # Try to remove a random block
        if current:
            idx = rng.randrange(len(current))
            b = current[idx]
            removable = True
            for ts in itertools.combinations(b, t):
                idx_ts = ts_to_idx.get(ts)
                if idx_ts is not None and cover_count[idx_ts] <= 1:
                    removable = False
                    break
            if removable:
                for ts in itertools.combinations(b, t):
                    idx_ts = ts_to_idx.get(ts)
                    if idx_ts is not None:
                        cover_count[idx_ts] -= 1
                current.pop(idx)
                continue
        
        # Try to swap: replace a random block with a random k-subset
        if current and len(current) > 1:
            idx = rng.randrange(len(current))
            old_b = current[idx]
            
            # Generate a random k-subset
            new_b = sorted(rng.sample(range(v), k))
            new_tuple = tuple(new_b)
            
            if new_tuple in current_set:
                continue
            
            # Check if new_b covers any uncovered t-subsets that old_b doesn't
            # and doesn't uncover any t-subsets that only old_b covers
            old_ts_set = set()
            for ts in itertools.combinations(old_b, t):
                idx_ts = ts_to_idx.get(ts)
                if idx_ts is not None:
                    old_ts_set.add(idx_ts)
            
            new_ts_set = set()
            for ts in itertools.combinations(new_b, t):
                idx_ts = ts_to_idx.get(ts)
                if idx_ts is not None:
                    new_ts_set.add(idx_ts)
            
            # Check if swap is valid: all t-subsets of new_b must be covered
            # by at least one other block OR be in old_b's coverage
            valid = True
            for idx_ts in new_ts_set:
                if idx_ts in old_ts_set:
                    # Covered by old_b, so after swap it will be uncovered unless
                    # covered by another block
                    if cover_count[idx_ts] <= 1:
                        valid = False
                        break
                else:
                    # Not covered by old_b, so must be covered by another block
                    if cover_count[idx_ts] <= 0:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Perform swap
            current[idx] = new_b
            current_set.discard(tuple(sorted(old_b)))
            current_set.add(new_tuple)
            
            # Update cover counts
            for idx_ts in old_ts_set:
                cover_count[idx_ts] -= 1
            for idx_ts in new_ts_set:
                cover_count[idx_ts] += 1
            
            # Try to remove redundant blocks after swap
            current, cover_count = _remove_redundant_blocks(v, k, t, current, all_ts, ts_to_idx, time_limit_fn)
    
    return current


def _is_valid_covering(v, k, t, blocks, all_ts=None, ts_to_idx=None):
    """Check if blocks form a valid covering."""
    if all_ts is None or ts_to_idx is None:
        all_ts, ts_to_idx = _build_index(v, k, t)
    
    if not blocks:
        return len(all_ts) == 0
    
    covered = [False] * len(all_ts)
    for b in blocks:
        if len(b) != k:
            return False
        for x in b:
            if x < 0 or x >= v:
                return False
        for ts in itertools.combinations(b, t):
            idx = ts_to_idx.get(ts)
            if idx is not None:
                covered[idx] = True
    
    return all(covered)


def _dedupe_blocks(blocks):
    """Remove duplicate blocks."""
    seen = set()
    result = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            result.append(list(key))
    return result


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    
    def time_limit_fn():
        return time.time() >= _DEADLINE
    
    # Edge cases
    if v < t:
        return []
    if k > v:
        k = v
    if t > k:
        return []
    if t <= 0:
        return []
    
    # Precompute
    all_ts, ts_to_idx = _build_index(v, k, t)
    
    # Initial greedy seed
    rng0 = random.Random(42)
    best_blocks = _greedy_cover(v, k, t, rng=rng0, time_limit_fn=time_limit_fn,
                                 all_ts=all_ts, ts_to_idx=ts_to_idx)
    best_blocks = _dedupe_blocks(best_blocks)
    
    # Local search on seed
    best_blocks = _local_search(v, k, t, best_blocks, all_ts, ts_to_idx, rng0, time_limit_fn)
    
    if _is_valid_covering(v, k, t, best_blocks, all_ts, ts_to_idx):
        best_len = len(best_blocks)
    else:
        # Fallback
        best_blocks = _greedy_cover(v, k, t, rng=rng0, time_limit_fn=time_limit_fn,
                                     all_ts=all_ts, ts_to_idx=ts_to_idx)
        best_blocks = _dedupe_blocks(best_blocks)
        best_len = len(best_blocks)
    
    # Portfolio: multiple randomized instances
    num_trials = 200
    trial_count = 0
    while trial_count < num_trials and not time_limit_fn():
        trial_count += 1
        seed_val = 42 + trial_count
        rng = random.Random(seed_val)
        
        candidate = _greedy_cover(v, k, t, rng=rng, time_limit_fn=time_limit_fn,
                                   all_ts=all_ts, ts_to_idx=ts_to_idx)
        candidate = _dedupe_blocks(candidate)
        
        # Local search
        candidate = _local_search(v, k, t, candidate, all_ts, ts_to_idx, rng, time_limit_fn)
        
        if _is_valid_covering(v, k, t, candidate, all_ts, ts_to_idx):
            if len(candidate) < best_len:
                best_len = len(candidate)
                best_blocks = candidate
    
    return best_blocks
