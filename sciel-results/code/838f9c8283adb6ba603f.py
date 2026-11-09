"""Stronger bootstrap SEARCH for covering design C(v,k,t).
Target: v=24, k=7, t=3 (but generic enough).

Key improvements over the previous version:
- Bitmask representation for t-subsets for O(1) membership / counting.
- Greedy seed that chooses blocks to maximize covered t-subsets.
- Post-processing: greedy removal of redundant blocks.
- Local search: swap a block for a better one; hill-climb to reduce count.
- Portfolio of randomized restarts with diverse point orderings.
- Strict wall-clock deadline with safety margin.

Score: minimize number of blocks.
"""

import time
import itertools
import random
import sys

# ---- deadline management ----
_DEADLINE = None
_TIME_LIMIT = 120
_MARGIN = 6  # safety margin before hard kill

def _deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    return _DEADLINE

def _time_up():
    return time.time() >= _DEADLINE

# ---- bitmask helpers ----
# For a given (v, t), map each t-subset to an index 0..C(v,t)-1
# We'll build this lazily and cache it.
_TSUBSET_CACHE = {}

def _tsubset_map(v, t):
    """Return dict: frozenset -> index, and reverse list."""
    key = (v, t)
    if key in _TSUBSET_CACHE:
        return _TSUBSET_CACHE[key]
    all_ts = list(itertools.combinations(range(v), t))
    fwd = {frozenset(ts): i for i, ts in enumerate(all_ts)}
    _TSUBSET_CACHE[key] = (fwd, all_ts)
    return fwd, all_ts

def _tsubset_indices(block, t, fwd):
    """Return list of indices for all t-subsets of block."""
    return [fwd[frozenset(ts)] for ts in itertools.combinations(block, t)]

def _count_tsubsets(blocks, v, t, fwd):
    """Return array of counts: how many blocks cover each t-subset."""
    n = len(fwd)
    counts = [0] * n
    for b in blocks:
        for idx in _tsubset_indices(b, t, fwd):
            counts[idx] += 1
    return counts

def _is_valid(v, k, t, blocks):
    """Check validity using bitmask counting."""
    if v < t:
        return True
    fwd, all_ts = _tsubset_map(v, t)
    counts = [0] * len(fwd)
    for b in blocks:
        if len(b) != k or len(set(b)) != k:
            return False
        for x in b:
            if x < 0 or x >= v:
                return False
        for idx in _tsubset_indices(b, t, fwd):
            counts[idx] += 1
    return all(c >= 1 for c in counts)

def _dedupe(blocks):
    seen = set()
    out = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out

# ---- greedy seed ----
def _greedy_seed(v, k, t, rng=None, time_fn=None):
    """Greedy: repeatedly pick the block that covers the most uncovered t-subsets."""
    if rng is None:
        rng = random.Random()
    fwd, all_ts = _tsubset_map(v, t)
    n_ts = len(fwd)
    
    # uncovered: bitset as a set of indices
    uncovered = set(range(n_ts))
    blocks = []
    
    while uncovered:
        if time_fn is not None and time_fn():
            break
        
        # For each candidate block, compute how many uncovered t-subsets it covers
        # This is expensive; for v=24,k=7 there are C(24,7)=346104 blocks.
        # Instead, use a smarter approach: pick a random uncovered t-subset,
        # then try to extend it to a k-subset that covers many uncovered t-subsets.
        
        # Randomly pick an uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = list(all_ts[seed_idx])
        
        # Greedily extend seed to size k
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time_fn is not None and time_fn():
                break
            best_x = -1
            best_gain = -1
            # Precompute (t-1)-subsets of current block
            if t > 1:
                block_t1 = list(itertools.combinations(block, t - 1))
            else:
                block_t1 = [()]
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_t1:
                    combo = frozenset(rest + (x,))
                    if len(combo) == t:  # should always be true if rest has t-1 elements
                        idx = fwd.get(combo, -1)
                        if idx != -1 and idx in uncovered:
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
            for idx in _tsubset_indices(block, t, fwd):
                uncovered.discard(idx)
    
    return blocks

# ---- remove redundant blocks ----
def _remove_redundant(v, k, t, blocks, time_fn=None):
    """Remove blocks whose t-subsets are all covered by other blocks."""
    if not blocks:
        return blocks
    
    fwd, _ = _tsubset_map(v, t)
    counts = [0] * len(fwd)
    for b in blocks:
        for idx in _tsubset_indices(b, t, fwd):
            counts[idx] += 1
    
    remaining = list(blocks)
    changed = True
    while changed:
        changed = False
        if time_fn is not None and time_fn():
            break
        for i in range(len(remaining)):
            if time_fn is not None and time_fn():
                break
            b = remaining[i]
            removable = True
            for idx in _tsubset_indices(b, t, fwd):
                if counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                for idx in _tsubset_indices(b, t, fwd):
                    counts[idx] -= 1
                remaining.pop(i)
                changed = True
                break
    
    return remaining

# ---- local search: try to replace a block with a better one, or remove via swap ----
def _local_search(v, k, t, blocks, rng, time_fn=None):
    """Hill-climb: try swaps and removals to reduce block count."""
    if not blocks:
        return blocks
    
    fwd, _ = _tsubset_map(v, t)
    current = [list(b) for b in blocks]
    counts = [0] * len(fwd)
    for b in current:
        for idx in _tsubset_indices(b, t, fwd):
            counts[idx] += 1
    
    best = list(current)
    best_len = len(best)
    
    # Iterative improvement
    improved = True
    while improved:
        improved = False
        if time_fn is not None and time_fn():
            break
        
        # Try removing each block
        for i in range(len(current)):
            if time_fn is not None and time_fn():
                break
            b = current[i]
            removable = True
            for idx in _tsubset_indices(b, t, fwd):
                if counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                for idx in _tsubset_indices(b, t, fwd):
                    counts[idx] -= 1
                current.pop(i)
                best = list(current)
                best_len = len(best)
                improved = True
                break
        
        if improved or (time_fn is not None and time_fn()):
            continue
        
        # Try swapping: replace block i with a new block that covers more
        # Only useful if we can then remove another block, but that's complex.
        # Instead, do a limited random perturbation: replace one block with a
        # random block covering a random uncovered t-subset, then re-remove.
        
        # Find t-subsets with count == 1 (critical)
        critical = [idx for idx, c in enumerate(counts) if c == 1]
        if not critical:
            break
        
        # Pick a random critical t-subset, find a block covering it, try to replace
        # Actually, for now just do random perturbation + removal
        if len(current) > 1:
            idx_remove = rng.randrange(len(current))
            old_block = current[idx_remove]
            old_idxs = _tsubset_indices(old_block, t, fwd)
            for idx in old_idxs:
                counts[idx] -= 1
            
            # Generate a new block covering a random uncovered t-subset
            # Pick random uncovered t-subset
            uncovered_idxs = [i for i, c in enumerate(counts) if c == 0]
            if uncovered_idxs:
                seed_idx = rng.choice(uncovered_idxs)
                seed = list(itertools.combinations(range(v), t)[seed_idx])
                # Extend to size k
                new_block = list(seed)
                new_set = set(new_block)
                while len(new_block) < k:
                    candidates = [x for x in range(v) if x not in new_set]
                    if not candidates:
                        break
                    x = rng.choice(candidates)
                    new_block.append(x)
                    new_set.add(x)
                if len(new_block) == k:
                    new_block.sort()
                    new_idxs = _tsubset_indices(new_block, t, fwd)
                    for idx in new_idxs:
                        counts[idx] += 1
                    current[idx_remove] = new_block
                    # Now try to remove redundant
                    removed = _remove_redundant(v, k, t, current, time_fn)
                    if len(removed) < len(current):
                        # Recompute counts
                        counts = [0] * len(fwd)
                        for b in removed:
                            for idx in _tsubset_indices(b, t, fwd):
                                counts[idx] += 1
                        current = removed
                        if len(current) < best_len:
                            best = list(current)
                            best_len = len(current)
                            improved = True
                    else:
                        # Revert
                        for idx in new_idxs:
                            counts[idx] -= 1
                        for idx in old_idxs:
                            counts[idx] += 1
                        current[idx_remove] = old_block
    
    return best

def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    
    def time_fn():
        return _time_up()
    
    # Edge cases
    if v < t:
        return []
    if k > v:
        k = v
    if t > k:
        return []
    
    # Initial greedy seed
    rng0 = random.Random(42)
    best_blocks = _greedy_seed(v, k, t, rng=rng0, time_fn=time_fn)
    best_blocks = _dedupe(best_blocks)
    best_blocks = _remove_redundant(v, k, t, best_blocks, time_fn=time_fn)
    
    if not _is_valid(v, k, t, best_blocks):
        # Fallback
        best_blocks = _greedy_seed(v, k, t, rng=rng0, time_fn=time_fn)
        best_blocks = _dedupe(best_blocks)
    
    best_len = len(best_blocks)
    
    # Portfolio: randomized restarts
    num_trials = 100
    for trial in range(num_trials):
        if time_fn():
            break
        seed_val = 1000 + trial
        rng = random.Random(seed_val)
        
        candidate = _greedy_seed(v, k, t, rng=rng, time_fn=time_fn)
        candidate = _dedupe(candidate)
        candidate = _remove_redundant(v, k, t, candidate, time_fn=time_fn)
        
        if not _is_valid(v, k, t, candidate):
            continue
        
        # Local search
        candidate = _local_search(v, k, t, candidate, rng, time_fn=time_fn)
        
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_blocks = candidate
    
    # Final cleanup
    best_blocks = _remove_redundant(v, k, t, best_blocks, time_fn=time_fn)
    
    return best_blocks
