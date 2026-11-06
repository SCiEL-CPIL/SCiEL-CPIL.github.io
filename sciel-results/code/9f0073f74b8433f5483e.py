"""Optimized covering design C(19,8,5) via randomized greedy with bitmask acceleration."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

# Precompute combinations for the specific case v=19, k=8, t=5
# But we'll make it general enough to work for any v, k, t passed in

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 120s
    
    # For the specific case v=19, k=8, t=5:
    # C(19,5) = 11628 t-subsets
    # C(19,8) = 75582 k-subsets
    # C(8,5) = 56 t-subsets per block
    
    # Bitmask approach: represent each t-subset as a bit in a large integer
    # Number of t-subsets: C(v,t)
    # For v=19, t=5: C(19,5) = 11628 bits -> fits in a Python int
    
    n_t_subsets = _n_choose_k(v, t)
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # For each point x and (t-1)-subset s of the remaining points, 
    # compute which t-subsets are formed by s + {x}
    # Actually, for greedy: given a partial block B (size b < k), 
    # adding point x covers t-subsets of form (s ∪ {x}) where s ⊆ B, |s| = t-1
    
    # Precompute for each point x: for each (t-1)-subset of other points, the t-subset index
    # This is expensive for general v, but for v=19 it's manageable
    
    # Strategy: randomized greedy with multiple restarts
    # 1. Start with random ordering of t-subsets
    # 2. Greedily build blocks, each time picking the point that covers most uncovered t-subsets
    # 3. Use bitmasks for fast uncovered tracking
    
    best_blocks = None
    best_count = float('inf')
    
    # Precompute: for each point x, and for each combination of (t-1) points from a set,
    # we need to know which t-subsets are covered.
    # 
    # Key insight: when building a block with current set B, adding point x covers
    # t-subsets of the form {x} ∪ s where s is a (t-1)-subset of B.
    # We can precompute for each x and each (t-1)-tuple of points, the bitmask of t-subsets covered.
    
    # Actually, let's precompute: cover_mask[x][tuple_of_t_minus_1_points] = bitmask
    # But that's too much memory. Instead, compute on the fly but efficiently.
    
    # For v=19, t=5: (t-1)=4. Number of 4-subsets of 19 points: C(19,4)=3876
    # For each point x (19 points) and each 4-subset not containing x: 
    #   The 5-subset is {x} ∪ s
    # Precompute: point_to_tsubset[x][s_tuple] = bitmask (just one bit, actually)
    
    # Simpler: precompute for each point x, a dict mapping frozenset of (t-1) points to the t-subset index
    # Then when we have partial block B, for each x not in B:
    #   gain = number of uncovered t-subsets of form {x} ∪ s where s ⊆ B, |s|=t-1
    
    # Let's precompute: for each x, for each (t-1)-subset s not containing x, the index of {x}∪s
    x_to_s_to_idx = {}
    for x in range(v):
        x_to_s_to_idx[x] = {}
        for s in itertools.combinations([p for p in range(v) if p != x], t - 1):
            ts = tuple(sorted(s + (x,)))
            x_to_s_to_idx[x][s] = t_subset_index[ts]
    
    # Now for greedy: maintain uncovered as a set of t-subset indices (or a bitmask)
    # Using a set of indices is fine for C(19,5)=11628
    
    # Multiple randomized restarts
    n_restarts = 20
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the initial order of uncovered t-subsets
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered t-subset to start the block
            start_idx = rng.choice(list(uncovered))
            start_ts = t_subsets[start_idx]
            
            # Build block starting with the points in start_ts
            block = list(start_ts)
            rng.shuffle(block)
            
            while len(block) < k and uncovered:
                # Find the point that covers the most uncovered t-subsets
                best_x = -1
                best_gain = -1
                block_set = set(block)
                
                # Precompute (t-1)-subsets of current block
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for s in block_t_minus_1:
                        idx = x_to_s_to_idx[x].get(s)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x < 0 or best_gain == 0:
                    # No point adds coverage; pick a random one to complete the block
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = rng.choice(remaining)
                    else:
                        break
                
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) < k:
                continue
            
            block.sort()
            
            # Remove covered t-subsets
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                uncovered.discard(idx)
            
            blocks.append(block)
        
        if not uncovered and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            # If we've reached a good count, we can try to improve with more restarts
            # But if we're close to the known best (335), we might stop early
            if best_count <= 340:
                # Try a few more restarts to see if we can do better
                continue
        
        # Early termination if we've spent too much time
        if time.time() > deadline - 10:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(n_t_subsets))
        blocks = []
        while uncovered:
            # Pick the first uncovered t-subset
            start_idx = min(uncovered)
            start_ts = t_subsets[start_idx]
            block = list(start_ts)
            
            while len(block) < k:
                block_set = set(block)
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for s in block_t_minus_1:
                        idx = x_to_s_to_idx[x].get(s)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = remaining[0]
                    else:
                        break
                block.append(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                uncovered.discard(idx)
            blocks.append(block)
        
        best_blocks = blocks
    
    # Verify validity
    if _verify_cover(best_blocks, v, k, t):
        return best_blocks
    else:
        # Should not happen, but return anyway
        return best_blocks


def _n_choose_k(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _verify_cover(blocks, v, k, t):
    """Verify that every t-subset is covered by at least one block."""
    if not blocks:
        return False
    covered = set()
    for block in blocks:
        if len(block) != k:
            return False
        if len(set(block)) != k:
            return False
        if any(x < 0 or x >= v for x in block):
            return False
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    
    total_t_subsets = _n_choose_k(v, t)
    return len(covered) == total_t_subsets
# EVOLVE-BLOCK-END
