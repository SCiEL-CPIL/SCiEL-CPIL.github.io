"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(tsubset_indices)))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = tsubset_indices[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_block_idx.get(fs)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        blocks.append(b_tuple)
        for c in itertools.combinations(block, t):
            uncovered.discard(subset_to_block_idx[frozenset(c)])
                
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute all t-subsets and map frozenset -> index
    tsubset_indices = []
    subset_to_block_idx = {}
    for c in itertools.combinations(range(v), t):
        fs = frozenset(c)
        subset_to_block_idx[fs] = len(tsubset_indices)
        tsubset_indices.append(c)
    
    total_t = len(tsubset_indices)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx)
    
    # Precompute t-subset indices for each block
    def block_tsubsets_indices(b):
        return [subset_to_block_idx[frozenset(c)] for c in itertools.combinations(b, t)]
    
    # Initialize counts array
    counts = [0] * total_t
    for b in blocks:
        for idx in block_tsubsets_indices(b):
            counts[idx] += 1
    
    # Track uncovered count
    uncovered_count = sum(1 for c in counts if c == 0)
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    def try_remove_block(blk_idx):
        """
        Try to remove block at blk_idx.
        Returns True if successful, False otherwise.
        """
        b = blocks[blk_idx]
        # Check if removal is valid: all t-subsets in b must have count > 1
        for idx in block_tsubsets_indices(b):
            if counts[idx] <= 1:
                return False
        
        # It is valid. Perform removal.
        for idx in block_tsubsets_indices(b):
            counts[idx] -= 1
        
        blocks.pop(blk_idx)
        return True
    
    # Simulated Annealing / Local Search Loop
    iteration = 0
    no_improve = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block
        removed = False
        # Randomly check blocks for redundancy
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            if try_remove_block(i):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    no_improve = 0
                else:
                    no_improve += 1
                removed = True
                break
        
        if removed:
            continue
            
        # 2. Try to swap a block to enable removal or just find a better configuration
        # Strategy: Pick a block that has "low" utility (covers many subsets that are already well-covered)
        # Or pick a block that is "critical" (covers unique subsets) and try to replace it with a block that covers those unique subsets + others.
        
        if len(blocks) > 0:
            # Pick a random block to potentially replace
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Identify subsets in old_block that are critical (count 1)
            critical_indices = []
            for idx in block_tsubsets_indices(old_block):
                if counts[idx] == 1:
                    critical_indices.append(idx)
            
            # If no critical subsets, we can just try to remove it? 
            # We already checked removal in step 1. If it failed, it means it has critical subsets.
            # So critical_indices is not empty.
            
            # We need to construct a new block that covers all critical_indices.
            # If len(critical_indices) > 1, the new block must contain the union of these subsets?
            # No, it must contain each subset.
            # Since k=7, t=3.
            # If we have two critical subsets {a,b,c} and {a,b,d}, the new block must contain a,b,c,d.
            # Let's collect the points involved in critical subsets.
            points_needed = set()
            for idx in critical_indices:
                points_needed.update(tsubset_indices[idx])
                
            # If points_needed > k, we cannot cover all critical subsets in one block.
            # In this case, we cannot simply replace this block with one block.
            # We might need to do a 2-block swap or revert.
            # For simplicity, if we can't form a valid replacement, we skip.
            if len(points_needed) <= k:
                # Try to form a new block containing points_needed and filling the rest greedily
                new_block_set = set(points_needed)
                # Fill remaining slots to maximize coverage of other subsets
                # Candidates: all points not in new_block_set
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                # Greedy fill
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    # Calculate gain: how many *uncovered* or *single-covered* subsets does adding x cover?
                    # We want to cover subsets that are currently 0 or 1.
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        idx = subset_to_block_idx.get(fs)
                        if idx is not None:
                            c = counts[idx]
                            if c == 0:
                                gain += 10 # High priority
                            elif c == 1:
                                gain += 1
                    if gain > 0:
                        new_block_set.add(x)
                    elif len(new_block_set) < k:
                        # If no gain, just add random to fill
                        new_block_set.add(x)
                        
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    
                    # Check if new_block is same as old_block
                    if new_block == old_block:
                        continue
                        
                    # Validate the swap:
                    # 1. Remove old_block from counts
                    # 2. Add new_block to counts
                    # 3. Check if all counts >= 1
                    # 4. Revert if invalid
                    
                    # Simulate removal
                    old_block_indices = block_tsubsets_indices(old_block)
                    new_block_indices = block_tsubsets_indices(new_block)
                    
                    # Check validity: Are there any 0s?
                    # We need to check if any subset became 0.
                    # Subsets that were 1 and are in old_block but not new_block become 0.
                    new_block_idx_set = set(new_block_indices)
                    valid = True
                    for idx in old_block_indices:
                        if counts[idx] == 1 and idx not in new_block_idx_set:
                            valid = False
                            break
                    
                    if valid:
                        # Accept swap
                        for idx in old_block_indices:
                            counts[idx] -= 1
                        for idx in new_block_indices:
                            counts[idx] += 1
                        blocks[i] = new_block
                        no_improve = 0
                    # else: skip (don't apply)
        
        # 3. Periodic greedy restart if stuck
        if no_improve > 500 or iteration % 200 == 0:
            new_cover = _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                # Rebuild counts
                counts = [0] * total_t
                for b in blocks:
                    for idx in block_tsubsets_indices(b):
                        counts[idx] += 1
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    no_improve = 0
                else:
                    no_improve = 0
            else:
                no_improve += 100
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
            no_improve = 0
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
