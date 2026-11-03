"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    # Precompute all t-subsets and map to indices for fast lookup
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # For each point, precompute which t-subsets contain it
    # point_subs[p] = list of indices of t-subsets containing p
    point_subs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_subs[p].append(i)
    
    # Coverage count: how many blocks cover each t-subset
    cover_count = np.zeros(n_t, dtype=np.int32)
    
    # Greedy phase: repeatedly pick the block that covers the most uncovered t-subsets
    # We'll use a heuristic to speed this up: for each candidate block, compute coverage
    # of uncovered subsets. But enumerating all C(v,k) blocks is too many.
    # Instead, use a smarter greedy: pick points one at a time to maximize uncovered coverage.
    
    blocks = []
    used_blocks = set()
    
    def compute_block_coverage(block_tuple):
        """Return number of uncovered t-subsets covered by this block."""
        count = 0
        for sub in itertools.combinations(block_tuple, t):
            idx = sub_to_idx[sub]
            if cover_count[idx] == 0:
                count += 1
        return count
    
    # Greedy construction with time check
    while True:
        if time.time() > deadline - 20:  # reserve 20s for local search
            break
        if cover_count.min() > 0:
            break
        
        # Find the best block to add: maximize uncovered t-subsets covered
        # Use a heuristic: start with the uncovered t-subset that has the fewest
        # candidate blocks, then build around it.
        
        # Find uncovered t-subset with minimum "degree" (number of possible blocks containing it)
        # For a t-subset, the number of k-blocks containing it is C(v-t, k-t)
        # So all uncovered subsets have the same degree. Instead, pick one randomly or
        # the first uncovered one.
        
        # Better: pick an uncovered t-subset, then greedily add points to maximize coverage
        uncovered_indices = np.where(cover_count == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered t-subset to anchor
        anchor_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        anchor_sub = all_t_subs[anchor_idx]
        
        # Build block starting with anchor points
        block_set = set(anchor_sub)
        
        # Greedily add points that maximize uncovered coverage
        while len(block_set) < k:
            best_point = None
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                # Compute gain: how many new uncovered t-subsets would be covered
                # A t-subset is newly covered if it contains p and (t-1) points from block_set
                # and is currently uncovered
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    sub = tuple(sorted(rest + (p,)))
                    idx = sub_to_idx[sub]
                    if cover_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point is None:
                break
            block_set.add(best_point)
        
        block = tuple(sorted(block_set))
        if block in used_blocks:
            continue
        used_blocks.add(block)
        blocks.append(list(block))
        
        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            cover_count[idx] += 1
    
    # Local search phase: try to improve by swapping points in blocks
    # Use time remaining
    if time.time() < deadline - 5:
        # Try to reduce block count by finding redundant blocks
        # A block is redundant if removing it still leaves all t-subsets covered
        # This is expensive, so we do a limited version
        
        # First, try simple local search: for each block, try replacing one point
        # with another to see if it covers more uncovered subsets (if any remain)
        
        # Actually, let's try a different approach: local search on the block set
        # Try to remove a block if it's redundant
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Check if block i is redundant
            # Temporarily remove it and check if all t-subsets are still covered
            # This is expensive, so we do a quick check:
            # Only check t-subsets that are covered exactly once by block i
            removable = True
            for sub in itertools.combinations(blocks[i], t):
                idx = sub_to_idx[sub]
                if cover_count[idx] == 1:
                    # This t-subset is only covered by block i
                    # Check if it's covered by another block
                    found = False
                    for j, blk in enumerate(blocks):
                        if j == i:
                            continue
                        if all(p in blk for p in sub):
                            found = True
                            break
                    if not found:
                        removable = False
                        break
            if removable:
                # Remove block i
                for sub in itertools.combinations(blocks[i], t):
                    idx = sub_to_idx[sub]
                    cover_count[idx] -= 1
                blocks.pop(i)
    
    # Final check: ensure all t-subsets are covered
    # If not, add blocks greedily
    while True:
        if time.time() > deadline - 2:
            break
        uncovered_indices = np.where(cover_count == 0)[0]
        if len(uncovered_indices) == 0:
            break
        anchor_idx = uncovered_indices[0]
        anchor_sub = all_t_subs[anchor_idx]
        block_set = set(anchor_sub)
        while len(block_set) < k:
            best_point = None
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    sub = tuple(sorted(rest + (p,)))
                    idx = sub_to_idx[sub]
                    if cover_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point is None:
                break
            block_set.add(best_point)
        block = tuple(sorted(block_set))
        if block not in used_blocks:
            used_blocks.add(block)
            blocks.append(list(block))
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                cover_count[idx] += 1
    
    return blocks
# EVOLVE-BLOCK-END
