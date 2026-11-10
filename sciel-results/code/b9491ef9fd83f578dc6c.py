import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0
    margin = 10.0
    
    def remaining_time():
        return deadline - margin - (time.time() - start_time)
    
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    
    if not uncovered:
        return []
    
    blocks = []
    
    # Precompute t-subsets for each possible k-block for faster gain calculation
    # For v=19, k=7, t=4: C(19,7) = 50388 blocks, each has C(7,4)=35 t-subsets
    # That's ~1.76M t-subsets total. Might be too much memory/time.
    # Instead, compute gain on the fly but cache per-block.
    
    block_tsubs = {}
    
    def get_block_tsubs(block):
        """Get or compute the t-subsets contained in a block."""
        if block not in block_tsubs:
            block_tsubs[block] = set(itertools.combinations(block, t))
        return block_tsubs[block]
    
    # Greedy phase: pick block with maximum gain at each step
    while uncovered and remaining_time() > 0:
        best_gain = -1
        best_block = None
        
        # Pick a random uncovered subset as seed to diversify
        seed = random.choice(list(uncovered)) if uncovered else None
        if seed is None:
            break
        
        seed_set = set(seed)
        remaining_points = [x for x in range(v) if x not in seed_set]
        num_to_add = k - t
        
        for add_combo in itertools.combinations(remaining_points, num_to_add):
            candidate_block = tuple(sorted(seed + add_combo))
            
            gain = 0
            for sub in itertools.combinations(candidate_block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
            elif gain == best_gain and best_block is not None:
                # Tie-breaking: prefer lexicographically smaller
                if candidate_block < best_block:
                    best_block = candidate_block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        # Update uncovered set
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    
    # Local improvement: remove redundant blocks
    if remaining_time() > 5.0:
        changed = True
        while changed and remaining_time() > 5.0:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                block_i = tuple(blocks[i])
                tsubs_i = get_block_tsubs(block_i)
                
                # Check if all t-subsets of block_i are covered by other blocks
                all_covered = True
                for sub in tsubs_i:
                    covered = False
                    for j, block_j in enumerate(blocks):
                        if j == i:
                            continue
                        if sub in get_block_tsubs(tuple(block_j)):
                            covered = True
                            break
                    if not covered:
                        all_covered = False
                        break
                
                if all_covered:
                    blocks.pop(i)
                    changed = True
                    break  # Restart from beginning since indices changed
    
    # Local improvement: try replacing individual blocks to cover their unique t-subsets better
    if remaining_time() > 5.0:
        for iteration in range(3):  # A few passes
            if remaining_time() < 5.0:
                break
            for i in range(len(blocks)):
                if remaining_time() < 5.0:
                    break
                block_i = tuple(blocks[i])
                tsubs_i = get_block_tsubs(block_i)
                
                # Find t-subsets only covered by this block
                unique_tsubs = set()
                for sub in tsubs_i:
                    only_in_i = True
                    for j, block_j in enumerate(blocks):
                        if j == i:
                            continue
                        if sub in get_block_tsubs(tuple(block_j)):
                            only_in_i = False
                            break
                    if only_in_i:
                        unique_tsubs.add(sub)
                
                if not unique_tsubs:
                    continue
                
                # Try to find a better block that covers all unique_tsubs
                # and potentially more uncovered subsets
                best_gain = -1
                best_new_block = None
                
                # The new block must contain all points in unique_tsubs
                # Actually, it must contain each t-subset in unique_tsubs
                # This is complex; instead, try replacing with a block that
                # covers the same or more of the currently covered t-subsets
                
                # Simpler: try all k-blocks that contain the intersection of unique_tsubs
                # For now, skip this complex step and just do redundancy removal
                pass
    
    return blocks
