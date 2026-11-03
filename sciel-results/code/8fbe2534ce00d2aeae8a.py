"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 118.0
    
    # Precompute
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    tsub_to_idx = {fs: i for i, fs in enumerate(all_tsubsets)}
    num_tsubs = len(all_tsubsets)
    
    def block_to_tsub_indices(block):
        return [tsub_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]

    def greedy_cover():
        uncovered = set(range(num_tsubs))
        blocks = []
        while uncovered:
            seed_idx = rng.choice(list(uncovered))
            seed_set = all_tsubsets[seed_idx]
            block = list(seed_set)
            rng.shuffle(block)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        fs = frozenset(list(rest) + [x])
                        idx = tsub_to_idx.get(fs)
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
            if b_tuple not in [tuple(b) for b in blocks]:
                blocks.append(b_tuple)
                for idx in block_to_tsub_indices(b_tuple):
                    uncovered.discard(idx)
        return blocks

    blocks = greedy_cover()
    counts = [0] * num_tsubs
    for b in blocks:
        for idx in block_to_tsub_indices(b):
            counts[idx] += 1
            
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    
    # Precompute all possible blocks? No.
    # But we can precompute the list of t-subset indices for a block quickly.
    
    while time.time() - start_time < time_limit:
        current_len = len(blocks)
        
        # 1. Try Remove
        i = rng.randrange(current_len)
        b = blocks[i]
        redundant = True
        b_indices = block_to_tsub_indices(b)
        for idx in b_indices:
            if counts[idx] <= 1:
                redundant = False
                break
        
        if redundant:
            for idx in b_indices:
                counts[idx] -= 1
            blocks.pop(i)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
            T = max(T_min, T * cooling_rate)
            continue
        
        # 2. Try Swap
        # Pick a block
        i = rng.randrange(current_len)
        old_block = blocks[i]
        old_indices = block_to_tsub_indices(old_block)
        
        criticals = [idx for idx in old_indices if counts[idx] == 1]
        
        if criticals:
            points_needed = set()
            for idx in criticals:
                points_needed.update(all_tsubsets[idx])
            
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                # Greedy fill
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        idx = tsub_to_idx.get(fs)
                        if idx is not None:
                            if counts[idx] == 0:
                                gain += 10
                            elif counts[idx] == 1:
                                gain += 1
                    if gain > 0:
                        new_block_set.add(x)
                    else:
                        new_block_set.add(x)
                
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block:
                        new_indices = block_to_tsub_indices(new_block)
                        new_idx_set = set(new_indices)
                        
                        # Check validity
                        valid = True
                        for c in criticals:
                            if c not in new_idx_set:
                                valid = False
                                break
                        if valid:
                            for idx in old_indices:
                                if idx not in new_idx_set and counts[idx] == 1:
                                    valid = False
                                    break
                        
                        if valid:
                            # Apply swap
                            for idx in old_indices:
                                counts[idx] -= 1
                            for idx in new_indices:
                                counts[idx] += 1
                            blocks[i] = new_block
                            T = max(T_min, T * cooling_rate)
                            continue
        
        # 3. Try Add-Remove (2-opt)
        # Add a random block
        # Generate a random block
        r_block = tuple(sorted(rng.sample(range(v), k)))
        if r_block not in blocks:
            r_indices = block_to_tsub_indices(r_block)
            # Tentatively add
            for idx in r_indices:
                counts[idx] += 1
            
            # Try to remove 2 blocks
            removed_count = 0
            # Shuffle indices to pick random blocks to remove
            indices_to_check = list(range(len(blocks) + 1)) # +1 for the new block? 
            # Actually, we added to counts, but not to blocks list yet.
            # Let's add to a temp list
            temp_blocks = blocks + [r_block]
            
            # Check all blocks for redundancy
            # We want to remove 2 blocks if possible
            # Pick 2 random blocks and check if they are redundant
            # This is expensive. 
            # Alternative: Just try to remove 1 redundant block.
            # If we remove 1, net change 0.
            # If we can remove 2, net change -1.
            
            # Let's just try to find ANY redundant block in the new configuration
            redundant_idx = -1
            for j in range(len(temp_blocks)):
                b = temp_blocks[j]
                is_red = True
                for idx in block_to_tsub_indices(b):
                    if counts[idx] <= 1:
                        is_red = False
                        break
                if is_red:
                    redundant_idx = j
                    break
            
            if redundant_idx != -1:
                # Remove it
                b_rem = temp_blocks[redundant_idx]
                for idx in block_to_tsub_indices(b_rem):
                    counts[idx] -= 1
                # Remove from list
                # If we removed the new block, we revert the add
                # If we removed an old block, we keep the new block
                if redundant_idx == len(blocks):
                    # Removed the new block, so revert add
                    # Counts are already reverted for b_rem (which is r_block)
                    # So we are back to original state
                    pass
                else:
                    # Removed an old block
                    # Update blocks list
                    blocks[redundant_idx] = r_block # Replace old with new? 
                    # No, we added r_block to counts, and removed b_rem from counts.
                    # We want the list to reflect this.
                    # Original: blocks
                    # New: blocks with b_rem removed and r_block added
                    # So:
                    blocks.pop(redundant_idx)
                    blocks.append(r_block)
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = list(blocks)
                    T = max(T_min, T * cooling_rate)
                    continue
            else:
                # No redundant block found, revert add
                for idx in r_indices:
                    counts[idx] -= 1
        
        T = max(T_min, T * cooling_rate)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
