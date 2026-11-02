import itertools
import time
import random
from functools import lru_cache

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    all_points = list(range(v))
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = [0] * 1
    # Actually, let's just enumerate
    t_subsets_list = list(itertools.combinations(all_points, t))
    
    # Convert each t-subset to a bitmask
    t_subset_bitmasks = []
    for sub in t_subsets_list:
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_subset_bitmasks.append(mask)
    
    # Set of uncovered bitmasks
    # For v=19, k=8, t=5: C(19,5)=11628 subsets
    
    @lru_cache(maxsize=None)
    def block_five_submasks(block_tuple):
        """Return tuple of bitmasks for all 5-subsets of the 8-element block."""
        result = []
        for sub in itertools.combinations(block_tuple, t):
            mask = 0
            for p in sub:
                mask |= (1 << p)
            result.append(mask)
        return tuple(result)
    
    def greedy_cover():
        uncovered = set(t_subset_bitmasks)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to break symmetry
            target = random.choice(list(uncovered))
            
            # Get the points in target
            target_points = []
            tmp = target
            for i in range(v):
                if tmp & (1 << i):
                    target_points.append(i)
            
            remaining = [x for x in all_points if x not in target_points]
            need = k - t
            
            best_gain = -1
            best_block = None
            
            for combo in itertools.combinations(remaining, need):
                block = tuple(sorted(target_points + list(combo)))
                # Get five-subset bitmasks for this block
                submasks = block_five_submasks(block)
                # Count how many are uncovered
                gain = 0
                for sm in submasks:
                    if sm in uncovered:
                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = block
                            if best_gain == 56:
                                break
                if best_gain == 56:
                    break
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered
            submasks = block_five_submasks(best_block)
            for sm in submasks:
                uncovered.discard(sm)
                    
            if time.time() > deadline - 15:
                break
                
        return blocks
    
    def delete_blocks(blocks):
        if not blocks:
            return blocks
        
        # Count coverage using bitmask keys
        counts = {}
        for b in blocks:
            for sm in block_five_submasks(b):
                counts[sm] = counts.get(sm, 0) + 1
        
        removed_any = True
        while removed_any:
            removed_any = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                b = blocks[i]
                can_remove = True
                for sm in block_five_submasks(b):
                    if counts[sm] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for sm in block_five_submasks(b):
                        counts[sm] -= 1
                    blocks[i] = None
                    removed_any = True
                    if time.time() > deadline - 5:
                        return [b for b in blocks if b is not None]
            
            if removed_any:
                blocks = [b for b in blocks if b is not None]
        
        # Swap-based local search: try replacing one block with another to enable more deletions
        # This is expensive, so only do it if we have time
        if time.time() < deadline - 20:
            blocks = swap_local_search(blocks, counts)
        
        return blocks
    
    def swap_local_search(blocks, counts):
        """Try to improve by replacing blocks. Simple version: try removing+adding."""
        # For now, just do a few rounds of trying to add a block that enables more removals
        # This is a heuristic; keep it simple
        improved = True
        max_rounds = 3
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 10:
            improved = False
            rounds += 1
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                # Try removing block i temporarily
                b = blocks[i]
                submasks = block_five_submasks(b)
                for sm in submasks:
                    counts[sm] -= 1
                    if counts[sm] < 0:
                        counts[sm] = 0  # shouldn't happen
                # Now try to see if we can add a different block that covers the newly uncovered
                # and allows removing another block
                # This is complex; skip for now and just re-add
                for sm in submasks:
                    counts[sm] += 1
            # Just do a deletion pass again
            removed_any = True
            while removed_any:
                removed_any = False
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                for i in indices:
                    b = blocks[i]
                    can_remove = True
                    for sm in block_five_submasks(b):
                        if counts[sm] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for sm in block_five_submasks(b):
                            counts[sm] -= 1
                        blocks[i] = None
                        removed_any = True
                        improved = True
                        if time.time() > deadline - 10:
                            return blocks
                if removed_any:
                    blocks = [b for b in blocks if b is not None]
        
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy instances
    while time.time() < deadline - 25:
        blocks = greedy_cover()
        if not blocks:
            continue
            
        blocks = delete_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we have time, try to improve the best found so far
        if best_blocks is not None and time.time() < deadline - 30:
            # Occasionally re-run deletion on the best with different random order
            pass
    
    if best_blocks is None:
        random.seed(42)
        blocks = greedy_cover()
        blocks = delete_blocks(blocks)
        return blocks
        
    return best_blocks
