import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    # Precompute all 6-subsets and their 5-subset membership
    all_6sets = list(combinations(range(12), 6))
    n_blocks = len(all_6sets)  # 924
    
    # Map each 5-subset to a unique index
    all_5subsets = list(combinations(range(12), 5))
    n_5subsets = len(all_5subsets)  # 792
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Precompute for each 6-set, which 5-subsets it contains (as bitset)
    block_5mask = [0] * n_blocks
    for i, b in enumerate(all_6sets):
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_5mask[i] = mask
    
    # Also precompute: for each 5-subset, which 6-sets contain it
    # This helps with greedy selection
    # But for now, let's use a smarter approach
    
    # Strategy: Use a greedy algorithm with a priority based on how "constrained" a block is
    # A block is more constrained if its 5-subsets are shared with more other blocks
    
    # Compute conflict count for each block
    # For each 5-subset in block i, count how many other blocks also contain it
    conflict_score = [0] * n_blocks
    for i in range(n_blocks):
        mask = block_5mask[i]
        # Count bits - but we need to know how many other blocks share these 5-subsets
        # Alternative: for each 5-subset, precompute how many blocks contain it
        pass
    
    # Simpler approach: precompute for each 5-subset, the number of blocks containing it
    # Then a block's "score" = sum over its 5-subsets of (1/count) - higher is better (less constrained)
    # Actually, we want to pick blocks that are "rare" first - blocks whose 5-subsets are in fewer other blocks
    
    # For each 5-subset, count how many 6-sets contain it
    # Each 5-subset is in exactly 7 6-sets (since we add one of the 7 remaining points)
    # So all 5-subsets have the same count = 7
    # This means all blocks are equally constrained in this sense
    
    # Let's try a different approach: use a greedy algorithm that picks blocks
    # in an order that maximizes coverage, using a randomization strategy
    
    import random
    
    best_blocks = None
    best_count = 0
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_blocks))
        else:
            random.seed(trial)
            order = list(range(n_blocks))
            random.shuffle(order)
        
        covered = 0  # bitset of covered 5-subsets
        blocks = []
        
        for idx in order:
            if time.time() > deadline:
                break
            mask = block_5mask[idx]
            if (covered & mask) == 0:
                blocks.append(list(all_6sets[idx]))
                covered |= mask
                if len(blocks) == 132:
                    return blocks
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
            if best_count == 132:
                return blocks
    
    # If we haven't found 132, try a backtracking approach
    # Start from the best partial solution and try to complete it
    if best_count < 132 and best_blocks is not None:
        # Try to extend the best solution
        random.seed(42)
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Take a random subset of the best solution and try to complete
            k = min(len(best_blocks), random.randint(10, 40))
            start_set = random.sample(best_blocks, k)
            
            covered = 0
            for b in start_set:
                # Find index of b in all_6sets
                b_tuple = tuple(b)
                if b_tuple in [all_6sets[i] for i in range(n_blocks)]:
                    idx = all_6sets.index(b_tuple)
                    covered |= block_5mask[idx]
            
            blocks = list(start_set)
            
            # Complete greedily
            order = list(range(n_blocks))
            random.shuffle(order)
            
            for idx in order:
                if time.time() > deadline:
                    break
                mask = block_5mask[idx]
                if (covered & mask) == 0:
                    blocks.append(list(all_6sets[idx]))
                    covered |= mask
                    if len(blocks) == 132:
                        return blocks
            
            if len(blocks) > best_count:
                best_count = len(blocks)
                best_blocks = blocks
                if best_count == 132:
                    return blocks
    
    return best_blocks if best_blocks is not None else []
