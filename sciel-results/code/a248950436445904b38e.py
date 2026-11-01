import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    total_5subsets = list(itertools.combinations(all_points, t))
    n_5subsets = len(total_5subsets)
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(total_5subsets)}
    
    # Precompute for each possible k-subset? No, too many.
    # We will compute on the fly for generated blocks.
    
    def get_block_subsets(block):
        """Return list of indices of 5-subsets contained in block"""
        subs = []
        for combo in itertools.combinations(block, t):
            idx = subset_to_idx.get(tuple(combo))
            if idx is not None:
                subs.append(idx)
        return subs

    def greedy_init():
        """Quick greedy to get a valid initial solution"""
        # This is a fallback to ensure we start valid, even though the main loop is LS.
        # But the directive says don't reuse designs. 
        # However, starting from invalid is hard.
        # Let's start with random blocks and repair.
        pass

    blocks = []
    cover_count = [0] * n_5subsets
    block_subs = [] # list of lists of subset indices for each block
    
    def add_block(block):
        block = tuple(sorted(block))
        if block in [tuple(b) for b in blocks]:
            return False
        subs = get_block_subsets(block)
        blocks.append(block)
        block_subs.append(subs)
        for idx in subs:
            cover_count[idx] += 1
        return True

    def remove_block(i):
        subs = block_subs[i]
        for idx in subs:
            cover_count[idx] -= 1
        blocks.pop(i)
        block_subs.pop(i)

    def is_redundant(i):
        for idx in block_subs[i]:
            if cover_count[idx] == 1:
                return False
        return True

    def prune():
        # Remove redundant blocks
        # We must iterate carefully as indices change
        i = 0
        while i < len(blocks):
            if is_redundant(i):
                remove_block(i)
            else:
                i += 1

    start_time = time.time()
    time_limit = 110
    
    # 1. Initialization: Add random blocks until valid
    rng = random.Random(42)
    while time.time() - start_time < time_limit * 0.2: # Use first 20% to get valid
        if all(c > 0 for c in cover_count):
            break
        # Pick a random uncovered subset to seed
        uncovered_indices = [i for i in range(n_5subsets) if cover_count[i] == 0]
        if not uncovered_indices:
            break
        seed_idx = rng.choice(uncovered_indices)
        seed_subset = total_5subsets[seed_idx]
        
        block = list(seed_subset)
        remaining = [p for p in all_points if p not in set(block)]
        rng.shuffle(remaining)
        block.extend(remaining[:k-t])
        rng.shuffle(block)
        block.sort()
        
        add_block(block)
        
    # If not valid, force fill
    while not all(c > 0 for c in cover_count):
        uncovered_indices = [i for i in range(n_5subsets) if cover_count[i] == 0]
        if not uncovered_indices:
            break
        seed_idx = rng.choice(uncovered_indices)
        seed_subset = total_5subsets[seed_idx]
        block = list(seed_subset)
        remaining = [p for p in all_points if p not in set(block)]
        rng.shuffle(remaining)
        block.extend(remaining[:k-t])
        rng.shuffle(block)
        block.sort()
        add_block(block)
        
    # Initial prune
    prune()
    
    best_len = len(blocks)
    best_blocks = [list(b) for b in blocks]
    
    # 2. Local Search Loop
    while time.time() - start_time < time_limit:
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Pick a random block to try to replace
        if not blocks:
            break
            
        i = rng.randint(0, len(blocks) - 1)
        block_i = blocks[i]
        
        # Find unique subsets of block_i
        unique_subs = []
        for idx in block_subs[i]:
            if cover_count[idx] == 1:
                unique_subs.append(idx)
                
        if not unique_subs:
            # Should be pruned, but just in case
            remove_block(i)
            continue
            
        # Calculate union of points in unique subsets
        union_points = set()
        for idx in unique_subs:
            union_points.update(total_5subsets[idx])
            
        if len(union_points) > k:
            # Cannot replace with single block
            continue
            
        # Construct new block
        new_block_set = set(union_points)
        remaining_points = [p for p in all_points if p not in new_block_set]
        rng.shuffle(remaining_points)
        new_block_set.update(remaining_points[:k - len(union_points)])
        new_block = tuple(sorted(new_block_set))
        
        if new_block == block_i:
            continue
            
        # Check if new block is already in set? 
        # If it is, replacing B_i with existing B_j means we can just remove B_i?
        # No, if B_j exists, it covers the same things. 
        # If we replace B_i with B_j, we effectively remove B_i and keep B_j.
        # But we are just swapping.
        
        # Tentative swap
        old_subs = block_subs[i]
        remove_block(i)
        
        added = add_block(new_block)
        
        if not added:
            # Block already existed? 
            # If add_block returns False, it means it was a duplicate.
            # But we just removed i. So it shouldn't be a duplicate unless 
            # the new block is identical to some other existing block.
            # In that case, we effectively removed B_i and the other block remains.
            # This is a good move (size decreased by 1).
            # But add_block returned False, so it didn't add.
            # So size decreased by 1.
            pass
        else:
            # Size stayed same
            pass
            
        # Prune
        prune()
        
        # Decide whether to keep
        # If size decreased, keep.
        # If size same, keep with prob 0.5?
        # If size increased (shouldn't happen with swap+prune? 
        # Swap keeps size same. Prune reduces. So size <= old_size.
        
        # We don't track old_size explicitly in this loop iteration easily 
        # because prune might remove multiple.
        # Let's just accept the move. It's a random walk with bias towards smaller (pruning).
        
    return best_blocks
