import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    N_t = len(all_t_subsets)
    
    # Helper to get indices of t-subsets in a block
    def get_block_indices(block):
        return [subset_to_idx[s] for s in itertools.combinations(block, t)]
    
    # State
    blocks = []
    cover_count = [0] * N_t
    
    def add_block(block):
        idxs = get_block_indices(block)
        for i in idxs:
            cover_count[i] += 1
        blocks.append(block)
        
    def remove_block(block):
        idxs = get_block_indices(block)
        for i in idxs:
            cover_count[i] -= 1
        blocks.remove(block)
        
    def is_redundant(block):
        idxs = get_block_indices(block)
        for i in idxs:
            if cover_count[i] <= 1:
                return False
        return True
        
    def try_remove_redundants():
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle removals safely
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(blocks[i]):
                    remove_block(blocks[i])
                    changed = True
                    if time.time() > deadline:
                        return
            
    # Initialization: Random Greedy
    uncovered = set(range(N_t))
    while uncovered:
        if time.time() > deadline:
            break
        # Pick a random uncovered subset
        s_idx = random.choice(list(uncovered))
        s = all_t_subsets[s_idx]
        
        # Generate a random block containing s
        # s is a tuple of t points
        remaining_points = [p for p in points if p not in s]
        extra_points = random.sample(remaining_points, k - t)
        block = tuple(sorted(s + tuple(extra_points)))
        
        add_block(block)
        
        # Update uncovered
        idxs = get_block_indices(block)
        for i in idxs:
            if i in uncovered:
                uncovered.remove(i)
                
    # Initial cleanup
    try_remove_redundants()
    
    best_blocks = [b for b in blocks]
    best_size = len(blocks)
    
    # Local Search
    # We perform 1-for-1 swaps
    # To generate a new block, we modify an existing one
    # Strategy: Pick a random block, change 1 or 2 points randomly.
    
    while time.time() < deadline:
        if not blocks:
            break
            
        # Try to remove redundants first
        try_remove_redundants()
        
        current_size = len(blocks)
        if current_size < best_size:
            best_size = current_size
            best_blocks = [b for b in blocks]
            
        # Perform a swap
        old_block = random.choice(blocks)
        
        # Generate candidate
        # Modify old_block by swapping 1 point
        # Pick a point in old_block to replace
        p_in = random.choice(old_block)
        # Pick a point not in old_block to replace with
        available = [p for p in points if p not in old_block]
        if not available:
            continue
        p_out = random.choice(available)
        
        new_block_set = set(old_block)
        new_block_set.remove(p_in)
        new_block_set.add(p_out)
        new_block = tuple(sorted(new_block_set))
        
        # Check if new_block is already in blocks
        if new_block in blocks:
            continue
            
        # Check validity of swap
        old_idxs = get_block_indices(old_block)
        new_idxs = get_block_indices(new_block)
        
        # Tentatively remove old
        valid = True
        for i in old_idxs:
            cover_count[i] -= 1
            if cover_count[i] < 0:
                # Should not happen
                pass
            elif cover_count[i] == 0:
                valid = False
                break
        
        if valid:
            # Tentatively add new
            for i in new_idxs:
                cover_count[i] += 1
            
            # Commit
            blocks.remove(old_block)
            blocks.append(new_block)
        else:
            # Rollback
            for i in old_idxs:
                cover_count[i] += 1

    # Final cleanup
    try_remove_redundants()
    if len(blocks) < best_size:
        best_blocks = [b for b in blocks]
        
    return [list(b) for b in best_blocks]
