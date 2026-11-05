import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_k_subs = list(itertools.combinations(points, k))
    k_sub_map = {sub: i for i, sub in enumerate(all_k_subs)}
    
    # For each k-subset, store the indices of the t-subsets it covers
    block_indices = []
    for block in all_k_subs:
        inds = [t_sub_map[sub] for sub in itertools.combinations(block, t)]
        block_indices.append(inds)
        
    # 2. Greedy Construction
    def greedy_construct():
        counts = np.zeros(num_t_subs, dtype=np.int16)
        uncovered = set(range(num_t_subs))
        blocks = []
        
        # Precompute which blocks contain which t-subset for faster candidate selection?
        # Or just use the random t-subset strategy.
        
        while uncovered:
            # Pick a random uncovered t-subset
            u_idx = random.choice(list(uncovered))
            
            # Find candidates: k-subsets containing u_idx
            # We can precompute a mapping from t-subset index to list of k-subset indices
            # But that's memory heavy. 
            # Alternative: Iterate all k-subsets? 50k * 150 steps = 7.5M checks.
            # Each check: is u_idx in block_indices[b]?
            # This is slow in Python.
            
            # Better: Precompute `blocks_containing_t`
            # blocks_containing_t[t_idx] = list of k_indices
            # Memory: 3876 * avg(153) * 4 bytes ~ 2.4MB. Fine.
            
            pass # Will implement in main scope to avoid re-computation if multiple greedy runs?
            # Actually, we only run greedy once or a few times.
            
            # Let's optimize the greedy loop by precomputing `blocks_containing_t` outside.
            
        return blocks, counts

    # Precompute blocks_containing_t
    blocks_containing_t = [[] for _ in range(num_t_subs)]
    for k_idx, inds in enumerate(block_indices):
        for t_idx in inds:
            blocks_containing_t[t_idx].append(k_idx)
            
    def run_greedy():
        counts = np.zeros(num_t_subs, dtype=np.int16)
        uncovered = set(range(num_t_subs))
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            u_idx = random.choice(list(uncovered))
            candidates = blocks_containing_t[u_idx]
            
            best_block = -1
            max_gain = -1
            # Sample a few candidates if list is large? List is ~153. Small.
            for b_idx in candidates:
                inds = block_indices[b_idx]
                gain = 0
                # Count how many of these are currently uncovered
                # Using numpy:
                # gain = np.sum(counts[inds] == 0)
                gain = np.sum(counts[inds] == 0)
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            blocks.append(best_block)
            inds = block_indices[best_block]
            counts[inds] += 1
            
            # Update uncovered set
            # This is slow in Python set.
            # We can just check counts directly in the main loop?
            # For greedy, we need to know if it's uncovered.
            # We can maintain a list of uncovered indices and filter it.
            new_uncovered = []
            for idx in uncovered:
                if counts[idx] == 0:
                    new_uncovered.append(idx)
            uncovered = set(new_uncovered)
            
        return blocks, counts

    # Run greedy a few times to find a good start
    best_blocks = None
    best_counts = None
    best_len = float('inf')
    
    for _ in range(5):
        b, c = run_greedy()
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b
            best_counts = c
            
    if best_blocks is None:
        # Fallback
        return [list(b) for b in all_k_subs[:best_len]]
        
    # 3. Local Search
    current_blocks = best_blocks
    current_counts = best_counts.copy()
    best_count = len(current_blocks)
    
    # Function to remove redundant blocks
    def remove_redundant(blocks, counts):
        new_blocks = []
        for b_idx in blocks:
            inds = block_indices[b_idx]
            counts[inds] -= 1
            # Check if any count is 0
            if np.any(counts[inds] == 0):
                counts[inds] += 1
                new_blocks.append(b_idx)
            # else: block is redundant, counts stay decremented
        return new_blocks

    # Main local search loop
    while time.time() < deadline - 5:
        # Try to remove redundant blocks
        new_blocks = remove_redundant(current_blocks, current_counts)
        
        if len(new_blocks) < best_count:
            best_count = len(new_blocks)
            current_blocks = new_blocks
            # current_counts is already updated by remove_redundant
        else:
            current_blocks = new_blocks # Keep the cleaned version
            
        # Perturbation
        if not current_blocks:
            break
            
        # Pick a random block to replace
        i = random.randint(0, len(current_blocks) - 1)
        old_b_idx = current_blocks[i]
        
        # Generate new block
        old_points = all_k_subs[old_b_idx]
        # Pick a random t-subset from the old block to anchor the new one
        rand_t_sub = random.choice(list(itertools.combinations(old_points, t)))
        # Pick random other points
        remaining_points = [p for p in points if p not in rand_t_sub]
        # Need k - t more points
        new_other_points = random.sample(remaining_points, k - t)
        new_points = tuple(sorted(rand_t_sub + new_other_points))
        new_b_idx = k_sub_map[new_points]
        
        # Apply replacement
        old_inds = block_indices[old_b_idx]
        new_inds = block_indices[new_b_idx]
        
        current_counts[old_inds] -= 1
        current_counts[new_inds] += 1
        
        # Check validity: no count should be < 0? 
        # Since we removed old and added new, counts can be 0 (uncovered) but not negative.
        # However, if the old block was the ONLY cover for some subset, and the new block doesn't cover it,
        # that subset becomes uncovered (count 0). This is allowed in the intermediate state,
        # BUT the final result must be a valid cover.
        # The local search strategy usually requires maintaining validity.
        # If we allow invalid states, we need a repair step or penalty.
        # Simpler: Only accept replacement if the cover remains valid.
        # i.e., all counts >= 1.
        
        if np.any(current_counts < 1):
            # Revert
            current_counts[old_inds] += 1
            current_counts[new_inds] -= 1
            continue
            
        current_blocks[i] = new_b_idx
        
        # Immediately try to remove redundant blocks again?
        # The loop will do it at the start of the next iteration.
        
    # Return the best blocks found
    # Convert indices back to lists of points
    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]

# Note: The code above has a bug in `remove_redundant` logic regarding `current_counts` update.
# `remove_redundant` modifies `counts` in place.
# If a block is kept, counts are restored.
# If a block is removed, counts are NOT restored.
# So `counts` reflects the new set of blocks.
# This is correct.

# One issue: `best_blocks` is updated only when a strictly smaller set is found.
# But `current_blocks` is updated to the redundant-removed set even if size is same.
# This is good for exploring the landscape.

# Final check on return type: list of lists of ints.
# `all_k_subs` are tuples. `list(...)` converts to list.

# Let's refine the code structure to be cleaner and ensure no infinite loops or errors.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_k_subs = list(itertools.combinations(points, k))
    k_sub_map = {sub: i for i, sub in enumerate(all_k_subs)}
    
    block_indices = []
    for block in all_k_subs:
        inds = [t_sub_map[sub] for sub in itertools.combinations(block, t)]
        block_indices.append(inds)
        
    blocks_containing_t = [[] for _ in range(num_t_subs)]
    for k_idx, inds in enumerate(block_indices):
        for t_idx in inds:
            blocks_containing_t[t_idx].append(k_idx)
            
    def run_greedy():
        counts = np.zeros(num_t_subs, dtype=np.int16)
        uncovered = set(range(num_t_subs))
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            u_idx = random.choice(list(uncovered))
            candidates = blocks_containing_t[u_idx]
            
            best_block = -1
            max_gain = -1
            for b_idx in candidates:
                inds = block_indices[b_idx]
                gain = np.sum(counts[inds] == 0)
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            blocks.append(best_block)
            inds = block_indices[best_block]
            counts[inds] += 1
            
            # Update uncovered
            new_uncovered = set(idx for idx in uncovered if counts[idx] == 0)
            uncovered = new_uncovered
            
        return blocks, counts

    best_blocks = None
    best_counts = None
    best_len = float('inf')
    
    for _ in range(5):
        b, c = run_greedy()
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b
            best_counts = c
            
    if best_blocks is None:
        return [list(b) for b in all_k_subs[:best_len]]
        
    current_blocks = best_blocks
    current_counts = best_counts.copy()
    best_count = len(current_blocks)
    
    def remove_redundant(blocks, counts):
        new_blocks = []
        for b_idx in blocks:
            inds = block_indices[b_idx]
            counts[inds] -= 1
            if np.any(counts[inds] == 0):
                counts[inds] += 1
                new_blocks.append(b_idx)
        return new_blocks

    while time.time() < deadline - 5:
        new_blocks = remove_redundant(current_blocks, current_counts)
        
        if len(new_blocks) < best_count:
            best_count = len(new_blocks)
            current_blocks = new_blocks
        else:
            current_blocks = new_blocks
            
        if not current_blocks:
            break
            
        i = random.randint(0, len(current_blocks) - 1)
        old_b_idx = current_blocks[i]
        
        old_points = all_k_subs[old_b_idx]
        rand_t_sub = random.choice(list(itertools.combinations(old_points, t)))
        remaining_points = [p for p in points if p not in rand_t_sub]
        new_other_points = random.sample(remaining_points, k - t)
        new_points = tuple(sorted(rand_t_sub + new_other_points))
        new_b_idx = k_sub_map[new_points]
        
        old_inds = block_indices[old_b_idx]
        new_inds = block_indices[new_b_idx]
        
        current_counts[old_inds] -= 1
        current_counts[new_inds] += 1
        
        if np.any(current_counts < 1):
            current_counts[old_inds] += 1
            current_counts[new_inds] -= 1
            continue
            
        current_blocks[i] = new_b_idx

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
