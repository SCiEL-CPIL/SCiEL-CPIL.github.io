import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Bitmask for each block
    # Each bit i in the mask corresponds to t_subset i
    block_masks = [0] * num_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        
    all_ones_mask = (1 << num_t_subsets) - 1
    
    # 2. Simulated Annealing
    # Initial solution: Random set of blocks
    # We start with a size slightly above the expected optimum (e.g., 350)
    initial_size = 350
    current_blocks = set(random.sample(range(num_blocks), initial_size))
    current_mask = 0
    for b in current_blocks:
        current_mask |= block_masks[b]
        
    # Energy function: uncovered + lambda * size
    # We want to minimize size, but must have 0 uncovered.
    # Let's use Energy = uncovered_count * 1000 + size
    # This ensures we prioritize covering, then minimize size.
    
    def get_energy(mask, size):
        uncovered = (all_ones_mask ^ mask).bit_count()
        return uncovered * 1000 + size
        
    current_energy = get_energy(current_mask, len(current_blocks))
    best_energy = current_energy
    best_mask = current_mask
    best_blocks = set(current_blocks)
    
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.999
    
    steps = 0
    while time.time() - start_time < time_limit:
        steps += 1
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random block
            b_add = random.randint(0, num_blocks - 1)
            if b_add in current_blocks:
                continue
            new_mask = current_mask | block_masks[b_add]
            new_size = len(current_blocks) + 1
            new_energy = get_energy(new_mask, new_size)
            delta = new_energy - current_energy
        elif move_type < 0.6:
            # Remove a random block
            if not current_blocks:
                continue
            b_rem = random.choice(list(current_blocks))
            new_mask = current_mask & ~block_masks[b_rem]
            new_size = len(current_blocks) - 1
            new_energy = get_energy(new_mask, new_size)
            delta = new_energy - current_energy
        else:
            # Swap a random block with another random block
            if len(current_blocks) < 2:
                continue
            b_old = random.choice(list(current_blocks))
            b_new = random.randint(0, num_blocks - 1)
            if b_new in current_blocks:
                continue
            
            # Calculate delta efficiently
            # loss: bits in b_old that are NOT in b_new and are in current_mask
            # gain: bits in b_new that are NOT in current_mask
            # Note: b_old is in current_mask.
            # new_mask = (current_mask & ~mask[b_old]) | mask[b_new]
            
            # loss = (current_mask & mask[b_old] & ~mask[b_new]).bit_count()
            # But current_mask contains b_old.
            # Let's just compute new_mask and compare?
            # new_mask = (current_mask & ~block_masks[b_old]) | block_masks[b_new]
            # delta_uncovered = (current_mask ^ new_mask).bit_count() is not quite right for signed delta.
            # delta_uncovered = new_uncovered - old_uncovered
            # old_uncovered = (all_ones_mask ^ current_mask).bit_count()
            # new_uncovered = (all_ones_mask ^ new_mask).bit_count()
            
            new_mask = (current_mask & ~block_masks[b_old]) | block_masks[b_new]
            new_size = len(current_blocks) # Size unchanged
            new_energy = get_energy(new_mask, new_size)
            delta = new_energy - current_energy
            
            # If accepted, update state
            # We need to track which block was removed/added for the set
            # Let's handle state update inside the acceptance logic
            
        # Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_mask = new_mask
                # Reconstruct best_blocks? 
                # It's expensive to reconstruct from mask.
                # We should track the set of blocks.
                pass
            
            # Update current state
            if move_type < 0.3:
                current_blocks.add(b_add)
                current_mask = new_mask
            elif move_type < 0.6:
                current_blocks.remove(b_rem)
                current_mask = new_mask
            else:
                current_blocks.remove(b_old)
                current_blocks.add(b_new)
                current_mask = new_mask
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Extract best blocks
    # We need to find which blocks form best_mask.
    # This is a set cover problem again, but on the best_mask?
    # Actually, we tracked best_energy but not best_blocks directly in the SA loop easily.
    # Let's modify the SA loop to track best_blocks.
    
    # Refined SA Loop
    current_blocks = set(random.sample(range(num_blocks), initial_size))
    current_mask = 0
    for b in current_blocks:
        current_mask |= block_masks[b]
    current_energy = get_energy(current_mask, len(current_blocks))
    best_energy = current_energy
    best_blocks = set(current_blocks)
    
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.3:
            b_add = random.randint(0, num_blocks - 1)
            if b_add in current_blocks:
                continue
            new_mask = current_mask | block_masks[b_add]
            new_size = len(current_blocks) + 1
            new_energy = get_energy(new_mask, new_size)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_blocks.add(b_add)
                current_mask = new_mask
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = set(current_blocks)
                    
        elif move_type < 0.6:
            if not current_blocks:
                continue
            b_rem = random.choice(list(current_blocks))
            new_mask = current_mask & ~block_masks[b_rem]
            new_size = len(current_blocks) - 1
            new_energy = get_energy(new_mask, new_size)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_blocks.remove(b_rem)
                current_mask = new_mask
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = set(current_blocks)
                    
        else:
            if len(current_blocks) < 2:
                continue
            b_old = random.choice(list(current_blocks))
            b_new = random.randint(0, num_blocks - 1)
            if b_new in current_blocks:
                continue
            
            new_mask = (current_mask & ~block_masks[b_old]) | block_masks[b_new]
            new_size = len(current_blocks)
            new_energy = get_energy(new_mask, new_size)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_blocks.remove(b_old)
                current_blocks.add(b_new)
                current_mask = new_mask
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = set(current_blocks)
                    
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return [list(all_blocks[b]) for b in best_blocks]
