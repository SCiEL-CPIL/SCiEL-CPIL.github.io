import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    m = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute t-subsets for each possible block (k-subset)
    # This is expensive but we'll compute lazily
    
    def get_block_t_subsets(block):
        """Get indices of t-subsets covered by a block"""
        return [subset_to_idx[s] for s in itertools.combinations(block, t)]
    
    # Greedy initialization
    def greedy_init():
        cov = np.zeros(m, dtype=np.int32)
        blocks = []
        
        for _ in range(k):
            # Find uncovered t-subsets
            uncovered = np.where(cov == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a random uncovered t-subset as seed
            seed_idx = random.choice(uncovered)
            seed_subset = set(all_t_subsets[seed_idx])
            
            # Build block greedily
            block = list(seed_subset)
            
            while len(block) < k:
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                
                best_x = None
                best_gain = -1
                
                for x in candidates:
                    # Count how many new t-subsets this would cover
                    gain = 0
                    for t_sub in itertools.combinations(block, t-1):
                        t_full = tuple(sorted(t_sub + (x,)))
                        idx = subset_to_idx[t_full]
                        if cov[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    best_x = random.choice(candidates)
                block.append(best_x)
            
            block.sort()
            for idx in get_block_t_subsets(block):
                cov[idx] += 1
            blocks.append(block)
        
        # Continue adding blocks until all covered
        while True:
            uncovered = np.where(cov == 0)[0]
            if len(uncovered) == 0:
                break
            
            seed_idx = random.choice(uncovered)
            seed_subset = set(all_t_subsets[seed_idx])
            
            block = list(seed_subset)
            while len(block) < k:
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                
                best_x = None
                best_gain = -1
                
                for x in candidates:
                    gain = 0
                    for t_sub in itertools.combinations(block, t-1):
                        t_full = tuple(sorted(t_sub + (x,)))
                        idx = subset_to_idx[t_full]
                        if cov[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    best_x = random.choice(candidates)
                block.append(best_x)
            
            block.sort()
            for idx in get_block_t_subsets(block):
                cov[idx] += 1
            blocks.append(block)
        
        return blocks, cov
    
    # Simulated annealing
    def simulated_annealing(blocks, cov, max_time):
        current_blocks = blocks[:]
        current_cov = cov.copy()
        current_cost = len(current_blocks)
        
        best_blocks = current_blocks[:]
        best_cov = current_cov.copy()
        best_cost = current_cost
        
        temperature = 1.0
        cooling_rate = 0.999
        min_temperature = 0.001
        
        while time.time() - start_time < max_time:
            if temperature < min_temperature:
                temperature = min_temperature
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.3 and len(current_blocks) > 0:
                # Try to remove a block
                i = random.randrange(len(current_blocks))
                block = current_blocks[i]
                
                # Check if any t-subset is only covered by this block
                can_remove = True
                for idx in get_block_t_subsets(block):
                    if current_cov[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove block
                    for idx in get_block_t_subsets(block):
                        current_cov[idx] -= 1
                    current_blocks.pop(i)
                    new_cost = len(current_blocks)
                    
                    # Accept (always improves)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = current_blocks[:]
                        best_cov = current_cov.copy()
                    current_cost = new_cost
                    
            elif move_type < 0.7:
                # Try to swap a block
                if len(current_blocks) == 0:
                    continue
                
                i = random.randrange(len(current_blocks))
                old_block = current_blocks[i]
                
                # Find critical t-subsets (only covered by this block)
                critical = []
                for idx in get_block_t_subsets(old_block):
                    if current_cov[idx] == 1:
                        critical.append(idx)
                
                if not critical:
                    # Block is redundant, remove it
                    for idx in get_block_t_subsets(old_block):
                        current_cov[idx] -= 1
                    current_blocks.pop(i)
                    new_cost = len(current_blocks)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = current_blocks[:]
                        best_cov = current_cov.copy()
                    current_cost = new_cost
                else:
                    # Try to find a replacement block that covers all critical t-subsets
                    # Generate candidate blocks that include points from critical subsets
                    critical_points = set()
                    for idx in critical:
                        critical_points.update(all_t_subsets[idx])
                    
                    # Try a few random replacements
                    for _ in range(5):
                        # Build new block starting from critical points
                        new_block = list(random.sample(list(critical_points), min(len(critical_points), k)))
                        
                        if len(new_block) < k:
                            remaining = [x for x in range(v) if x not in new_block]
                            new_block.extend(random.sample(remaining, k - len(new_block)))
                        
                        new_block.sort()
                        
                        # Check if this block is already in use
                        if new_block in current_blocks:
                            continue
                        
                        # Check if it covers all critical t-subsets
                        new_block_set = set(new_block)
                        valid = True
                        for idx in critical:
                            for x in all_t_subsets[idx]:
                                if x not in new_block_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        # Accept the swap
                        for idx in get_block_t_subsets(old_block):
                            current_cov[idx] -= 1
                        for idx in get_block_t_subsets(new_block):
                            current_cov[idx] += 1
                        
                        current_blocks[i] = new_block
                        # Cost stays the same, but we might have improved coverage
                        # Check if we can now remove other blocks
                        break
            
            else:
                # Try to add a block (rarely useful, but for completeness)
                # Find uncovered t-subsets
                uncovered = np.where(current_cov == 0)[0]
                if len(uncovered) > 0:
                    seed_idx = random.choice(uncovered)
                    seed_subset = set(all_t_subsets[seed_idx])
                    
                    new_block = list(seed_subset)
                    while len(new_block) < k:
                        remaining = [x for x in range(v) if x not in new_block]
                        new_block.append(random.choice(remaining))
                    
                    new_block.sort()
                    
                    for idx in get_block_t_subsets(new_block):
                        current_cov[idx] += 1
                    current_blocks.append(new_block)
                    current_cost = len(current_blocks)
            
            # Cool down
            temperature *= cooling_rate
        
        return best_blocks, best_cov
    
    # Main: run multiple iterations
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple independent trials
    num_trials = 5
    trial_time = (time_limit - (time.time() - start_time)) / num_trials
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(trial * 1000 + 42)
        
        blocks, cov = greedy_init()
        blocks, cov = simulated_annealing(blocks, cov, start_time + trial_time * (trial + 1) + 5)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
    
    return best_blocks
