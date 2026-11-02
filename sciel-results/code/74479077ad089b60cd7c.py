import itertools
import random
import time
import math

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    m = len(all_t_subsets)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Helper to get t-subsets of a block
    def get_block_t_subsets(block):
        return list(itertools.combinations(block, t))

    # Helper to compute coverage vector for a list of blocks
    def compute_coverage(blocks):
        cov = [0] * m
        for block in blocks:
            for t_sub in itertools.combinations(block, t):
                idx = subset_to_idx[t_sub]
                cov[idx] += 1
        return cov

    # Helper to evaluate a state (blocks, cov)
    # Objective: Minimize (num_blocks, num_uncovered)
    # We use a weighted sum for SA: Cost = num_blocks * weight_blocks + num_uncovered * weight_uncovered
    # To prioritize validity, weight_uncovered should be very high.
    
    def evaluate(blocks, cov):
        num_blocks = len(blocks)
        num_uncovered = sum(1 for c in cov if c == 0)
        # We want to minimize num_blocks first, then num_uncovered?
        # Actually, we need a valid covering (num_uncovered = 0).
        # So Cost = num_uncovered * 10000 + num_blocks
        return num_uncovered * 10000 + num_blocks

    # Initial Construction: Greedy
    def greedy_construct():
        cov = [0] * m
        blocks = []
        
        while True:
            min_cov_val = min(cov)
            if min_cov_val > 0:
                break
            
            candidates = [i for i in range(m) if cov[i] == min_cov_val]
            idx = random.choice(candidates)
            seed_subset = all_t_subsets[idx]
            
            block = list(seed_subset)
            
            while len(block) < k:
                best_gain = -1
                best_x = -1
                candidates_x = [x for x in range(v) if x not in block]
                block_t_minus_1 = list(itertools.combinations(block, t-1))
                
                for x in candidates_x:
                    gain = 0
                    for s_prime in block_t_minus_1:
                        t_sub = tuple(sorted(s_prime + (x,)))
                        i = subset_to_idx[t_sub]
                        if cov[i] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    
                if best_x == -1:
                    best_x = random.choice(candidates_x)
                block.append(best_x)
            
            block.sort()
            for t_sub in itertools.combinations(block, t):
                i = subset_to_idx[t_sub]
                cov[i] += 1
            blocks.append(block)
            
        return blocks, cov

    # Simulated Annealing
    def simulated_annealing(initial_blocks, initial_cov):
        blocks = initial_blocks[:]
        cov = initial_cov[:]
        current_cost = evaluate(blocks, cov)
        best_blocks = blocks[:]
        best_cov = cov[:]
        best_cost = current_cost
        
        # Temperature parameters
        T_start = 100.0
        T_end = 0.1
        alpha = 0.995 # Cooling rate
        
        T = T_start
        
        # Precompute block t-subsets for speed? 
        # We need to update cov incrementally.
        
        while T > T_end:
            # Determine move type
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a block
                new_block = sorted(random.sample(range(v), k))
                if new_block in blocks:
                    continue
                blocks.append(new_block)
                # Update cov
                for t_sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[t_sub]
                    cov[idx] += 1
                new_cost = evaluate(blocks, cov)
                
                # Acceptance
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = blocks[:]
                        best_cov = cov[:]
                else:
                    # Reject: Revert
                    blocks.pop()
                    for t_sub in itertools.combinations(new_block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] -= 1
                        
            elif move_type < 0.6:
                # Remove a block
                if len(blocks) == 0:
                    continue
                i = random.randrange(len(blocks))
                old_block = blocks[i]
                blocks.pop(i)
                # Update cov
                for t_sub in itertools.combinations(old_block, t):
                    idx = subset_to_idx[t_sub]
                    cov[idx] -= 1
                new_cost = evaluate(blocks, cov)
                
                # Acceptance
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = blocks[:]
                        best_cov = cov[:]
                else:
                    # Reject: Revert
                    blocks.insert(i, old_block)
                    for t_sub in itertools.combinations(old_block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] += 1

            else:
                # Replace a block
                if len(blocks) == 0:
                    continue
                i = random.randrange(len(blocks))
                old_block = blocks[i]
                new_block = sorted(random.sample(range(v), k))
                if new_block == old_block:
                    continue
                if new_block in blocks:
                    continue
                
                blocks[i] = new_block
                # Update cov: remove old, add new
                for t_sub in itertools.combinations(old_block, t):
                    idx = subset_to_idx[t_sub]
                    cov[idx] -= 1
                for t_sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[t_sub]
                    cov[idx] += 1
                new_cost = evaluate(blocks, cov)
                
                # Acceptance
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = blocks[:]
                        best_cov = cov[:]
                else:
                    # Reject: Revert
                    blocks[i] = old_block
                    for t_sub in itertools.combinations(old_block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] += 1
                    for t_sub in itertools.combinations(new_block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] -= 1

            T *= alpha
            
            # Time check
            if time.time() - start_time > time_limit - 10:
                break

        return best_blocks, best_cov

    # Main Loop: Multiple restarts
    best_blocks = None
    best_size = float('inf')
    
    # Run a few restarts
    for run in range(5):
        if time.time() - start_time > time_limit - 20:
            break
            
        random.seed(run * 12345 + 6789)
        
        blocks, cov = greedy_construct()
        blocks, cov = simulated_annealing(blocks, cov)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    return best_blocks
