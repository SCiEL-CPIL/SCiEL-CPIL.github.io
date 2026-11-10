import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Simulated Annealing.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Precompute all possible k-subsets? 
    # C(19, 7) = 50388. This is manageable.
    all_k_blocks = list(itertools.combinations(points, k))
    
    # Map t-subset to index for faster counting?
    # We will use a count array for uncovered subsets.
    
    def count_uncovered(blocks):
        """
        Returns the number of uncovered t-subsets and the list of blocks.
        This is O(N_blocks * C(k,t)).
        """
        # Use a set of covered t-subsets
        # For efficiency with small v, a boolean array or set of tuples is fine.
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        # Count how many of all_t_subsets are NOT in covered
        # This is O(N_t_subsets)
        uncovered_count = 0
        for ts in all_t_subsets:
            if ts not in covered:
                uncovered_count += 1
        return uncovered_count

    # Greedy Initialization
    def greedy_initial():
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            best_block = None
            max_cov = -1
            # Randomize order of candidates to break ties and add diversity
            candidates = random.sample(all_k_blocks, len(all_k_blocks))
            for block in candidates:
                cov = 0
                for ts in itertools.combinations(block, t):
                    if ts in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_block = block
                    if cov == len(uncovered): # Perfect cover for remaining
                        break
            if best_block is None:
                break
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)
        return blocks

    start_time = time.time()
    time_limit = 110
    
    # Initial Solution
    current_blocks = greedy_initial()
    current_uncovered = count_uncovered(current_blocks)
    current_cost = len(current_blocks) + 1000 * current_uncovered
    
    best_blocks = [list(b) for b in current_blocks]
    best_cost = current_cost
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            # Restart with a new greedy solution or perturb current one
            # Let's just restart the annealing schedule with a slightly perturbed state
            # or just break if we are stuck
            if random.random() < 0.5:
                current_blocks = greedy_initial()
                current_uncovered = count_uncovered(current_blocks)
                current_cost = len(current_blocks) + 1000 * current_uncovered
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [list(b) for b in current_blocks]
            T = 10.0
        
        # Generate Neighbor
        # Move types:
        # 1. Remove a random block (if any)
        # 2. Add a random block
        # 3. Replace a random block
        
        if len(current_blocks) > 0 and random.random() < 0.33:
            # Remove
            idx = random.randint(0, len(current_blocks) - 1)
            new_blocks = current_blocks[:idx] + current_blocks[idx+1:]
        elif random.random() < 0.5:
            # Add
            new_block = random.choice(all_k_blocks)
            new_blocks = current_blocks + [list(new_block)]
        else:
            # Replace
            idx = random.randint(0, len(current_blocks) - 1)
            new_block = random.choice(all_k_blocks)
            new_blocks = current_blocks[:idx] + [list(new_block)] + current_blocks[idx+1:]
        
        new_uncovered = count_uncovered(new_blocks)
        new_cost = len(new_blocks) + 1000 * new_uncovered
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_blocks = new_blocks
            current_uncovered = new_uncovered
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_blocks = [list(b) for b in current_blocks]
                
        T *= cooling_rate
        
    # Ensure the best solution is valid
    # If best_blocks is invalid, repair it
    if count_uncovered(best_blocks) > 0:
        uncovered = set(all_t_subsets)
        covered = set()
        for block in best_blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        for ts in all_t_subsets:
            if ts not in covered:
                uncovered.add(ts)
                
        # Greedily add blocks to cover remaining
        while uncovered:
            best_block = None
            max_cov = -1
            for block in all_k_blocks:
                cov = 0
                for ts in itertools.combinations(block, t):
                    if ts in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_block = block
            if best_block is None:
                break
            best_blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)
                
    return best_blocks
