"""Covering design C(24,7,3) using optimized greedy + stochastic local search."""
import itertools
import random
import time
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer for safety
    
    # Precompute all t-subsets
    all_t_subsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # 1. Optimized Greedy Construction
    while uncovered:
        if time.time() - start_time > timeout:
            break
        
        seed_subset = min(uncovered)
        block = list(seed_subset)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() - start_time > timeout:
                break
            
            best_x = -1
            best_gain = -1
            pairs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for p in pairs:
                    tri = tuple(sorted((x, p[0], p[1])))
                    if tri in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain == len(pairs):
                        break
            
            if best_x == -1:
                break
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) < k:
            break
            
        block.sort()
        blocks.append(block)
        
        for comb in itertools.combinations(block, t):
            uncovered.discard(tuple(sorted(comb)))
            
        if time.time() - start_time > timeout:
            break
    
    # If greedy didn't finish, return what we have (likely invalid, score 0)
    if uncovered:
        return blocks

    # We have a valid covering in 'blocks'
    # Store it as the best valid solution so far
    best_valid_blocks = blocks[:]
    best_valid_size = len(best_valid_blocks)
    
    # 2. Stochastic Local Search (Simulated Annealing)
    # Objective: Minimize number of blocks.
    # We allow temporary invalid states but penalize them.
    
    current_blocks = blocks[:]
    current_uncovered = 0
    current_cost = len(current_blocks)
    
    def calculate_uncovered_count(block_list):
        covered = set()
        for b in block_list:
            for c in itertools.combinations(b, t):
                covered.add(tuple(sorted(c)))
        # Count how many of all_t_subsets are NOT in covered
        unc = 0
        for ts in all_t_subsets:
            if ts not in covered:
                unc += 1
        return unc

    def random_block():
        return sorted(random.sample(range(v), k))

    temp = 1.0
    cooling = 0.995
    min_temp = 0.01
    
    while temp > min_temp and time.time() - start_time < timeout - 5:
        # Generate neighbor
        move_type = random.randint(1, 3)
        new_blocks = current_blocks[:]
        
        if move_type == 1: # Remove
            if len(new_blocks) > 1:
                idx = random.randint(0, len(new_blocks) - 1)
                new_blocks.pop(idx)
        elif move_type == 2: # Add
            new_blocks.append(random_block())
        else: # Swap
            if len(new_blocks) > 0:
                idx = random.randint(0, len(new_blocks) - 1)
                new_blocks[idx] = random_block()
        
        # Evaluate
        unc_count = calculate_uncovered_count(new_blocks)
        new_cost = len(new_blocks) + 100 * unc_count
        
        # Acceptance
        if new_cost < current_cost:
            current_blocks = new_blocks
            current_uncovered = unc_count
            current_cost = new_cost
            
            # Update best valid if this state is valid and smaller
            if unc_count == 0 and len(new_blocks) < best_valid_size:
                best_valid_blocks = new_blocks[:]
                best_valid_size = len(new_blocks)
            
            temp = max(min_temp, temp * cooling)
        else:
            delta = new_cost - current_cost
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / temp)
            if random.random() < prob:
                current_blocks = new_blocks
                current_uncovered = unc_count
                current_cost = new_cost
                
                # Even if accepted due to SA, if it's valid and better, update best
                if unc_count == 0 and len(new_blocks) < best_valid_size:
                    best_valid_blocks = new_blocks[:]
                    best_valid_size = len(new_blocks)
                    
                temp = max(min_temp, temp * cooling)
            else:
                temp = max(min_temp, temp * cooling)

    # Return the best valid covering found
    return best_valid_blocks

# Test locally
if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, 3):
            covered.add(tuple(sorted(c)))
    
    all_subsets = set(itertools.combinations(range(24), 3))
    if covered == all_subsets:
        print("Valid covering")
    else:
        print("Invalid covering")
        print(f"Missing: {len(all_subsets - covered)}")
