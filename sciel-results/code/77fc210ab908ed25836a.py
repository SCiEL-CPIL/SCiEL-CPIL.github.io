import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # Helper to get subsets of a block
    def get_block_subsets(block):
        return [subset_to_idx[sub] for sub in itertools.combinations(block, t)]
    
    # Greedy construction to get a valid initial covering
    def greedy_construct():
        uncovered = [1] * num_subsets
        blocks = []
        remaining_points = list(range(v))
        
        # Pre-compute subsets for potential blocks? Too many.
        # Just pick a random uncovered subset and find a block covering it.
        
        while any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered subset
            unc_idx = [i for i, c in enumerate(uncovered) if c > 0]
            if not unc_idx:
                break
            target_idx = random.choice(unc_idx)
            target_subset = all_t_subsets[target_idx]
            
            # Find a block containing target_subset that covers the most uncovered subsets
            best_block = None
            best_gain = -1
            
            # Points in target_subset
            target_points = set(target_subset)
            remaining_for_block = [p for p in range(v) if p not in target_points]
            
            # We need to choose k-t points from remaining_for_block
            # To speed up, we can sample a few combinations if the space is large
            # C(21, 4) is 5985, which is small enough to iterate fully
            
            for combo in itertools.combinations(remaining_for_block, k - t):
                candidate_block = tuple(sorted(target_subset + combo))
                
                gain = 0
                for sub in itertools.combinations(candidate_block, t):
                    idx = subset_to_idx[sub]
                    if uncovered[idx] > 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered
            for sub in itertools.combinations(best_block, t):
                uncovered[subset_to_idx[sub]] -= 1
        
        return blocks

    # Initial solution
    best_blocks = greedy_construct()
    
    # Check if valid
    if not best_blocks:
        # Fallback if greedy failed
        return []

    # Calculate initial coverage counts
    def calculate_coverage(blocks):
        counts = [0] * num_subsets
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[subset_to_idx[sub]] += 1
        return counts

    current_coverage = calculate_coverage(best_blocks)
    current_count = len(best_blocks)
    best_count = current_count
    best_solution = best_blocks.copy()
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    # Pre-compute all possible blocks?
    # C(24, 7) = 346,104. This is large but manageable if we don't store all.
    # We can generate random blocks on the fly.
    
    def is_valid_coverage(counts):
        return all(c > 0 for c in counts)
    
    def try_remove_redundant(blocks, counts):
        # Returns new blocks, new counts, and number of blocks removed
        removed = 0
        new_blocks = []
        for b in blocks:
            is_redundant = True
            for sub in itertools.combinations(b, t):
                idx = subset_to_idx[sub]
                if counts[idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for sub in itertools.combinations(b, t):
                    idx = subset_to_idx[sub]
                    counts[idx] -= 1
                removed += 1
            else:
                new_blocks.append(b)
        
        return new_blocks, counts, removed

    # Main SA loop
    while time.time() < deadline and T > T_min:
        
        # Pick a move
        if random.random() < 0.5 and current_count > 0:
            # Try to swap a block
            i = random.randint(0, current_count - 1)
            old_block = best_blocks[i]
            
            # Remove old block
            temp_blocks = best_blocks[:i] + best_blocks[i+1:]
            temp_counts = current_coverage[:]
            
            for sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[sub]
                temp_counts[idx] -= 1
                
            # Check for uncovered
            uncovered_indices = [j for j, c in enumerate(temp_counts) if c == 0]
            
            if not uncovered_indices:
                # Valid, just need a new block
                # Generate a random new block
                new_block = tuple(sorted(random.sample(range(v), k)))
                
                # Avoid adding duplicate
                if new_block in temp_blocks:
                    # Pick another
                    while new_block in temp_blocks:
                        new_block = tuple(sorted(random.sample(range(v), k)))
                        
                temp_blocks.append(new_block)
                
                for sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[sub]
                    temp_counts[idx] += 1
                    
            else:
                # Need to cover the holes
                # Pick one uncovered subset
                target_idx = random.choice(uncovered_indices)
                target_subset = all_t_subsets[target_idx]
                
                # Find a block that covers target_subset and ideally other uncovered ones
                # Iterate through blocks containing target_subset
                target_points = set(target_subset)
                remaining_for_block = [p for p in range(v) if p not in target_points]
                
                best_new_block = None
                best_gain = -1
                
                for combo in itertools.combinations(remaining_for_block, k - t):
                    candidate_block = tuple(sorted(target_subset + combo))
                    
                    gain = 0
                    for sub in itertools.combinations(candidate_block, t):
                        idx = subset_to_idx[sub]
                        if temp_counts[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_new_block = candidate_block
                        
                if best_new_block:
                    temp_blocks.append(best_new_block)
                    for sub in itertools.combinations(best_new_block, t):
                        idx = subset_to_idx[sub]
                        temp_counts[idx] += 1
                else:
                    # Should not happen
                    continue

            # Try to remove redundant blocks from temp_blocks
            temp_blocks, temp_counts, removed = try_remove_redundant(temp_blocks, temp_counts)
            
            new_count = len(temp_blocks)
            delta_e = new_count - current_count
            
            # Accept/Reject
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                best_blocks = temp_blocks
                current_coverage = temp_counts
                current_count = new_count
                
                if current_count < best_count:
                    best_count = current_count
                    best_solution = best_blocks.copy()
                    
        else:
            # Try to add a block (to escape local minima)
            new_block = tuple(sorted(random.sample(range(v), k)))
            if new_block not in best_blocks:
                temp_blocks = best_blocks + [new_block]
                temp_counts = current_coverage[:]
                
                for sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[sub]
                    temp_counts[idx] += 1
                    
                # Try to remove redundant blocks
                temp_blocks, temp_counts, removed = try_remove_redundant(temp_blocks, temp_counts)
                
                new_count = len(temp_blocks)
                delta_e = new_count - current_count
                
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    best_blocks = temp_blocks
                    current_coverage = temp_counts
                    current_count = new_count
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_solution = best_blocks.copy()
                    
        T *= cooling
        
    return best_solution
