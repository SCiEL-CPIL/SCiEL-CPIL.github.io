import time
import random
import itertools
import math

def construct(v, k, t):
    """
    Uses Simulated Annealing to minimize the number of blocks in a covering design.
    1. Generate an initial valid covering using a fast greedy method.
    2. Iteratively attempt to remove blocks or swap blocks to reduce the count.
    3. Accept worse solutions with a probability based on temperature to escape local minima.
    """
    deadline = time.time() + 140.0  # Safety margin for 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Helper to check if a block covers a specific t-subset
    # We represent coverage as a set of indices of covered t-subsets
    
    def greedy_initial(seed):
        """Fast greedy construction to get a valid starting point."""
        rng = random.Random(seed)
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            start_subset = all_t_subsets[start_idx]
            
            # Build a block containing this subset
            block = list(start_subset)
            block_set = set(block)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                # Precompute (t-1) subsets of current block for gain calculation
                if len(block) >= t - 1:
                    block_subsets = set(itertools.combinations(block, t - 1))
                else:
                    block_subsets = set()
                
                for x in candidates:
                    gain = 0
                    for sub in block_subsets:
                        new_t_sub = tuple(sorted(sub + (x,)))
                        if new_t_sub in t_subset_to_idx:
                            idx = t_subset_to_idx[new_t_sub]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    # Fill with random points if no gain
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                # Update uncovered
                for sub in itertools.combinations(block, t):
                    idx = t_subset_to_idx[sub]
                    uncovered.discard(idx)
        
        return blocks

    # Generate initial solution
    current_blocks = greedy_initial(42)
    
    # Verify validity of initial solution
    def is_valid(blocks):
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        return len(covered) == n_t_subsets

    if not is_valid(current_blocks):
        # Fallback if greedy failed (should be rare)
        current_blocks = greedy_initial(1)
        if not is_valid(current_blocks):
            return current_blocks # Return best effort

    best_blocks = list(current_blocks)
    best_count = len(current_blocks)
    
    # Precompute coverage counts for the current set of blocks
    # coverage_count[i] = number of blocks covering t-subset i
    def compute_coverage_counts(blocks):
        counts = [0] * n_t_subsets
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                counts[idx] += 1
        return counts

    current_coverage = compute_coverage_counts(current_blocks)
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    max_steps = 100000
    
    rng = random.Random(123)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        if T < min_T:
            break
            
        # Choose a move:
        # 1. Try to remove a random block (if it doesn't break validity)
        # 2. Try to swap a random block with a new random block (if it improves or is accepted)
        
        move_type = rng.random()
        
        if move_type < 0.5 and len(current_blocks) > 0:
            # Attempt removal
            idx_to_remove = rng.randrange(len(current_blocks))
            block_to_remove = current_blocks[idx_to_remove]
            
            # Check if any t-subset becomes uncovered
            is_safe = True
            for sub in itertools.combinations(block_to_remove, t):
                idx = t_subset_to_idx[sub]
                if current_coverage[idx] == 1:
                    is_safe = False
                    break
            
            if is_safe:
                # Apply removal
                current_blocks.pop(idx_to_remove)
                current_coverage = compute_coverage_counts(current_blocks) # Recalc is slow, optimize later? 
                # For now, recalc is safe. v=24, k=7, blocks~70. 70*35=2450 updates.
                # Actually, we can update incrementally.
                
                # Let's do incremental update for speed
                # We already popped. Let's reverse the logic: 
                # If we accept, we update. If we reject, we revert.
                # It's easier to just check validity locally.
                
                # Let's refine: 
                # The check above `is_safe` is O(k choose t). 
                # If safe, the new count is len-1.
                new_count = len(current_blocks)
                
                # Accept always if it reduces count (greedy improvement)
                # Or accept with prob exp(-delta/T) if it increases count (but removal always decreases)
                
                # Since removal always decreases count, we accept if valid.
                # We already checked validity.
                
                # Update coverage counts incrementally
                for sub in itertools.combinations(block_to_remove, t):
                    idx = t_subset_to_idx[sub]
                    current_coverage[idx] -= 1
                
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
            else:
                # Revert (did nothing)
                pass

        else:
            # Attempt swap: Replace block i with a new random block j
            if len(current_blocks) == 0:
                continue
                
            idx_to_swap = rng.randrange(len(current_blocks))
            old_block = current_blocks[idx_to_swap]
            
            # Generate a new random block
            new_block = rng.sample(range(v), k)
            new_block.sort()
            
            # If new block is same as old, skip
            if new_block == old_block:
                continue
            
            # Calculate change in coverage
            # We need to know if removing old_block and adding new_block maintains validity.
            # Validity condition: For all t-subsets, count >= 1.
            # Current counts are in current_coverage.
            
            # Remove old block from counts
            old_subsets = list(itertools.combinations(old_block, t))
            new_subsets = list(itertools.combinations(new_block, t))
            
            # Check if any subset in old_block becomes uncovered (count becomes 0)
            # AND is not covered by new_block
            # AND is not covered by other blocks (count > 1 before removal)
            
            is_valid_swap = True
            # First, simulate removal
            for sub in old_subsets:
                idx = t_subset_to_idx[sub]
                if current_coverage[idx] == 1:
                    # This subset is ONLY covered by old_block.
                    # It must be covered by new_block.
                    if sub not in new_subsets:
                        is_valid_swap = False
                        break
            
            if not is_valid_swap:
                continue
                
            # If valid swap, calculate "cost" or "energy"
            # Energy = number of blocks. Swap keeps count same.
            # So delta_E = 0. We should accept based on some diversity metric?
            # Or just accept with prob 1 to explore?
            # In SA for minimization with fixed cost moves, we often use a secondary objective
            # or just random walk. 
            # However, swaps allow us to reach states where a removal becomes possible.
            
            # Apply swap
            current_blocks[idx_to_swap] = new_block
            
            # Update coverage counts
            for sub in old_subsets:
                idx = t_subset_to_idx[sub]
                current_coverage[idx] -= 1
            for sub in new_subsets:
                idx = t_subset_to_idx[sub]
                current_coverage[idx] += 1
            
            # After swap, check if we can now remove a block?
            # We'll let the next iteration handle removals.
            
            # Cooling
            T *= cooling_rate
            
    return best_blocks

# Optimization: The incremental coverage update in the loop above is correct but 
# the `compute_coverage_counts` inside the removal branch was removed in thought process, 
# let's ensure the code below is clean and efficient.
# I will rewrite the loop part to be robust.

def construct(v, k, t):
    deadline = time.time() + 140.0
    start_time = time.time()
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    def greedy_initial(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        while uncovered and time.time() < deadline:
            start_idx = rng.choice(list(uncovered))
            start_subset = all_t_subsets[start_idx]
            
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                if len(block) >= t - 1:
                    block_subsets = set(itertools.combinations(block, t - 1))
                else:
                    block_subsets = set()
                
                for x in candidates:
                    gain = 0
                    for sub in block_subsets:
                        new_t_sub = tuple(sorted(sub + (x,)))
                        if new_t_sub in t_subset_to_idx:
                            idx = t_subset_to_idx[new_t_sub]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    idx = t_subset_to_idx[sub]
                    uncovered.discard(idx)
        
        return blocks

    current_blocks = greedy_initial(42)
    
    def is_valid(blocks):
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        return len(covered) == n_t_subsets

    if not is_valid(current_blocks):
        current_blocks = greedy_initial(1)
        if not is_valid(current_blocks):
            return current_blocks

    best_blocks = list(current_blocks)
    best_count = len(current_blocks)
    
    def compute_coverage_counts(blocks):
        counts = [0] * n_t_subsets
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                counts[idx] += 1
        return counts

    current_coverage = compute_coverage_counts(current_blocks)
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    rng = random.Random(123)
    
    # Main SA Loop
    while time.time() < deadline:
        # 1. Try Removal
        if len(current_blocks) > 0 and rng.random() < 0.3:
            idx = rng.randrange(len(current_blocks))
            block = current_blocks[idx]
            
            # Check if removal is safe
            safe = True
            for sub in itertools.combinations(block, t):
                idx_sub = t_subset_to_idx[sub]
                if current_coverage[idx_sub] == 1:
                    safe = False
                    break
            
            if safe:
                # Perform removal
                current_blocks.pop(idx)
                for sub in itertools.combinations(block, t):
                    idx_sub = t_subset_to_idx[sub]
                    current_coverage[idx_sub] -= 1
                
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                
                # Cooling on improvement
                T = max(min_T, T * 0.9)
            else:
                # Rejection
                T *= cooling_rate
        else:
            # 2. Try Swap
            if len(current_blocks) == 0:
                continue
                
            idx = rng.randrange(len(current_blocks))
            old_block = current_blocks[idx]
            
            # Generate new block
            # Bias towards points that are currently under-covered? 
            # For now, uniform random.
            new_block = rng.sample(range(v), k)
            new_block.sort()
            
            if new_block == old_block:
                T *= cooling_rate
                continue
            
            old_subs = list(itertools.combinations(old_block, t))
            new_subs = list(itertools.combinations(new_block, t))
            
            # Check validity of swap
            # A subset becomes invalid if:
            # 1. It was in old_block, not in new_block, and had count 1.
            valid = True
            for sub in old_subs:
                if sub not in new_subs:
                    idx_sub = t_subset_to_idx[sub]
                    if current_coverage[idx_sub] == 1:
                        valid = False
                        break
            
            if not valid:
                T *= cooling_rate
                continue
            
            # If valid, accept swap (delta energy is 0, but it changes landscape for removals)
            # We can also add a small bias: accept with prob 1 always for valid swaps?
            # Yes, to explore.
            
            # Apply swap
            current_blocks[idx] = new_block
            
            for sub in old_subs:
                idx_sub = t_subset_to_idx[sub]
                current_coverage[idx_sub] -= 1
            for sub in new_subs:
                idx_sub = t_subset_to_idx[sub]
                current_coverage[idx_sub] += 1
            
            T *= cooling_rate
            
    return best_blocks

