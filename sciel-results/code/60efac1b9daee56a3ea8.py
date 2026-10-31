import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Simulated Annealing.
    """
    
    # Precompute all t-subsets for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map subset to index for faster coverage tracking
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Coverage array: counts how many blocks cover each t-subset
    # Length is C(v, t)
    num_subsets = len(all_t_subsets)
    
    def get_block_subsets(block):
        """Yield indices of t-subsets covered by a block."""
        for sub in itertools.combinations(block, t):
            yield subset_to_idx[sub]
            
    def greedy_cover():
        """
        Fast greedy initialization.
        Returns a list of blocks (tuples).
        """
        uncovered = [1] * num_subsets
        blocks = []
        
        while any(uncovered):
            # Find a random uncovered subset to anchor the block
            # Optimization: Instead of random.choice on list, we can pick random indices
            # but for simplicity and speed in Python, we might just pick a random point set
            # or a random uncovered subset.
            
            # To be fast, let's pick a random uncovered subset
            # This might be slow if we scan every time. 
            # Better: Keep a list of uncovered indices?
            # For v=19, t=4, C(19,4) = 3876. Scanning is fast.
            uncov_indices = [i for i, c in enumerate(uncovered) if c > 0]
            if not uncov_indices:
                break
                
            target_idx = random.choice(uncov_indices)
            target_sub = all_t_subsets[target_idx]
            
            # Start block with target_sub
            block = list(target_sub)
            
            # Greedily add points to cover most uncovered subsets
            while len(block) < k:
                best_point = None
                best_gain = -1
                # Precompute (t-1) subsets of current block
                current_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                # Candidate points not in block
                # Optimization: v is small (19), so iterating all is fine
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in current_t_minus_1:
                        new_subset = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx[new_subset]
                        if uncovered[idx] > 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point is None:
                    # Fallback
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        break
                block.append(best_point)
                
            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)
            
            # Update uncovered
            for idx in get_block_subsets(block_tuple):
                uncovered[idx] -= 1
                if uncovered[idx] < 0:
                    uncovered[idx] = 0 # Safety, shouldn't happen if logic is tight

        return blocks

    def calculate_cost(blocks, uncovered):
        """
        Cost = Number of blocks + Penalty * Uncovered Count
        """
        num_uncovered = sum(1 for c in uncovered if c == 0)
        # Heavy penalty for uncovered subsets to ensure validity is prioritized
        # But we want to minimize blocks. 
        # If valid, cost = len(blocks).
        # If invalid, cost = len(blocks) + 1000 * num_uncovered.
        return len(blocks) + 1000 * num_uncovered

    def simulated_annealing(initial_blocks, time_limit=110):
        """
        SA Optimizer.
        """
        # State
        current_blocks = list(initial_blocks)
        current_uncovered = [0] * num_subsets
        for b in current_blocks:
            for idx in get_block_subsets(b):
                current_uncovered[idx] += 1
        
        current_cost = calculate_cost(current_blocks, current_uncovered)
        
        best_blocks = list(current_blocks)
        best_uncovered = list(current_uncovered)
        best_cost = current_cost
        
        start_time = time.time()
        T = 100.0
        cooling_rate = 0.99
        min_T = 0.1
        
        # Move generators
        def try_remove_block(blocks, uncovered):
            """Try to remove a random block if valid."""
            if not blocks:
                return None, None, False
            idx = random.randint(0, len(blocks) - 1)
            b = blocks[idx]
            # Check if valid to remove
            valid = True
            for sub_idx in get_block_subsets(b):
                if uncovered[sub_idx] <= 1:
                    valid = False
                    break
            if valid:
                new_blocks = blocks[:]
                new_blocks.pop(idx)
                new_uncovered = uncovered[:]
                for sub_idx in get_block_subsets(b):
                    new_uncovered[sub_idx] -= 1
                return new_blocks, new_uncovered, True
            return None, None, False

        def try_replace_block(blocks, uncovered):
            """Try to replace a random block with a random new block."""
            if not blocks:
                return None, None, False
            idx = random.randint(0, len(blocks) - 1)
            old_b = blocks[idx]
            
            # Generate random new block
            new_b = tuple(sorted(random.sample(range(v), k)))
            
            # Check validity of replacement
            # 1. Subsets unique to old_b must be covered by new_b
            # 2. We don't strictly need new_b to cover everything old_b covered, 
            #    but if it doesn't, those subsets become uncovered (penalized).
            #    However, for efficiency, we prefer replacements that don't increase uncovered count.
            
            # Let's just check if it's a "good" swap: 
            # If a subset is covered ONLY by old_b, new_b MUST cover it.
            valid = True
            for sub_idx in get_block_subsets(old_b):
                if uncovered[sub_idx] == 1:
                    # This subset is unique to old_b
                    # Check if new_b covers it
                    # We need to check if the subset is in new_b
                    # Efficient check:
                    # The subset is all_t_subsets[sub_idx]
                    sub = all_t_subsets[sub_idx]
                    # Check if sub is subset of new_b
                    # Since k=7, t=4, we can just check membership
                    # But sub is a tuple of 4 ints. new_b is tuple of 7 ints.
                    # all(x in new_b for x in sub)
                    if not all(x in new_b for x in sub):
                        valid = False
                        break
            
            if not valid:
                return None, None, False
            
            # Perform swap
            new_blocks = blocks[:]
            new_uncovered = uncovered[:]
            
            # Remove old_b coverage
            for sub_idx in get_block_subsets(old_b):
                new_uncovered[sub_idx] -= 1
                
            # Add new_b coverage
            for sub_idx in get_block_subsets(new_b):
                new_uncovered[sub_idx] += 1
                
            new_blocks[idx] = new_b
            return new_blocks, new_uncovered, True

        def try_add_block(blocks, uncovered):
            """Add a random block. Useful if we are stuck with high uncovered count."""
            new_b = tuple(sorted(random.sample(range(v), k)))
            new_blocks = blocks + [new_b]
            new_uncovered = uncovered[:]
            for sub_idx in get_block_subsets(new_b):
                new_uncovered[sub_idx] += 1
            return new_blocks, new_uncovered, True

        def try_remove_redundant(blocks, uncovered):
            """Remove all redundant blocks."""
            new_blocks = []
            new_uncovered = list(uncovered)
            changed = False
            for b in blocks:
                redundant = True
                for sub_idx in get_block_subsets(b):
                    if new_uncovered[sub_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                    for sub_idx in get_block_subsets(b):
                        new_uncovered[sub_idx] -= 1
                else:
                    new_blocks.append(b)
            if changed:
                return new_blocks, new_uncovered, True
            return None, None, False

        while time.time() - start_time < time_limit and T > min_T:
            # Choose a move
            move_type = random.random()
            new_blocks = None
            new_uncovered = None
            success = False
            
            if move_type < 0.4:
                new_blocks, new_uncovered, success = try_remove_block(current_blocks, current_uncovered)
            elif move_type < 0.8:
                new_blocks, new_uncovered, success = try_replace_block(current_blocks, current_uncovered)
            elif move_type < 0.9:
                new_blocks, new_uncovered, success = try_add_block(current_blocks, current_uncovered)
            else:
                new_blocks, new_uncovered, success = try_remove_redundant(current_blocks, current_uncovered)
                
            if success and new_blocks is not None:
                new_cost = calculate_cost(new_blocks, new_uncovered)
                delta_cost = new_cost - current_cost
                
                # Acceptance criterion
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    current_blocks = new_blocks
                    current_uncovered = new_uncovered
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        best_uncovered = list(current_uncovered)
                        
            T *= cooling_rate
            
        # Final cleanup
        blocks, uncovered, _ = try_remove_redundant(best_blocks, best_uncovered)
        if blocks:
            return blocks
        return best_blocks

    # Main Execution
    # Run greedy to get a valid start
    initial_blocks = greedy_cover()
    
    # Run SA
    optimized_blocks = simulated_annealing(initial_blocks, time_limit=110)
    
    # Ensure we return a list of lists
    return [list(b) for b in optimized_blocks]
