import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast counting
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    def get_block_tsubs(block):
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    # 1. Greedy Construction (Randomized)
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = [0] * num_tsubs
        for i in range(num_tsubs):
            uncovered[i] = 1
            
        blocks = []
        # Pre-compute candidate blocks? No, too many.
        # Instead, for each uncovered t-subset, pick a random block containing it.
        # This is a "randomized greedy" or "randomized set cover".
        
        # To make it better, we can try to pick the block that covers the most *currently* uncovered t-subsets.
        # But calculating that for all candidates is expensive.
        # Heuristic: Pick a random uncovered t-subset. Pick a random block containing it.
        # This is fast but might be suboptimal.
        
        # Better Heuristic:
        # Maintain a list of uncovered t-subsets.
        # While uncovered list is not empty:
        #   Pick a random t-subset from the list.
        #   Generate all blocks containing this t-subset.
        #   For each candidate block, count how many *other* uncovered t-subsets it covers.
        #   Pick the best candidate.
        
        uncovered_set = set(range(num_tsubs))
        while uncovered_set:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered_set))
            target_tsub = all_tsubs[target_idx]
            
            # Generate all blocks containing target_tsub
            # target_tsub has t elements. We need to add k-t elements from the remaining v-t elements.
            remaining_points = [p for p in range(v) if p not in target_tsub]
            best_block = None
            best_gain = -1
            
            # Sample a few candidates to speed up, or iterate all if v is small enough
            # v=24, t=3, k=7. Remaining points = 21. Choose 4 from 21. C(21,4) = 5985.
            # This is small enough to iterate all candidates for the best block.
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_tsub + combo))
                # Calculate gain: how many uncovered t-subsets does this block cover?
                gain = 0
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx[ts]
                    if idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)
                    if best_gain == k: # Max possible gain for a block? No, max is C(k,t)
                        # Actually max gain is C(k,t) if all were uncovered.
                        # But we can break early if we find a perfect block?
                        pass
            
            if best_block is None:
                # Should not happen
                break
                
            blocks.append(best_block)
            # Update uncovered_set
            for ts in itertools.combinations(best_block, t):
                uncovered_set.discard(tsub_to_idx[ts])
            
            if time.time() > deadline - 30: # Check time
                break
                
        return blocks

    # 2. Local Search Minimization
    def local_search(blocks, seed):
        random.seed(seed)
        blocks = [list(b) for b in blocks]
        
        # Pre-compute t-subsets for each block
        block_tsubs = [get_block_tsubs(b) for b in blocks]
        
        # Coverage counts
        coverage = [0] * num_tsubs
        for tsubs in block_tsubs:
            for idx in tsubs:
                coverage[idx] += 1
        
        # Helper to check if a block is removable
        def is_removable(block_idx):
            for idx in block_tsubs[block_idx]:
                if coverage[idx] == 1:
                    return False
            return True
        
        # Helper to remove a block
        def remove_block(block_idx):
            for idx in block_tsubs[block_idx]:
                coverage[idx] -= 1
            del blocks[block_idx]
            del block_tsubs[block_idx]
            
        # Helper to add a block
        def add_block(block):
            blocks.append(block)
            tsubs = get_block_tsubs(block)
            block_tsubs.append(tsubs)
            for idx in tsubs:
                coverage[idx] += 1
                
        # Main loop
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Find all removable blocks
            removable_indices = [i for i in range(len(blocks)) if is_removable(i)]
            
            if removable_indices:
                # Remove one randomly
                idx_to_remove = random.choice(removable_indices)
                remove_block(idx_to_remove)
                improved = True
            else:
                # Stuck in local minimum. Perform a "kick" (perturbation)
                # Remove a random block (even if it breaks coverage)
                if not blocks:
                    break
                idx_to_remove = random.choice(range(len(blocks)))
                remove_block(idx_to_remove)
                
                # Repair: Greedily add blocks to cover uncovered t-subsets
                uncovered_set = set(i for i in range(num_tsubs) if coverage[i] == 0)
                while uncovered_set and time.time() < deadline - 5:
                    target_idx = random.choice(list(uncovered_set))
                    target_tsub = all_tsubs[target_idx]
                    remaining_points = [p for p in range(v) if p not in target_tsub]
                    best_block = None
                    best_gain = -1
                    
                    # Sample candidates to speed up repair
                    # C(21, 4) = 5985 is fine.
                    for combo in itertools.combinations(remaining_points, k - t):
                        block = tuple(sorted(target_tsub + combo))
                        gain = 0
                        for ts in itertools.combinations(block, t):
                            idx = tsub_to_idx[ts]
                            if idx in uncovered_set:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = list(block)
                            
                    if best_block is None:
                        break
                    add_block(best_block)
                    for ts in itertools.combinations(best_block, t):
                        uncovered_set.discard(tsub_to_idx[ts])
                        
        return blocks

    # 3. Main Execution
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple iterations with different seeds
    # Time budget: ~135s for search
    # We will run a loop that generates a new initial solution or continues local search
    
    # First, get a quick valid solution
    initial_blocks = greedy_construct(42)
    if not initial_blocks:
        # Fallback: simple greedy
        initial_blocks = []
        uncovered = set(all_tsubs)
        while uncovered:
            min_uncovered = min(uncovered)
            block = list(min_uncovered)
            # Add random points
            remaining = [p for p in range(v) if p not in block]
            random.seed(0)
            random.shuffle(remaining)
            block.extend(remaining[:k-t])
            block.sort()
            initial_blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
                
    best_blocks = initial_blocks
    best_size = len(best_blocks)
    
    # Local Search Phase
    # We will try to improve the best solution found so far.
    # We can also generate new initial solutions and local search them.
    
    num_iterations = 0
    while time.time() < deadline - 10:
        num_iterations += 1
        # Strategy: Alternate between improving current best and trying new starts
        if num_iterations % 5 == 0:
            # Try a new random start
            seed = num_iterations * 101
            new_blocks = greedy_construct(seed)
            if new_blocks:
                optimized = local_search(new_blocks, seed)
                if len(optimized) < best_size:
                    best_blocks = optimized
                    best_size = len(optimized)
        else:
            # Improve current best
            # Copy current best to avoid modifying it if search fails
            seed = num_iterations * 103
            current_copy = [list(b) for b in best_blocks]
            optimized = local_search(current_copy, seed)
            if len(optimized) < best_size:
                best_blocks = optimized
                best_size = len(optimized)
                
    return best_blocks
