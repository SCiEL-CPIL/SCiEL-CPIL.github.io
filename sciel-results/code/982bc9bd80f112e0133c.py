import itertools
import random
import time
import math

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_t_indices = []
    for block in all_blocks:
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(t_subset_indices[ts])
        block_t_indices.append(t_indices)
        
    # Precompute t_subset_to_blocks mapping
    # t_to_blocks[i] is a list of block indices that cover t_subset_indices[i]
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, block in enumerate(all_blocks):
        for t_idx in block_t_indices[b_idx]:
            t_to_blocks[t_idx].append(b_idx)
            
    # Helper: Greedy Set Cover Initialization
    def greedy_init():
        covered = [0] * num_t
        uncovered_count = num_t
        blocks = []
        
        # List of uncovered indices for fast random selection
        uncovered_indices = list(range(num_t))
        
        while uncovered_count > 0:
            # Pick a random uncovered t-subset
            target_idx = random.choice(uncovered_indices)
            
            # Find the best block among those covering target_idx
            # "Best" = covers the most currently uncovered t-subsets
            best_block_idx = -1
            max_gain = -1
            
            # To speed this up, we can sample if the list is huge, but 5985 is manageable
            for b_idx in t_to_blocks[target_idx]:
                gain = 0
                # Only count uncovered t-subsets
                for t_idx in block_t_indices[b_idx]:
                    if not covered[t_idx]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = b_idx
                    if gain == 35: # Max possible
                        break
            
            # Add the best block
            blocks.append(best_block_idx)
            
            # Update coverage
            for t_idx in block_t_indices[best_block_idx]:
                if not covered[t_idx]:
                    covered[t_idx] = 1
                    uncovered_count -= 1
            
            # Rebuild uncovered list (slow, but greedy init is only run a few times)
            # Optimization: Instead of rebuilding, we can just pick randomly from 0..num_t and check
            # But for correctness of "random choice from uncovered", we need the list or a check.
            # Given num_t is ~2000, rebuilding is O(2000). 76 steps * 2000 = 152k ops. Negligible.
            uncovered_indices = [i for i in range(num_t) if not covered[i]]
            
        return blocks

    # Helper: Local Search to minimize block count
    def local_search(initial_blocks):
        current_blocks = set(initial_blocks)
        
        # Initialize cover counts
        cover_count = [0] * num_t
        for b_idx in current_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        best_blocks = set(current_blocks)
        best_cost = len(current_blocks)
        
        start_time = time.time()
        time_limit = 110 # Leave 10s buffer
        
        # Precompute for each pair of t-subsets (i, j) the list of blocks covering both?
        # Or just use the single t-subset lists and filter.
        # Since critical set size is usually small, filtering is fast.
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to remove redundant blocks
            # Iterate over a random permutation of current blocks to avoid bias
            block_list = list(current_blocks)
            random.shuffle(block_list)
            
            for b_idx in block_list:
                if b_idx not in current_blocks:
                    continue
                # Check if b_idx is redundant
                redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    current_blocks.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    improved = True
                    if len(current_blocks) < best_cost:
                        best_cost = len(current_blocks)
                        best_blocks = set(current_blocks)
            
            # 2. If no block removed, try to swap to enable future removals
            if not improved:
                # Pick a random block to try swapping
                if not current_blocks:
                    break
                b_old = random.choice(list(current_blocks))
                
                # Find critical t-subsets for b_old
                critical_ts = []
                for t_idx in block_t_indices[b_old]:
                    if cover_count[t_idx] == 1:
                        critical_ts.append(t_idx)
                
                # If no critical, it's redundant (should have been caught above, but race condition or logic?)
                if not critical_ts:
                    # Force remove?
                    current_blocks.remove(b_old)
                    for t_idx in block_t_indices[b_old]:
                        cover_count[t_idx] -= 1
                    improved = True
                    if len(current_blocks) < best_cost:
                        best_cost = len(current_blocks)
                        best_blocks = set(current_blocks)
                    continue
                
                # Find a block b_new that covers all critical_ts
                # Start with blocks covering the first critical t-subset
                candidates = t_to_blocks[critical_ts[0]]
                
                found_swap = False
                # Limit candidates to avoid long searches if many
                # Shuffle candidates to avoid bias
                random.shuffle(candidates)
                
                for b_new in candidates:
                    if b_new in current_blocks:
                        continue
                    
                    # Check if b_new covers all other critical t-subsets
                    valid_swap = True
                    for t_idx in critical_ts[1:]:
                        if t_idx not in block_t_indices[b_new]:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Perform swap
                        # Remove b_old
                        for t_idx in block_t_indices[b_old]:
                            cover_count[t_idx] -= 1
                        # Add b_new
                        for t_idx in block_t_indices[b_new]:
                            cover_count[t_idx] += 1
                        
                        current_blocks.remove(b_old)
                        current_blocks.add(b_new)
                        found_swap = True
                        break
                
                if not found_swap:
                    # Try a random swap to escape local minima?
                    # Or just break if we can't find any swap?
                    # Let's try a few random b_new from the whole space if specific search fails?
                    # No, if no block covers the critical set, we can't swap b_old out directly.
                    # We might need to swap *other* blocks first.
                    # For simplicity, if we can't swap, we stop this iteration.
                    pass
        
        return list(best_blocks)

    # Main execution
    best_overall = float('inf')
    best_solution = None
    start_total = time.time()
    
    # Run multiple restarts of Greedy + Local Search
    while time.time() - start_total < 100:
        init_blocks = greedy_init()
        ls_result = local_search(init_blocks)
        
        if len(ls_result) < best_overall:
            best_overall = len(ls_result)
            best_solution = ls_result
            
        # Early stop if we hit lower bound
        if best_overall <= 69:
            break
            
    if best_solution is None:
        # Fallback
        init_blocks = greedy_init()
        best_solution = init_blocks
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[idx]) for idx in best_solution]
    
    return final_blocks
