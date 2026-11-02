import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the list of t-subset IDs it covers
    block_t_ids = []
    for block in all_k_subsets:
        ids = [t_subset_to_id[s] for s in itertools.combinations(block, t)]
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)

    # 1. Greedy Construction for Initial Solution
    # We use a standard greedy set cover: pick the block that covers the most uncovered elements.
    def greedy_cover():
        is_covered = [False] * num_t_subsets
        selected_blocks = set()
        uncovered_count = num_t_subsets
        
        while uncovered_count > 0:
            # Find the block that covers the most uncovered t-subsets
            best_block = -1
            max_new_cover = -1
            
            # Optimization: Instead of checking all k-subsets, we can be smarter,
            # but for v=19, k=8, num_k_subsets ~ 75k, and num_t_subsets ~ 11k.
            # Checking all blocks every step is O(N_blocks * C(k,t)).
            # C(8,5) = 56. 75k * 56 ~ 4M ops per step.
            # If we have ~300 blocks, 300 * 4M = 1.2B ops. Too slow in Python.
            
            # Better Greedy:
            # Pick a random uncovered t-subset.
            # Find the best block covering it that covers the most *other* uncovered subsets.
            # This is O(N_candidates * C(k,t)).
            # N_candidates is the number of blocks covering a specific t-subset.
            # A specific t-subset (5 points) is contained in C(19-5, 8-5) = C(14,3) = 364 blocks.
            # So we check ~364 blocks. 364 * 56 ~ 20k ops per step.
            # 300 steps * 20k = 6M ops. Very fast.
            
            # To diversify or improve, we can pick the uncovered t-subset with the fewest candidates?
            # Or just random. Let's use random to avoid bias.
            
            # Find an uncovered t-subset
            # We maintain a list of uncovered indices? Updating it is expensive.
            # Just scan is_covered.
            uncovered_indices = [i for i in range(num_t_subsets) if not is_covered[i]]
            if not uncovered_indices:
                break
                
            target_tid = random.choice(uncovered_indices)
            
            candidates = t_to_blocks[target_tid]
            
            for b_idx in candidates:
                if b_idx in selected_blocks:
                    continue
                new_cover = 0
                for tid in block_t_ids[b_idx]:
                    if not is_covered[tid]:
                        new_cover += 1
                if new_cover > max_new_cover:
                    max_new_cover = new_cover
                    best_block = b_idx
            
            if best_block != -1:
                selected_blocks.add(best_block)
                for tid in block_t_ids[best_block]:
                    if not is_covered[tid]:
                        uncovered_count -= 1
                    is_covered[tid] = True
            else:
                # Should not happen
                break
                
        return selected_blocks

    # Get initial solution
    initial_solution = greedy_cover()
    
    if not initial_solution:
        return [list(all_k_subsets[0])]
    
    # 2. Local Search Optimization
    # State: set of active blocks
    active_blocks = set(initial_solution)
    
    # Maintain cover counts
    cover_counts = [0] * num_t_subsets
    for b_idx in active_blocks:
        for tid in block_t_ids[b_idx]:
            cover_counts[tid] += 1
            
    uncovered_count = sum(1 for c in cover_counts if c == 0)
    
    best_solution = set(active_blocks)
    best_size = len(active_blocks)
    
    # Local Search Loop
    # We perform two types of moves:
    # 1. Remove: If a block is redundant (removing it doesn't uncover any t-subset), remove it.
    # 2. Swap: Replace a block in the solution with a block not in the solution if it allows
    #    for a future removal or directly improves the "potential" for removal.
    #    Specifically, we look for a swap that results in a state where a block becomes removable.
    #    Or simpler: Try to swap any block in with any block out if the new state has the same size
    #    but allows more removals? That's hard to measure.
    
    # Alternative Local Search:
    # Iteratively scan all blocks in the solution. If it can be removed, remove it.
    # If no block can be removed, try swaps.
    # For each block in the solution, try swapping it with a random block not in the solution.
    # If the new state allows at least one removal (i.e., some block in the new state is redundant),
    # accept the swap. Then try to remove redundant blocks again.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Remove redundant blocks
        # A block is redundant if for every t-subset it covers, that t-subset is covered by at least one other block.
        # i.e., cover_counts[tid] > 1 for all tid in block_t_ids[b].
        
        blocks_to_remove = []
        for b_idx in list(active_blocks):
            redundant = True
            for tid in block_t_ids[b_idx]:
                if cover_counts[tid] <= 1:
                    redundant = False
                    break
            if redundant:
                blocks_to_remove.append(b_idx)
        
        for b_idx in blocks_to_remove:
            active_blocks.remove(b_idx)
            for tid in block_t_ids[b_idx]:
                cover_counts[tid] -= 1
            uncovered_count += 0 # It was redundant, so uncovered_count doesn't change (it was 0 and stays 0)
            improved = True
            
        if len(active_blocks) < best_size and uncovered_count == 0:
            best_size = len(active_blocks)
            best_solution = set(active_blocks)
            
        if not improved:
            # Phase 2: Try Swaps
            # Pick a random block in active_blocks
            # Pick a random block not in active_blocks
            # If the swap results in a state where a block can be removed, accept it.
            
            if not active_blocks:
                break
                
            b_in = random.choice(list(active_blocks))
            b_out = random.randint(0, num_k_subsets - 1)
            
            if b_out in active_blocks:
                continue
                
            # Simulate swap
            # Remove b_in, Add b_out
            # Check if any block in the new set is redundant.
            
            # To check if a block is redundant, we need the new cover counts.
            # new_counts[tid] = cover_counts[tid] - (1 if tid in b_in else 0) + (1 if tid in b_out else 0)
            
            # We don't want to build the whole new_counts array (size 11k) for every swap attempt.
            # We can just check if the swap *creates* a redundancy for some block.
            # A block X is redundant in the new state if for all tid in X, new_counts[tid] > 1.
            
            # This is still expensive to check for all X.
            
            # Heuristic: Accept the swap if it reduces the "minimum cover count" or something?
            # Let's try a simpler swap acceptance:
            # Accept if the new state allows removing at least one block.
            # To check this efficiently:
            # The only blocks whose redundancy status might change are those in b_in, b_out, or blocks that share t-subsets with b_in/b_out.
            # Actually, removing b_in decreases counts. Adding b_out increases counts.
            # A block X becomes redundant if all its t-subsets have count > 1 in the new state.
            
            # Let's just do a limited check.
            # If the swap is accepted, we will immediately run Phase 1 again.
            
            # How to decide to accept the swap?
            # Let's accept the swap with a certain probability, or if it seems "good".
            # A good swap might be one where b_out covers more "critical" (low count) t-subsets than b_in.
            
            # Calculate "utility" of b_in and b_out.
            # Utility = sum(1/cover_counts[tid] for tid in block) ?
            # If b_out has higher utility, it's a better candidate.
            
            # Let's just try the swap and see if it leads to an improvement in the next iteration.
            # To avoid getting stuck, we can use a random restart or random swap acceptance.
            # Let's accept the swap with probability 0.1, or always if it looks promising.
            
            # Let's calculate the number of t-subsets where count becomes 1 (critical) vs 0 (bad).
            # Removing b_in:
            #   If count[tid] == 1, it becomes 0 (BAD, invalid).
            #   If count[tid] == 2, it becomes 1 (CRITICAL).
            # Adding b_out:
            #   If count[tid] == 0, it becomes 1 (GOOD, fixes bad).
            #   If count[tid] == 1, it becomes 2 (GOOD, fixes critical).
            
            # We must ensure the state remains valid (uncovered_count == 0).
            
            delta_uncovered = 0
            # Check if removing b_in makes any tid uncovered
            for tid in block_t_ids[b_in]:
                if cover_counts[tid] == 1:
                    # This tid would become uncovered
                    # Unless b_out also covers it and adds it back?
                    # No, we remove then add.
                    # If b_out covers tid, then new_count = 1 - 1 + 1 = 1. OK.
                    # If b_out does not cover tid, new_count = 1 - 1 + 0 = 0. BAD.
                    if tid not in block_t_ids[b_out]:
                        delta_uncovered += 1
            
            # If delta_uncovered > 0, the swap is invalid.
            if delta_uncovered > 0:
                continue
            
            # If valid, accept the swap with some probability to explore.
            # Or accept if it seems to improve the "slack".
            
            # Let's just accept it. The next loop will try to remove redundant blocks.
            # If the swap was bad, we might not remove anything, but we changed the state.
            # To prevent cycling, we could track recent states, but let's keep it simple.
            
            # Accept Swap
            active_blocks.remove(b_in)
            for tid in block_t_ids[b_in]:
                cover_counts[tid] -= 1
                
            active_blocks.add(b_out)
            for tid in block_t_ids[b_out]:
                cover_counts[tid] += 1
                
            # Note: uncovered_count should still be 0 if delta_uncovered was 0.
            # We don't need to update uncovered_count explicitly if we know it's 0.
            
            improved = True # Force next loop to check for removals
            
    # 3. Final Repair
    # Ensure validity
    final_solution = set(best_solution)
    final_cover_counts = [0] * num_t_subsets
    for b_idx in final_solution:
        for tid in block_t_ids[b_idx]:
            final_cover_counts[tid] += 1
            
    uncovered_t_ids = [i for i in range(num_t_subsets) if final_cover_counts[i] == 0]
    
    while uncovered_t_ids:
        if time.time() - start_time > time_limit - 2.0:
            break
            
        target_tid = uncovered_t_ids[0]
        candidates = t_to_blocks[target_tid]
        best_block = -1
        max_new_cover = -1
        
        for b_idx in candidates:
            if b_idx in final_solution:
                continue
            new_cover = 0
            for tid in block_t_ids[b_idx]:
                if final_cover_counts[tid] == 0:
                    new_cover += 1
            if new_cover > max_new_cover:
                max_new_cover = new_cover
                best_block = b_idx
        
        if best_block != -1:
            final_solution.add(best_block)
            for tid in block_t_ids[best_block]:
                final_cover_counts[tid] += 1
                if final_cover_counts[tid] == 1 and tid in uncovered_t_ids:
                    uncovered_t_ids.remove(tid)
        else:
            break
            
    result = [list(all_k_subsets[i]) for i in final_solution]
    return result
