import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    limit = 115.0 
    deadline = start_time + limit

    # 1. Precomputation
    # Generate all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Generate all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map each block to the indices of the t-subsets it covers
    # block_covers[i] is a list of subset indices covered by block i
    block_covers = []
    # Map each t-subset to the indices of blocks that cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(blocks):
        covers = []
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            sub_to_blocks[idx].append(i)
        block_covers.append(np.array(covers, dtype=np.int32))
        
    if not _deadline_ok(deadline):
        return []

    # 2. Fast Greedy Initialization (Vectorized Gain Update)
    # We use a numpy array to track the marginal gain of each block.
    # gain[i] = number of currently uncovered subsets covered by block i.
    
    uncovered_mask = np.ones(n_t, dtype=bool)
    gains = np.full(n_blocks, 0, dtype=np.int32)
    
    # Initialize gains: all blocks start with C(k,t) potential gain
    # But we can just compute it once. Since all blocks have the same size,
    # they all cover the same number of t-subsets initially.
    # However, as we select blocks, gains decrease.
    
    # Actually, it's safer to just initialize gains to the length of block_covers[i]
    # which is constant C(k,t).
    initial_gain = len(block_covers[0]) if n_blocks > 0 else 0
    gains[:] = initial_gain
    
    selected = []
    selected_set = set()
    
    # Precompute for each t-subset, the list of blocks covering it (already done in sub_to_blocks)
    # To update gains efficiently:
    # When we pick block b:
    #   For each subset s in block_covers[b]:
    #     If s is currently uncovered:
    #       Mark s as covered
    #       For each block b' covering s:
    #         gains[b'] -= 1
    
    # This update is O(sum_{s in b} |sub_to_blocks[s]|).
    # |block_covers[b]| = C(k,t).
    # |sub_to_blocks[s]| is roughly constant.
    
    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break
            
        # Find the block with the maximum gain
        # If max gain is 0, we are stuck (should not happen for valid parameters)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Pick the block with max gain. If ties, pick random or first.
        # Using argmax is O(N).
        # To avoid bias towards lower indices, we can pick randomly among maxes, 
        # but argmax is fast enough. Let's stick to argmax for speed.
        best_block = np.argmax(gains)
        
        if best_block in selected_set:
            # This shouldn't happen if we set gains to 0 for selected blocks, 
            # but let's be safe.
            gains[best_block] = -1
            continue
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        # Update coverage and gains
        covers_b = block_covers[best_block]
        # Which of these are currently uncovered?
        # We can just iterate through them.
        for s_idx in covers_b:
            if uncovered_mask[s_idx]:
                uncovered_mask[s_idx] = False
                # Update gains for all blocks covering this subset
                for b_prime in sub_to_blocks[s_idx]:
                    gains[b_prime] -= 1
        
        # Set gain of selected block to -1 or 0 so it's not picked again
        gains[best_block] = -1

    if not selected:
        return []
        
    # 3. Local Search Optimization
    # We now have a valid cover. We want to minimize |selected|.
    # Strategy: 
    # 1. Remove redundant blocks (those where all covered subsets have count > 1).
    # 2. Try to swap blocks to reduce size or maintain size with better structure?
    #    The inspiration suggests an "Add-Remove" or "Swap" strategy.
    #    A common effective local search for covering designs is:
    #    - Pick a random block in the solution.
    #    - Try to remove it.
    #    - If removal causes uncovered subsets, try to add ONE other block that covers all those uncovered subsets.
    #    - If successful and size decreases (impossible with 1-for-1 swap unless the new block allows further removals? No, 1-for-1 keeps size same).
    #    - Wait, the goal is to MINIMIZE size.
    #    - Simple removal works if the block is redundant.
    #    - To reduce size, we need to find a block that is redundant.
    #    - If no block is redundant, we are in a local minimum for 1-removal.
    #    - To escape, we can try 2-for-1 swaps? Or just restart?
    #    - The "Add-Remove" strategy from Inspiration 1:
    #      Add a random block, then remove as many redundant blocks as possible.
    #      If the net result is a smaller set, keep it.
    #      This can escape local minima where no single block is removable.
    
    # Let's implement the "Add-Remove" local search.
    
    # Helper: Calculate counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        counts[block_covers[b]] += 1
        
    # Helper: Remove redundant blocks iteratively
    def prune_redundant(selected_list, counts, deadline):
        """
        Removes redundant blocks from selected_list and updates counts.
        Returns True if any block was removed.
        """
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            # Iterate backwards to safely remove elements
            for i in range(len(selected_list) - 1, -1, -1):
                b = selected_list[i]
                # Check if redundant: all subsets in b have count > 1
                # Note: counts include b's contribution.
                # So we check if counts[s] > 1 for all s in block_covers[b]
                # Actually, if counts[s] == 1, it means only b covers it (or b and others, but total is 1? No, count is total).
                # If count[s] == 1, then removing b makes it 0. So b is critical.
                # If count[s] > 1 for all s, b is redundant.
                
                is_redundant = True
                covers_b = block_covers[b]
                # Vectorized check
                if not np.all(counts[covers_b] > 1):
                    is_redundant = False
                    
                if is_redundant:
                    # Remove b
                    selected_list.pop(i)
                    counts[covers_b] -= 1
                    changed = True
        
        return changed

    # Initial prune
    selected_list = list(selected)
    prune_redundant(selected_list, counts, deadline)
    
    best_size = len(selected_list)
    best_solution = selected_list.copy()
    best_counts = counts.copy()
    
    # Local Search Loop: Add-Remove
    # We will run this until deadline.
    # Each iteration:
    # 1. Pick a random block NOT in solution.
    # 2. Add it to a COPY of current solution.
    # 3. Prune redundant blocks from the copy.
    # 4. If copy size < best_size, update best.
    # 5. If copy size <= current_size (or some threshold), accept as new current?
    #    Usually, we accept if it improves or is equal (to allow exploration).
    #    However, to avoid getting stuck in a large solution, we should reset to best if we drift too far.
    
    rng = random.Random(42)
    
    # To make "Add" fast, we might want a list of all block indices.
    all_block_indices = list(range(n_blocks))
    
    # We need to track which blocks are in the current solution for fast "is in" check?
    # Using a set is O(1).
    current_set = set(selected_list)
    
    # We also need to be able to reconstruct counts if we reset.
    # Reconstructing counts is O(N * C(k,t)). N ~ 300, C(k,t) ~ 126. 300*126 = 37800 ops. Fast.
    
    def rebuild_counts(block_list):
        c = np.zeros(n_t, dtype=np.int32)
        for b in block_list:
            c[block_covers[b]] += 1
        return c

    # Main Loop
    while _deadline_ok(deadline):
        # Pick a random block to add
        # We should probably pick a block that covers some "weakly covered" subsets?
        # Random is a good baseline.
        b_add = rng.choice(all_block_indices)
        
        if b_add in current_set:
            continue
            
        # Create a temporary copy of the solution
        # Copying list and set is O(N).
        temp_list = selected_list.copy()
        temp_set = current_set.copy()
        
        # Add the block
        temp_list.append(b_add)
        temp_set.add(b_add)
        
        # Update counts for temp
        # Instead of copying the whole counts array (O(n_t) = 11k), we can just update the relevant parts?
        # But we need to prune, which modifies counts.
        # Copying counts array: 11k integers. Very fast in numpy.
        temp_counts = counts.copy()
        temp_counts[block_covers[b_add]] += 1
        
        # Prune redundant blocks from temp
        prune_redundant(temp_list, temp_counts, deadline)
        
        current_size = len(temp_list)
        
        if current_size < best_size:
            best_size = current_size
            best_solution = temp_list.copy()
            best_counts = temp_counts.copy()
            
            # Update current to be the new best? 
            # Or just update current to temp?
            # If we update current to temp, we continue from this improved state.
            selected_list = temp_list
            current_set = temp_set
            counts = temp_counts
            
        elif current_size < len(selected_list):
            # Improvement over current, but not best. Accept to explore.
            selected_list = temp_list
            current_set = temp_set
            counts = temp_counts
            
        # Drift control:
        # If current solution is much worse than best, reset to best.
        # "Much worse" is subjective. Let's say if len(selected_list) > best_size + 5.
        if len(selected_list) > best_size + 5:
            selected_list = best_solution.copy()
            current_set = set(best_solution)
            counts = best_counts.copy()

    # Final Result
    # Ensure we return the best found
    # The best_solution is a list of block indices.
    result = [list(blocks[i]) for i in best_solution]
    return result
