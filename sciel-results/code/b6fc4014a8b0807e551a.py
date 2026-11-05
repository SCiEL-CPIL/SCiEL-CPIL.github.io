import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    triple_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Function to get the index of a triple (tuple of t sorted ints)
    # We assume inputs are sorted tuples
    def get_triple_idx(triple):
        return triple_to_idx[triple]

    # Function to get the t-subsets of a block
    # block is a sorted tuple/list of k ints
    def get_block_triples(block):
        return [get_triple_idx(tuple(sub)) for sub in itertools.combinations(block, t)]

    # Greedy construction
    def greedy_construct(seed=0):
        random.seed(seed)
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        blocks = []
        
        # Precompute list of points
        points = list(range(v))
        
        while np.any(counts == 0):
            # Find the triple with minimum count (0)
            # If multiple, pick one randomly to add diversity
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            if len(candidates) > 0:
                worst_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                break
            
            worst_triple = all_t_subsets[worst_idx]
            block = list(worst_triple)
            
            # Greedily add points to block
            available = [p for p in points if p not in block]
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                current_block_set = set(block)
                
                # Calculate current triples covered by partial block?
                # No, we need to see what adding a point would cover.
                # A point x added to block B covers triples (B \ {some}) + x?
                # Actually, the triples covered by the final block are combinations of the final k points.
                # When adding x to current block C (size c < k), the new triples formed are those containing x and t-1 points from C.
                
                for x in available:
                    if x in current_block_set:
                        continue
                    # Triples formed: choose t-1 from C, add x
                    # Sort is needed for lookup
                    gain = 0
                    # Optimization: only count if the triple is currently 0? 
                    # Or count weighted by 1/(counts+1) to prefer filling 0s?
                    # Let's stick to max count of 0s covered, then max total.
                    
                    # To speed up, we can precompute the t-1 subsets of C
                    # But C changes.
                    
                    # For v=24, k=7, t=3, inner loop is small.
                    for rest in itertools.combinations(block, t - 1):
                        # rest is tuple of t-1 points
                        # new_triple = sorted(rest + (x,))
                        # Since rest is sorted and x is added, we need to sort
                        # We can do this efficiently
                        new_triple_list = list(rest)
                        new_triple_list.append(x)
                        new_triple_list.sort()
                        idx = triple_to_idx[tuple(new_triple_list)]
                        if counts[idx] == 0:
                            gain += 1
                        elif counts[idx] == 1:
                            gain += 0.1 # Slight preference for covering 1s over 2s?
                        # If we just want to cover 0s, gain += (counts[idx] == 0)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    break
                block.append(best_point)
                # Remove best_point from available? 
                # available is list, remove is O(n). n=24. Fine.
                available.remove(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update counts
            for tri_idx in get_block_triples(block):
                counts[tri_idx] += 1
                
        return blocks, counts

    # Local Search: Try to reduce block count
    def local_search(blocks, counts, time_limit):
        blocks = [list(b) for b in blocks]
        counts = counts.copy()
        
        # Helper to check if a block is redundant
        def is_redundant(block_idx):
            block = blocks[block_idx]
            for tri_idx in get_block_triples(block):
                if counts[tri_idx] == 1:
                    return False
            return True

        # Helper to apply a block change
        def apply_block_change(old_block, new_block, block_idx):
            # Remove old block contributions
            for tri_idx in get_block_triples(old_block):
                counts[tri_idx] -= 1
            # Check validity
            if np.any(counts < 0):
                # Rollback
                for tri_idx in get_block_triples(old_block):
                    counts[tri_idx] += 1
                return False
            # If any triple is 0, it's invalid unless we are deleting the block and it was the only cover?
            # No, covering design requires every triple covered by AT LEAST one block.
            # So if counts[tri] becomes 0, it's invalid.
            if np.any(counts == 0):
                 # Wait, if we are SWAPPING, the new block might cover it.
                 # We must add new block before checking?
                 # Let's check after adding.
                 pass
            
            blocks[block_idx] = new_block
            for tri_idx in get_block_triples(new_block):
                counts[tri_idx] += 1
                
            if np.any(counts < 0) or np.any(counts == 0):
                # Rollback
                for tri_idx in get_block_triples(new_block):
                    counts[tri_idx] -= 1
                blocks[block_idx] = old_block
                for tri_idx in get_block_triples(old_block):
                    counts[tri_idx] += 1
                return False
            return True

        best_blocks = blocks
        best_counts = counts
        best_len = len(blocks)
        
        while time.time() < time_limit:
            improved = False
            # 1. Try to delete blocks
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(i):
                    # Remove block i
                    old_block = blocks[i]
                    for tri_idx in get_block_triples(old_block):
                        counts[tri_idx] -= 1
                    blocks.pop(i)
                    improved = True
                    # Reset loop to check again
                    break
            
            if improved:
                continue

            # 2. Try to swap points to make blocks redundant
            # For each block, try swapping a point in it with a point outside
            random.shuffle(range(len(blocks))) # Randomize order
            for i in range(len(blocks)):
                if time.time() >= time_limit: break
                block = blocks[i]
                block_set = set(block)
                
                # Get points in block
                # Get points not in block
                outside = [p for p in range(v) if p not in block_set]
                
                # Try swapping each point in block with each point outside
                # To limit search, maybe only try a few swaps per block per iteration
                swaps_tried = 0
                for x in block:
                    for y in outside:
                        if swaps_tried > 5: break # Limit work
                        new_block = [p if p != x else y for p in block]
                        new_block.sort()
                        
                        # Check if new block is same as old
                        if new_block == block:
                            continue
                        
                        # Try the swap
                        # We need to check if this swap makes the block redundant or helps another block become redundant.
                        # Simple heuristic: Accept if the swap is valid (no counts drop to 0) 
                        # AND (the block becomes redundant OR some other block becomes redundant).
                        
                        # To check if it helps, we can simulate the swap and check redundancy.
                        # This is expensive.
                        
                        # Alternative: Accept if valid and random() < 0.1 (Simulated Annealing style?)
                        # Or Accept if valid and it increases the number of triples with count > 1?
                        
                        # Let's try: Accept if valid and (is_redundant_after_swap(i) or any_other_block_redundant())
                        # This is too slow to check "any other".
                        
                        # Let's try: Accept if valid and the swap reduces the "uniqueness" of the block.
                        # Uniqueness = number of triples in block with count == 1.
                        
                        old_uniq = sum(1 for tri_idx in get_block_triples(block) if counts[tri_idx] == 1)
                        
                        # Simulate swap
                        temp_counts = counts.copy()
                        for tri_idx in get_block_triples(block):
                            temp_counts[tri_idx] -= 1
                        for tri_idx in get_block_triples(new_block):
                            temp_counts[tri_idx] += 1
                        
                        if np.any(temp_counts == 0):
                            continue
                            
                        new_uniq = sum(1 for tri_idx in get_block_triples(new_block) if temp_counts[tri_idx] == 1)
                        
                        # If new block is less unique, or if it allows deletion?
                        # If we just reduce uniqueness, we might be able to delete it later.
                        if new_uniq < old_uniq:
                            # Apply swap
                            for tri_idx in get_block_triples(block):
                                counts[tri_idx] -= 1
                            blocks[i] = new_block
                            for tri_idx in get_block_triples(new_block):
                                counts[tri_idx] += 1
                            improved = True
                            swaps_tried += 1
                            break
                    if swaps_tried > 5:
                        break
                
                if improved:
                    break # Restart deletion phase
            
            if len(blocks) < best_len:
                best_blocks = [b[:] for b in blocks]
                best_counts = counts.copy()
                best_len = len(blocks)
                
            if not improved:
                # Stuck? Try a random restart of the local search phase?
                # Or just break if no improvement in many iterations?
                # For now, break to try a new greedy seed
                break
                
        return best_blocks, best_counts

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy seeds
    seed = 0
    while time.time() < deadline:
        blocks, counts = greedy_construct(seed)
        len_b = len(blocks)
        
        if len_b < best_len:
            # Try local search on this promising solution
            time_left = deadline - time.time()
            # Give local search some time, e.g., 50% of remaining or fixed 10s
            ls_time = min(10.0, time_left * 0.5)
            if ls_time > 1.0:
                ls_blocks, ls_counts = local_search(blocks, counts, time.time() + ls_time)
                len_ls = len(ls_blocks)
                if len_ls < best_len:
                    best_len = len_ls
                    best_solution = ls_blocks
            else:
                if len_b < best_len:
                    best_len = len_b
                    best_solution = blocks
            
        seed += 1
        
    if best_solution is None:
        # Fallback: return the last greedy result
        blocks, counts = greedy_construct(0)
        return [list(b) for b in blocks]
        
    return [list(b) for b in best_solution]
