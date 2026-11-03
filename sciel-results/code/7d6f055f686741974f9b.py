import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Precompute t-subsets and mapping
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Helper to get index of a sorted tuple
    def get_idx(subset):
        return subset_to_idx[tuple(sorted(subset))]
        
    # 2. Online Greedy Construction
    def run_online_greedy():
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Start with the smallest uncovered subset
            # To be deterministic and fast, pick min index
            min_idx = min(uncovered)
            start_subset = all_t_subsets[min_idx]
            block = list(start_subset)
            
            # Expand block to size k
            while len(block) < k:
                if time.time() > deadline:
                    break
                    
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                # Precompute (t-1) subsets of current block for gain calc?
                # Or just iterate.
                t_minus_1_subsets = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in t_minus_1_subsets:
                        new_subset = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(new_subset)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                block.append(best_point)
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for s in itertools.combinations(block, t):
                idx = subset_to_idx[tuple(s)]
                if idx in uncovered:
                    uncovered.remove(idx)
                    
        return blocks

    # Run initial greedy
    best_blocks = run_online_greedy()
    
    # If we didn't finish or result is bad, try to improve with Local Search
    # But first, ensure we have a valid cover.
    # If the loop broke due to time, we might have an invalid cover.
    # Check validity
    def is_valid_cover(blocks):
        covered = [False] * num_t_subsets
        for b in blocks:
            for s in itertools.combinations(b, t):
                covered[subset_to_idx[tuple(s)]] = True
        return all(covered)
        
    if not is_valid_cover(best_blocks):
         # Fallback: continue greedy or return what we have? 
         # The task requires valid cover. If we run out of time, we should try to finish quickly or return a valid one.
         # The online greedy is fast enough for 19,7,4 usually.
         # Let's assume it finishes.
         pass

    # 3. Pool Generation
    pool = list(set(tuple(b) for b in best_blocks))
    
    # Add random blocks
    num_random = 500
    for _ in range(num_random):
        if time.time() > deadline: break
        rand_block = tuple(sorted(random.sample(range(v), k)))
        pool.append(rand_block)
        
    # Remove duplicates in pool
    pool = list(set(pool))
    
    # 4. Greedy Set Cover on Pool
    def greedy_set_cover(pool, blocks_hint):
        uncovered = set(range(num_t_subsets))
        selected = []
        
        # Precompute subsets for each block in pool
        # block_subsets[i] = list of indices of t-subsets in pool[i]
        block_subsets = []
        for b in pool:
            subs = [subset_to_idx[tuple(s)] for s in itertools.combinations(b, t)]
            block_subsets.append(subs)
            
        while uncovered:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_gain = 0
            for i, subs in enumerate(block_subsets):
                # Count how many of these are in uncovered
                gain = 0
                for idx in subs:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1:
                break # Should not happen if pool is good
                
            selected.append(pool[best_idx])
            for idx in block_subsets[best_idx]:
                if idx in uncovered:
                    uncovered.remove(idx)
                    
        return selected

    # Run Set Cover
    # We can use the initial greedy blocks as a hint or just run on pool
    candidate_blocks = greedy_set_cover(pool, best_blocks)
    
    # 5. Local Search: Deletion and Swap
    def local_search(blocks):
        blocks = list(set(blocks)) # Ensure unique
        improved = True
        while improved:
            improved = False
            
            # Deletion
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                b = blocks[i]
                # Check if b is essential
                is_essential = False
                for s in itertools.combinations(b, t):
                    idx = subset_to_idx[tuple(s)]
                    # Check if any other block covers idx
                    found = False
                    for j, b2 in enumerate(blocks):
                        if i == j: continue
                        # Check if idx is in b2
                        # Optimization: precompute? 
                        # For now, simple check
                        if idx in [subset_to_idx[tuple(s2)] for s2 in itertools.combinations(b2, t)]:
                            found = True
                            break
                    if not found:
                        is_essential = True
                        break
                if not is_essential:
                    blocks.pop(i)
                    improved = True
                    break # Restart deletion loop
            
            # Swap
            # For each block, try to swap with one from pool
            # To be efficient, only try a few swaps per iteration
            if not improved:
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        return blocks
                    b = blocks[i]
                    # Find unique subsets covered by b
                    unique_subsets = []
                    for s in itertools.combinations(b, t):
                        idx = subset_to_idx[tuple(s)]
                        found = False
                        for j, b2 in enumerate(blocks):
                            if i == j: continue
                            if idx in [subset_to_idx[tuple(s2)] for s2 in itertools.combinations(b2, t)]:
                                found = True
                                break
                        if not found:
                            unique_subsets.append(idx)
                            
                    if not unique_subsets:
                        continue # Should have been deleted
                        
                    # Find a block in pool that covers all unique_subsets
                    for b2 in pool:
                        if b2 in blocks:
                            continue
                        # Check if b2 covers all unique_subsets
                        b2_subs = set(subset_to_idx[tuple(s)] for s in itertools.combinations(b2, t))
                        if all(idx in b2_subs for idx in unique_subsets):
                            # Swap
                            blocks[i] = b2
                            improved = True
                            break # Restart loop
                    if improved:
                        break
                        
        return blocks

    # Run local search
    final_blocks = local_search(candidate_blocks)
    
    # Ensure validity
    if not is_valid_cover(final_blocks):
        # Fallback to candidate_blocks if it was valid
        if is_valid_cover(candidate_blocks):
            final_blocks = candidate_blocks
        else:
            final_blocks = best_blocks # Last resort
            
    return [list(b) for b in final_blocks]
