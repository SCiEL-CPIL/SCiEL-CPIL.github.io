import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precomputation
    # 1. All t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # 2. All k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # 3. For each block, list of t-subset indices it covers
    block_tsubs = [None] * n_blocks
    for i, b in enumerate(all_blocks):
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_tsubs[i] = subs
        
    # 4. For each t-subset, list of block indices covering it
    tsub_to_blocks = [[] for _ in range(n_t)]
    for i, b in enumerate(all_blocks):
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            tsub_to_blocks[idx].append(i)

    def time_ok():
        return time.time() < deadline

    def greedy_with_randomness(seed_val):
        random.seed(seed_val)
        # gain[i] = number of uncovered t-subsets in block i
        gain = np.full(n_blocks, k, dtype=np.int32) # C(k,t) is the initial count
        # Actually C(k,t) is 56 for k=8, t=5.
        initial_gain = len(all_t_subs[0]) # No, C(k,t)
        # Let's just use the length of block_tsubs[0]
        init_g = len(block_tsubs[0])
        gain = np.full(n_blocks, init_g, dtype=np.int32)
        
        is_covered = np.zeros(n_t, dtype=bool)
        selected_blocks = []
        
        while time_ok():
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            # To allow randomness, pick randomly from those with max gain
            candidates = np.where(gain == max_gain)[0]
            if len(candidates) == 0:
                break
                
            # Random selection among candidates for diversity
            best_block_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            selected_blocks.append(best_block_idx)
            
            # Update gains
            b_tsubs = block_tsubs[best_block_idx]
            for sub_idx in b_tsubs:
                if not is_covered[sub_idx]:
                    is_covered[sub_idx] = True
                    # Decrement gain for all blocks containing this sub_idx
                    for blk_idx in tsub_to_blocks[sub_idx]:
                        gain[blk_idx] -= 1
                        if gain[blk_idx] == 0:
                            # Optional: we can't easily remove from candidates, 
                            # but max will handle it
                            pass
        
        return selected_blocks

    def optimize_deletion(block_indices):
        """Remove redundant blocks from the list."""
        # Count coverage for each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bi in block_indices:
            for sub_idx in block_tsubs[bi]:
                cover_count[sub_idx] += 1
        
        blocks_to_keep = list(block_indices)
        changed = True
        while changed and time_ok():
            changed = False
            # Iterate in reverse to allow safe popping
            for i in range(len(blocks_to_keep) - 1, -1, -1):
                bi = blocks_to_keep[i]
                # Check if all t-subsets in this block have count > 1
                redundant = True
                for sub_idx in block_tsubs[bi]:
                    if cover_count[sub_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block bi
                    for sub_idx in block_tsubs[bi]:
                        cover_count[sub_idx] -= 1
                    blocks_to_keep.pop(i)
                    changed = True
        return blocks_to_keep

    def local_search_mutation(block_indices):
        """Try to improve by mutating points in blocks to enable further deletions."""
        # This is a heuristic: try swapping a point in a block, then run deletion.
        # If the result is smaller, keep it.
        best_blocks = list(block_indices)
        best_len = len(best_blocks)
        
        # We will try a few mutations
        # To be efficient, we only try mutations if we have time
        if not time_ok():
            return best_blocks
            
        # Build a set of blocks for fast lookup? 
        # Actually, let's just try random mutations on the current best list
        # and see if deletion reduces the size.
        
        # To avoid being too slow, we limit the number of mutation attempts
        attempts = 0
        max_attempts = 200 # Adjust based on time
        
        while time_ok() and attempts < max_attempts:
            attempts += 1
            # Pick a random block
            if not best_blocks:
                break
            bi_idx = random.randint(0, len(best_blocks) - 1)
            bi = best_blocks[bi_idx]
            block = all_blocks[bi]
            
            # Pick a random point in the block to remove
            pos = random.randint(0, k - 1)
            old_point = block[pos]
            
            # Pick a random new point not in the block
            candidates = [p for p in range(v) if p != old_point and p not in block]
            if not candidates:
                continue
            new_point = random.choice(candidates)
            
            # Create new block
            new_block_list = list(block)
            new_block_list[pos] = new_point
            new_block_list.sort()
            new_block = tuple(new_block_list)
            
            # Check if this new block already exists in our set
            # If it does, we might have duplicates, which is bad.
            # The problem statement says blocks must be distinct.
            # If we create a duplicate, we can just merge them (remove one).
            
            new_block_idx = block_to_idx.get(new_block, -1)
            
            # Construct the new list of blocks
            # Replace bi with new_block_idx
            # If new_block_idx is -1, it's a new block type, but we only have precomputed indices.
            # Wait, all_blocks contains ALL k-subsets. So new_block must be in all_blocks.
            # Yes, because it's a k-subset of v points.
            
            # Check for duplicates
            # If new_block_idx is already in best_blocks (other than at bi_idx), 
            # then adding it creates a duplicate. We should remove the duplicate.
            
            new_blocks_list = list(best_blocks)
            new_blocks_list[bi_idx] = new_block_idx
            
            # Check for duplicates
            seen = set()
            unique_blocks = []
            for idx in new_blocks_list:
                if idx not in seen:
                    seen.add(idx)
                    unique_blocks.append(idx)
                # else: duplicate, ignore
            
            # Run deletion optimization on this new candidate
            optimized = optimize_deletion(unique_blocks)
            
            if len(optimized) < best_len:
                best_blocks = optimized
                best_len = len(optimized)
                
        return best_blocks

    # Main Loop: Run multiple greedy passes with different seeds, then optimize
    best_solution = []
    best_len = float('inf')
    
    # First run: Deterministic-ish (seed 0)
    if time_ok():
        greedy_blocks = greedy_with_randomness(0)
        optimized_blocks = optimize_deletion(greedy_blocks)
        if len(optimized_blocks) < best_len:
            best_solution = optimized_blocks
            best_len = len(optimized_blocks)
            
    # Subsequent runs: Randomized greedy
    seed = 1
    while time_ok() and (time.time() - start_time) < 130.0: # Leave 15s for final cleanup/mutation
        greedy_blocks = greedy_with_randomness(seed)
        optimized_blocks = optimize_deletion(greedy_blocks)
        
        if len(optimized_blocks) < best_len:
            best_solution = optimized_blocks
            best_len = len(optimized_blocks)
            
        seed += 1
        
    # Local Search Mutation on the best solution found so far
    if time_ok() and best_solution:
        improved = local_search_mutation(best_solution)
        if len(improved) < best_len:
            best_solution = improved
            best_len = len(improved)
            
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return final_blocks
