import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Pre-compute all t-subsets and map them to indices for fast counting
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Helper to get the list of t-subset indices for a given block
    def get_block_t_indices(block):
        return [t_subset_to_idx[s] for s in itertools.combinations(block, t)]

    # 1. Greedy Initialization
    # uncovered: set of indices of t-subsets not yet covered
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    # Weight function: prefer t-subsets that are contained in fewer candidate blocks?
    # For simplicity and speed, standard greedy (pick first uncovered, extend best) is fast.
    # But "Weighted" greedy can be better. 
    # Let's stick to a fast greedy first, then optimize.
    
    while uncovered:
        # Pick the "most constrained" uncovered t-subset?
        # For v=19, t=4, all t-subsets are symmetric initially.
        # We just pick the smallest index for determinism/speed.
        first_uncovered = min(uncovered)
        # Find the t-subset
        target_t = all_t_subsets[first_uncovered]
        
        # Initialize block with this t-subset
        block = list(target_t)
        
        # Extend block to size k
        while len(block) < k:
            best_point = -1
            best_gain = -1
            # For each candidate point, count how many new t-subsets it covers
            # A new t-subset is formed by (t-1) elements from current block + new point
            # We need to check if this new t-subset is in 'uncovered'
            
            # Optimization: Pre-compute (t-1) subsets of current block
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for sub in t_minus_1_subsets:
                    # Create the t-subset
                    new_t = tuple(sorted(sub + (x,)))
                    idx = t_subset_to_idx[new_t]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Should not happen if v is large enough
                break
            block.append(best_point)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for idx in get_block_t_indices(block):
            uncovered.discard(idx)
            
        if time.time() > deadline - 5:
            break

    # Ensure we have a valid covering if we broke early
    # If we broke early, uncovered is not empty. We need to finish it quickly.
    while uncovered and time.time() < deadline - 1:
        first_uncovered = min(uncovered)
        target_t = all_t_subsets[first_uncovered]
        block = list(target_t)
        # Randomly fill the rest? Or greedy? Greedy is safer.
        while len(block) < k:
            best_point = -1
            best_gain = -1
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for sub in t_minus_1_subsets:
                    new_t = tuple(sorted(sub + (x,)))
                    idx = t_subset_to_idx[new_t]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1: break
            block.append(best_point)
        block.sort()
        blocks.append(block)
        for idx in get_block_t_indices(block):
            uncovered.discard(idx)

    # 2. Reduction Phase
    # Maintain cover counts
    cover_count = [0] * num_t_subsets
    
    def add_block(block):
        for idx in get_block_t_indices(block):
            cover_count[idx] += 1
            
    def remove_block(block):
        for idx in get_block_t_indices(block):
            cover_count[idx] -= 1
            
    def is_valid_covering():
        return all(c > 0 for c in cover_count)

    # Initialize counts
    for b in blocks:
        add_block(b)
        
    # Iterative removal
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try to remove each block
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            # Tentatively remove
            remove_block(b)
            if is_valid_covering():
                blocks.pop(i)
                improved = True
            else:
                # Restore
                add_block(b)

    # 3. Local Search Phase
    # Strategy: Randomly pick a block, replace it with a random different block, 
    # then try to reduce. If size decreases, keep.
    
    # Pre-generate a pool of random blocks? Or generate on fly.
    # Generating on fly is fine.
    
    current_size = len(blocks)
    
    # We need a fast way to check validity. 
    # is_valid_covering() iterates 3876 elements. That's fast.
    
    for _ in range(10000):
        if time.time() > deadline - 5:
            break
        if len(blocks) == 0: break
        
        # Pick a random block to swap
        idx_to_swap = random.randint(0, len(blocks) - 1)
        old_block = blocks[idx_to_swap]
        
        # Generate a random new block
        new_block = sorted(random.sample(range(v), k))
        
        if new_block == old_block:
            continue
            
        # Swap
        remove_block(old_block)
        add_block(new_block)
        blocks[idx_to_swap] = new_block
        
        # Try to reduce
        local_improved = False
        # We can do a quick reduction pass on the whole set, or just check if we can remove the new one?
        # Better: Run the full reduction loop but limited iterations?
        # The full reduction loop is O(N * 35) which is fast.
        
        # Let's just run the reduction loop again.
        # To avoid infinite loops or long runs, we limit the number of removals in this pass.
        # Actually, the reduction loop above is already efficient.
        
        # Let's just call a helper that tries to remove redundant blocks.
        # We'll modify the reduction to be a function.
        
        # For speed, we just do a single pass of reduction here.
        for i in range(len(blocks) - 1, -1, -1):
             b = blocks[i]
             remove_block(b)
             if is_valid_covering():
                 blocks.pop(i)
                 local_improved = True
             else:
                 add_block(b)
                 
        if len(blocks) < current_size:
            current_size = len(blocks)
            # If we improved, we can continue searching from this state
        else:
            # If no improvement, revert the swap?
            # If we didn't remove the new block, it's still there.
            # If we removed it, it's gone.
            # If len didn't decrease, we should revert to previous state to avoid drifting?
            # Or keep it? 
            # Usually, if a swap doesn't lead to immediate reduction, it might be part of a larger sequence.
            # But for simplicity and stability, if no reduction, revert.
            
            # Revert
            remove_block(new_block)
            add_block(old_block)
            blocks[idx_to_swap] = old_block

    return blocks
