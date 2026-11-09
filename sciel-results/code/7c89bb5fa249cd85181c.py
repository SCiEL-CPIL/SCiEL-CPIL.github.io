import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using optimized greedy set cover
    with randomized restarts, fast redundancy removal, and 1-exchange local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(block_masks)
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    # For each block, precompute which t-subsets it covers (as a list of indices)
    block_t_indices = []
    for i in range(num_blocks):
        mask = block_masks[i]
        indices = []
        for j in range(num_t_subsets):
            if mask & (1 << j):
                indices.append(j)
        block_t_indices.append(indices)
    
    def greedy_randomized(seed_offset=0):
        """Greedy set cover with randomized tie-breaking for diversity."""
        rng = random.Random(seed_offset)
        coverage_count = [0] * num_t_subsets
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_indices = []
        chosen_masks = []
        
        while current_uncovered != 0:
            if time.time() > deadline - 8.0:
                break
            
            best_gain = -1
            candidates = []
            
            for i in range(num_blocks):
                # Compute gain: number of currently uncovered t-subsets this block covers
                gain = 0
                for j in block_t_indices[i]:
                    if coverage_count[j] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain <= 0:
                break
            
            # Randomly pick from best candidates
            chosen_idx = rng.choice(candidates)
            chosen_indices.append(chosen_idx)
            chosen_masks.append(block_masks[chosen_idx])
            
            # Update coverage counts
            for j in block_t_indices[chosen_idx]:
                coverage_count[j] += 1
            
            current_uncovered &= ~block_masks[chosen_idx]
        
        return chosen_indices, chosen_masks
    
    def remove_redundant_fast(chosen_indices, chosen_masks):
        """Remove redundant blocks using efficient union computation."""
        if not chosen_indices:
            return chosen_indices, chosen_masks
        
        n = len(chosen_indices)
        # Build union of all masks
        total_union = 0
        for m in chosen_masks:
            total_union |= m
        
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen_indices)):
                if time.time() > deadline - 4.0:
                    break
                
                mask_i = chosen_masks[i]
                
                # Check if mask_i is subset of union of others
                # union_others = total_union without mask_i's unique contributions
                # A t-subset is covered by others if it's in total_union AND not only in mask_i
                # Simpler: check if mask_i & ~union_of_others == 0
                union_others = 0
                for j in range(len(chosen_indices)):
                    if i != j:
                        union_others |= chosen_masks[j]
                
                if (mask_i & ~union_others) == 0:
                    chosen_indices.pop(i)
                    chosen_masks.pop(i)
                    total_union = 0
                    for m in chosen_masks:
                        total_union |= m
                    changed = True
                    break
        
        return chosen_indices, chosen_masks
    
    def local_search_1exchange(chosen_indices, chosen_masks):
        """Try 1-exchange: remove one block, add one different block to maintain coverage."""
        n = len(chosen_indices)
        full_mask = (1 << num_t_subsets) - 1
        improved = True
        
        while improved and time.time() < deadline - 6.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 6.0:
                    break
                
                mask_i = chosen_masks[i]
                
                # Compute what's covered by all other blocks
                union_others = 0
                for j in range(n):
                    if i != j:
                        union_others |= chosen_masks[j]
                
                # t-subsets that are ONLY covered by block i
                only_in_i = mask_i & ~union_others
                if only_in_i == 0:
                    # Block i is redundant, remove it
                    chosen_indices.pop(i)
                    chosen_masks.pop(i)
                    n -= 1
                    improved = True
                    break
                
                # Try to find a block b that covers all of only_in_i
                # and doesn't add unnecessary new coverage
                for b in range(num_blocks):
                    if b == chosen_indices[i]:
                        continue
                    if (block_masks[b] & only_in_i) == only_in_i:
                        # Block b covers all the unique t-subsets of block i
                        # Replace i with b
                        chosen_indices[i] = b
                        chosen_masks[i] = block_masks[b]
                        n = len(chosen_indices)
                        improved = True
                        break
                
                if improved:
                    break
        
        return chosen_indices, chosen_masks
    
    def local_search_2for1(chosen_indices, chosen_masks):
        """Try to replace pairs of blocks with single blocks to reduce count."""
        n = len(chosen_indices)
        full_mask = (1 << num_t_subsets) - 1
        improved = True
        
        while improved and time.time() < deadline - 6.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 6.0:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 6.0:
                        break
                    
                    # Combined unique coverage of blocks i and j
                    union_others = 0
                    for idx in range(n):
                        if idx != i and idx != j:
                            union_others |= chosen_masks[idx]
                    
                    # What must be covered by the replacement
                    must_cover = (chosen_masks[i] | chosen_masks[j]) & ~union_others
                    
                    if must_cover == 0:
                        # Both blocks are redundant
                        # Remove j first (higher index), then i
                        chosen_indices.pop(j)
                        chosen_masks.pop(j)
                        n -= 1
                        chosen_indices.pop(i)
                        chosen_masks.pop(i)
                        n -= 1
                        improved = True
                        break
                    
                    # Find a single block that covers must_cover
                    for b in range(num_blocks):
                        if b == chosen_indices[i] or b == chosen_indices[j]:
                            continue
                        if (block_masks[b] & must_cover) == must_cover:
                            # Replace i and j with b
                            # Remove j, then i, then add b
                            chosen_indices.pop(j)
                            chosen_masks.pop(j)
                            chosen_indices.pop(i)
                            chosen_masks.pop(i)
                            chosen_indices.append(b)
                            chosen_masks.append(block_masks[b])
                            n = len(chosen_indices)
                            improved = True
                            break
                    
                    if improved:
                        break
                if improved:
                    break
        
        return chosen_indices, chosen_masks
    
    # Run multiple greedy variants with different random seeds and keep the best
    best_indices = None
    best_masks = None
    best_count = float('inf')
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 12.0:
            break
        
        seed_offset = restart * 1000 + 42
        indices, masks = greedy_randomized(seed_offset)
        indices, masks = remove_redundant_fast(indices, masks)
        
        if len(indices) < best_count:
            best_count = len(indices)
            best_indices = indices
            best_masks = masks
    
    # Apply local search to the best solution
    if best_indices is not None and time.time() < deadline - 8.0:
        # 1-exchange local search
        best_indices, best_masks = local_search_1exchange(best_indices, best_masks)
        best_count = len(best_indices)
        
        # 2-for-1 local search
        if time.time() < deadline - 6.0:
            best_indices, best_masks = local_search_2for1(best_indices, best_masks)
            best_count = len(best_indices)
        
        # Final redundancy removal
        best_indices, best_masks = remove_redundant_fast(best_indices, best_masks)
    
    if best_indices is None:
        # Fallback: simple greedy
        indices, masks = greedy_randomized(0)
        indices, masks = remove_redundant_fast(indices, masks)
        best_indices = indices
    
    return [list(all_blocks[idx]) for idx in best_indices]
