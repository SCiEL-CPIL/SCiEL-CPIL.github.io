import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy set cover
    with an inverted index for fast candidate selection, followed by local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    # and build an inverted index: t_subset_idx -> list of block indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for bi, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            t_subset_to_blocks[idx].append(bi)
        block_masks[bi] = mask
    
    full_mask = (1 << num_t_subsets) - 1
    
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
        
    def greedy_set_cover(randomize=True, rng=None):
        """
        Run greedy set cover using inverted index for candidate selection.
        """
        if rng is None:
            rng = random.Random()
            
        current_uncovered = full_mask
        chosen_indices = []
        chosen_masks = []
        
        while current_uncovered != 0:
            if time.time() > deadline:
                break
            
            # Select a target t-subset to cover
            # We pick a random uncovered t-subset to break symmetries
            # and ensure we explore different parts of the search space
            n_uncovered = count_bits(current_uncovered)
            if n_uncovered == 0:
                break
            
            # Pick a random index among the set bits
            # This is O(n_uncovered) worst case, but n_uncovered decreases.
            # A faster way: generate random r in [0, n_uncovered), then find r-th set bit.
            r = rng.randrange(n_uncovered)
            temp = current_uncovered
            # Clear r bits to find the r-th set bit
            for _ in range(r):
                temp &= temp - 1
            target_bit = temp & (-temp)
            target_idx = target_bit.bit_length() - 1
            
            # Get candidates: blocks that cover target_idx
            candidates = t_subset_to_blocks[target_idx]
            
            best_gain = 0
            best_idx = -1
            best_mask = 0
            
            for bi in candidates:
                # Check if this block is already chosen (optional, but gain will be 0 anyway)
                # Actually, if it's chosen, its bits are already cleared from current_uncovered, 
                # so gain will be 0. No need to check membership.
                gain = count_bits(current_uncovered & block_masks[bi])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = bi
                    best_mask = block_masks[bi]
                
                # Early termination if we found a perfect block for this step?
                # Not really, but if gain == n_uncovered, we covered everything.
                if gain == n_uncovered:
                    break
            
            if best_gain == 0:
                # Should not happen if covering exists
                break
                
            chosen_indices.append(best_idx)
            chosen_masks.append(best_mask)
            current_uncovered &= ~best_mask
            
        return chosen_indices, chosen_masks

    def remove_redundant(chosen_indices, chosen_masks):
        """Remove blocks whose coverage is fully covered by other chosen blocks."""
        n = len(chosen_indices)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if time.time() > deadline:
                    return chosen_indices, chosen_masks
                
                mask_i = chosen_masks[i]
                
                # Calculate union of all other masks
                union_others = 0
                for j in range(n):
                    if i != j:
                        union_others |= chosen_masks[j]
                
                # Check if mask_i is subset of union_others
                if (mask_i & ~union_others) == 0:
                    chosen_indices.pop(i)
                    chosen_masks.pop(i)
                    n -= 1
                    changed = True
                    break
        
        return chosen_indices, chosen_masks

    def local_search_1_exchange(chosen_indices, chosen_masks):
        """
        Try to reduce block count by replacing one block with another 
        that covers the 'lost' t-subsets.
        """
        n = len(chosen_indices)
        if n == 0:
            return chosen_indices, chosen_masks
        
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            mask_i = chosen_masks[i]
            
            # Union of all other blocks
            union_others = 0
            for j in range(n):
                if j != i:
                    union_others |= chosen_masks[j]
            
            # t-subsets only covered by block i
            lost = mask_i & ~union_others
            if lost == 0:
                # Redundant
                chosen_indices.pop(i)
                chosen_masks.pop(i)
                n -= 1
                improved = True
                # Restart loop to check for more redundancies
                return local_search_1_exchange(chosen_indices, chosen_masks)
            
            # Find a single block that covers all of 'lost'
            # We can use the inverted index to find candidates covering the first lost t-subset
            # But 'lost' might have multiple bits. 
            # Strategy: Pick one bit from 'lost', find blocks covering it, check if they cover ALL 'lost'.
            
            # Get the first set bit in 'lost'
            first_lost_bit = lost & (-lost)
            first_lost_idx = first_lost_bit.bit_length() - 1
            
            candidates = t_subset_to_blocks[first_lost_idx]
            
            best_replacement = -1
            best_gain = -1 # We want to maximize new coverage (minimize redundancy), but any valid replacement is OK for count reduction.
            
            for bi in candidates:
                if bi == chosen_indices[i]:
                    continue
                # Check if bi covers all 'lost'
                if (lost & ~block_masks[bi]) == 0:
                    # Valid replacement
                    # Prefer one that adds more coverage to the rest of the design?
                    # Or just take the first one?
                    # Taking the first one is fast.
                    best_replacement = bi
                    break # Just take the first valid one to save time
            
            if best_replacement != -1:
                # Replace block i with best_replacement
                chosen_indices[i] = best_replacement
                chosen_masks[i] = block_masks[best_replacement]
                improved = True
                # Continue searching for more improvements
                # Note: replacing might create new redundancies, so we don't restart immediately, 
                # but we continue the loop. The redundancy check at the start of the next iteration
                # or the final remove_redundant call will handle it.
                
        return chosen_indices, chosen_masks

    # Main search
    best_result = None
    best_count = float('inf')
    
    # 1. Deterministic Greedy (fast baseline)
    det_indices, det_masks = greedy_set_cover(randomize=False)
    det_indices, det_masks = remove_redundant(det_indices, det_masks)
    
    # Local search on deterministic result
    improved = True
    while improved:
        if time.time() > deadline:
            break
        old_count = len(det_indices)
        det_indices, det_masks = local_search_1_exchange(det_indices, det_masks)
        det_indices, det_masks = remove_redundant(det_indices, det_masks)
        if len(det_indices) < old_count:
            improved = True
        else:
            improved = False
            
    if len(det_indices) < best_count:
        best_count = len(det_indices)
        best_result = (list(det_indices), list(det_masks))
        
    # 2. Randomized Greedy with Multiple Restarts
    # The inverted index makes each step ~100x faster, allowing many more restarts.
    rng = random.Random(42)
    
    # Estimate time per restart
    # Let's just run as many as we can until deadline
    restart_count = 0
    while time.time() < deadline - 5.0: # Leave 5s for final polish
        restart_count += 1
        # Vary random seed for diversity
        temp_rng = random.Random(42 + restart_count * 13)
        
        rand_indices, rand_masks = greedy_set_cover(randomize=True, rng=temp_rng)
        
        if len(rand_indices) == 0:
            continue
        
        # Remove redundant
        rand_indices, rand_masks = remove_redundant(rand_indices, rand_masks)
        
        # Local search
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            old_count = len(rand_indices)
            rand_indices, rand_masks = local_search_1_exchange(rand_indices, rand_masks)
            rand_indices, rand_masks = remove_redundant(rand_indices, rand_masks)
            if len(rand_indices) < old_count:
                improved = True
            else:
                improved = False
        
        if len(rand_indices) < best_count:
            best_count = len(rand_indices)
            best_result = (list(rand_indices), list(rand_masks))
            
    # 3. Final Polish on Best Result
    if best_result is not None:
        best_idx, best_msk = best_result
        
        # Try more aggressive local search
        for _ in range(50):
            if time.time() > deadline:
                break
            old_count = len(best_idx)
            best_idx, best_msk = local_search_1_exchange(best_idx, best_msk)
            best_idx, best_msk = remove_redundant(best_idx, best_msk)
            if len(best_idx) < old_count:
                best_count = len(best_idx)
            else:
                # If no improvement, break to save time for other things? 
                # Or continue if we have time?
                # Local search is cheap now. Let's keep trying a few more times.
                pass
        
        best_result = (best_idx, best_msk)

    if best_result is None:
        # Fallback: return empty or minimal
        return []
    
    return [list(all_blocks[idx]) for idx in best_result[0]]
