import itertools
import random
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    block_masks = [0] * n_k_blocks
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Helper to get mask of a set of t-subset indices
    # We will represent the "gap" U as a bitmask of t-subset indices
    
    # 3. Greedy Set Cover for initial solution
    uncovered = (1 << n_t_subs) - 1
    selected_indices = []
    in_solution = [False] * n_k_blocks
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        in_solution[best_idx] = True
        uncovered &= ~block_masks[best_idx]
        
    # Initialize counts
    counts = [0] * n_t_subs
    for idx in selected_indices:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            m ^= lsb
            
    # Helper to update counts
    def add_block(idx):
        in_solution[idx] = True
        selected_indices.append(idx)
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            m ^= lsb
            
    def remove_block(idx):
        in_solution[idx] = False
        selected_indices.remove(idx)
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            m ^= lsb

    # Local Search
    # We will try to minimize len(selected_indices)
    
    # Function to find a block covering a specific mask (gap)
    def find_block_covering(gap_mask):
        # gap_mask is a bitmask of t-subset indices that need to be covered
        if gap_mask == 0:
            return -1 # No gap
        # Find a block that covers all bits in gap_mask
        # Optimization: Check blocks that cover the 'rarest' bit in gap_mask
        # Get bits in gap_mask
        bits_in_gap = []
        m = gap_mask
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            bits_in_gap.append(bit_idx)
            m ^= lsb
            
        # Sort bits by count (ascending) to pick the most constrained one first?
        # Actually, just pick the first one and filter candidates.
        # For efficiency, let's just iterate all blocks and check.
        # 75k blocks is small.
        for i in range(n_k_blocks):
            if in_solution[i]:
                continue
            if (block_masks[i] & gap_mask) == gap_mask:
                return i
        return -1

    # Main Local Search Loop
    # We allow a certain number of iterations or time limit
    # Given 120s limit, we can run many iterations.
    
    max_iterations = 10000
    iter_count = 0
    
    while iter_count < max_iterations:
        iter_count += 1
        improved = False
        
        # 1. Try to remove redundant blocks
        # A block is redundant if all its t-subsets have count > 1
        # We iterate through selected blocks
        removable = -1
        for idx in selected_indices:
            m = block_masks[idx]
            is_redundant = True
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    is_redundant = False
                    break
                m ^= lsb
            if is_redundant:
                removable = idx
                break
                
        if removable != -1:
            remove_block(removable)
            improved = True
            continue # Restart loop to catch new redundancies
        
        # 2. Try 1-for-1 Swap (Neutral move to escape local minima)
        # Pick a random block to remove, find a replacement that covers the gap.
        # If found, swap. This keeps size constant but changes configuration.
        
        # To avoid infinite loops, we can track visited states or limit swaps.
        # For simplicity, we just do a few random swaps.
        
        if len(selected_indices) > 0:
            # Pick a random block
            out_idx = random.choice(selected_indices)
            
            # Calculate gap if out_idx is removed
            gap_mask = 0
            m = block_masks[out_idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    gap_mask |= lsb
                m ^= lsb
                
            if gap_mask != 0:
                # Find a block that covers gap_mask
                in_idx = find_block_covering(gap_mask)
                if in_idx != -1:
                    # Perform swap
                    remove_block(out_idx)
                    add_block(in_idx)
                    improved = True
                    continue
        
        # 3. Try 2-for-1 Swap (Improvement move)
        # Remove 2 blocks, find 1 block to cover the combined gap.
        if len(selected_indices) >= 2:
            # Pick 2 random distinct blocks
            i1, i2 = random.sample(selected_indices, 2)
            idx1, idx2 = i1, i2
            
            # Calculate combined gap
            gap_mask = 0
            # Temporarily remove counts for idx1 and idx2 to find unique parts?
            # No, we need the union of their unique parts relative to the *rest* of the solution.
            # Let's just compute the gap by checking counts after hypothetical removal.
            
            # Hypothetical counts
            # We can just iterate the bits of mask[idx1] | mask[idx2] and check if count == 1 (for idx1) or count == 1 (for idx2) or (count == 2 and both cover it)?
            # Actually, a t-subset is uncovered if its count in the *remaining* set is 0.
            # Current count is C.
            # If we remove idx1 and idx2:
            # New count = C - (1 if idx1 covers it else 0) - (1 if idx2 covers it else 0).
            # If New count == 0, it's in the gap.
            
            m1 = block_masks[idx1]
            m2 = block_masks[idx2]
            union_m = m1 | m2
            
            while union_m:
                lsb = union_m & -union_m
                bit_idx = lsb.bit_length() - 1
                # Check if this bit is covered by idx1 or idx2
                covered_by_1 = (m1 & lsb) != 0
                covered_by_2 = (m2 & lsb) != 0
                new_count = counts[bit_idx]
                if covered_by_1:
                    new_count -= 1
                if covered_by_2:
                    new_count -= 1
                if new_count == 0:
                    gap_mask |= lsb
                union_m ^= lsb
                
            if gap_mask != 0:
                in_idx = find_block_covering(gap_mask)
                if in_idx != -1:
                    # Perform 2-for-1 swap
                    remove_block(idx1)
                    remove_block(idx2)
                    add_block(in_idx)
                    improved = True
                    continue
            else:
                # Gap is 0, meaning both blocks are redundant together?
                # If gap is 0, we can just remove one of them?
                # If removing both leaves no gap, removing one might also leave no gap?
                # Not necessarily.
                # But if gap is 0, it means all subsets covered by idx1 or idx2 are covered by others.
                # So idx1 is redundant?
                # If idx1 is redundant, step 1 would have caught it.
                # So this case (gap 0 for 2 blocks) implies both are redundant?
                # If step 1 didn't catch it, maybe my redundancy check is flawed?
                # My redundancy check: "is_redundant = True ... if counts[bit] == 1: is_redundant = False".
                # If I remove idx1, a bit is uncovered if counts[bit] == 1 and idx1 covers it.
                # If I remove idx1 AND idx2, a bit is uncovered if counts[bit] - (cov1) - (cov2) == 0.
                # If gap is 0, it means for all bits in m1|m2, counts[bit] >= cov1 + cov2.
                # This implies for bits in m1, counts[bit] >= cov1.
                # If counts[bit] == cov1, then removing idx1 makes count 0? No, count becomes counts[bit]-cov1.
                # If counts[bit] == 1 and cov1=1, new count 0.
                # If gap is 0, no such bits exist.
                # So idx1 is redundant?
                # Yes, if for all bits in m1, counts[bit] > cov1 (i.e. >= 2 if cov1=1), then idx1 is redundant.
                # So if gap is 0, both are redundant. Step 1 should have removed one.
                # So we can ignore this or just remove one.
                if idx1 in selected_indices:
                    remove_block(idx1)
                    improved = True
                    continue

        if not improved:
            # Stuck. Try random perturbation or break.
            # For now, break to avoid infinite loop.
            # In a real SA, we'd accept worse moves. Here we only accept neutral or better.
            # If we can't improve, we are at a local minimum.
            break

    return [list(k_blocks[i]) for i in selected_indices]
