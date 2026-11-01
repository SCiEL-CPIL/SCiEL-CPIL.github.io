import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer

    # 1. Precomputation
    # All t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # All k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    num_k_subs = len(all_k_subs)
    
    # For each k-subset, store the indices of the t-subsets it contains
    # Also create a bitset mask for fast intersection with uncovered set
    k_sub_t_indices = []
    k_sub_masks = [0] * num_k_subs
    
    # Map t-subset index to bitmask bit position
    # We use Python integers as bitsets
    for i, k_sub in enumerate(all_k_subs):
        indices = []
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_index[t_sub]
            indices.append(idx)
            mask |= (1 << idx)
        k_sub_t_indices.append(indices)
        k_sub_masks[i] = mask

    # Helper to get gain (number of uncovered t-subsets covered by block i)
    # uncovered_mask is a Python int bitset where bit j is 1 if t_sub j is uncovered
    
    def get_gain(block_idx, uncovered_mask):
        # Intersection of block's mask and uncovered mask
        # Count bits
        return (k_sub_masks[block_idx] & uncovered_mask).bit_count()

    # Initial Greedy to get a valid cover
    # This serves as the starting point and baseline
    uncovered_mask = (1 << num_t_subs) - 1
    remaining = num_t_subs
    current_blocks = []
    used_blocks = set()
    
    # Standard Greedy Initialization
    while remaining > 0:
        if time.time() > deadline - 30.0:
            break
        best_gain = -1
        best_idx = -1
        for i in range(num_k_subs):
            if i in used_blocks:
                continue
            gain = get_gain(i, uncovered_mask)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early break if we cover all remaining?
                if best_gain == remaining:
                    break
        
        if best_gain <= 0:
            break # Should not happen if valid cover exists
            
        current_blocks.append(best_idx)
        used_blocks.add(best_idx)
        uncovered_mask &= ~k_sub_masks[best_idx]
        remaining -= best_gain
        
    best_blocks = current_blocks[:]
    best_len = len(best_blocks)
    
    # Check validity of initial greedy
    def is_valid(blocks):
        # Reconstruct mask
        mask = 0
        for b in blocks:
            mask |= k_sub_masks[b]
        return mask == ((1 << num_t_subs) - 1)
        
    if not is_valid(best_blocks):
        # If initial greedy failed due to time, we might have an invalid cover.
        # But for v=19, it's fast. If it fails, we return empty or best effort.
        # The task says "Valid when every t-subset...".
        # If we can't make a valid one, score is 0.
        # But greedy on 19 points is extremely fast.
        pass

    # Lagrangian / Weighted Greedy Refinement
    # We will run multiple iterations of a weighted greedy construction.
    # Weights lambda_s start at 1.0.
    # In each iteration, we build a cover using weights to bias selection.
    # Then we update weights based on coverage redundancy.
    
    # Initialize weights
    weights = [1.0] * num_t_subs
    
    # Precompute weighted gain? No, weights change.
    # We need to calculate gain dynamically: sum(weights[s] for s in B if s uncovered)
    # This is O(C(k,t)) per block per candidate.
    # C(7,4) = 35. 50k blocks * 35 = 1.75M ops per scan.
    # 1.75M * 100 iterations = 175M ops. Might be too slow in Python.
    
    # Optimization: 
    # Instead of full re-evaluation, we can use the bitset for "uncovered" check,
    # but we need weighted sum.
    # We can approximate by only considering blocks that cover at least one uncovered set.
    
    # Alternative: Use the weights to *prune* or *rank* candidates faster?
    # Let's try a simpler structural change: 
    # "Greedy with Randomized Ties" + "Redundancy Removal"
    # This is often more effective than complex weight updates if the weight update is O(N^2).
    
    # Let's stick to the Weighted Greedy but optimize the inner loop.
    # We will limit the number of weighted greedy restarts.
    
    num_restarts = 10 # Fixed number of structural restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Reset weights with some randomness to escape local minima
        # Or decay previous weights?
        # Let's use a fresh set of weights biased by "difficulty"
        # Difficulty: how many blocks cover this t-subset in the *best known* solution?
        # We can compute this from best_blocks.
        
        # Compute coverage counts in best_blocks
        cover_counts = [0] * num_t_subs
        for b in best_blocks:
            for s_idx in k_sub_t_indices[b]:
                cover_counts[s_idx] += 1
        
        # Initialize weights: Higher weight for sets with low coverage (hard to cover without redundancy)
        # Or lower weight for sets with high coverage (easy to cover, so we can afford to ignore them slightly?)
        # Actually, if a set is covered 10 times, it's "wasteful" to prioritize it.
        # If a set is covered 1 time, it's "critical".
        # So weight[s] = 1 / cover_counts[s] ?
        # If cover_counts[s] == 0, weight = infinity?
        
        new_weights = [0.0] * num_t_subs
        for s in range(num_t_subs):
            if cover_counts[s] == 0:
                new_weights[s] = 100.0 # Critical
            else:
                new_weights[s] = 1.0 / cover_counts[s]
        
        # Run Weighted Greedy
        # Objective: Maximize sum(new_weights[s] for s in B if s uncovered)
        # To speed this up, we can precompute the weighted mask? No, weights are floats.
        
        # We will do a standard greedy scan, but the "gain" is weighted.
        # To avoid O(1.75M) per step, we can limit candidates or use a faster heuristic.
        # Given v=19, 150 steps * 50k candidates * 35 t-sets = 262M ops.
        # In Python, 262M ops is ~20-30 seconds. Too slow for 10 restarts.
        
        # Strategy: Use the bitset to quickly filter candidates.
        # Only consider blocks that cover at least one uncovered set.
        # And among those, we need weighted sum.
        
        # Let's try a different approach: 
        # Use the weights to *perturb* the standard gain.
        # gain = unweighted_gain + epsilon * weighted_gain
        # This is still expensive.
        
        # Let's abandon the full weighted greedy if it's too slow.
        # Instead, use **Local Search (Swap/Remove/Add)** but guided by the weights?
        # Or just **Multi-Start Greedy with Random Noise**.
        
        # Let's implement **Greedy with Random Tie-Breaking**.
        # This is fast and explores different solutions.
        
        uncovered_mask = (1 << num_t_subs) - 1
        remaining = num_t_subs
        curr_blocks = []
        used = set()
        
        while remaining > 0:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_candidates = []
            
            for i in range(num_k_subs):
                if i in used:
                    continue
                gain = get_gain(i, uncovered_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_gain <= 0:
                break
                
            # Randomly pick from best candidates to diversify
            chosen = random.choice(best_candidates)
            
            curr_blocks.append(chosen)
            used.add(chosen)
            uncovered_mask &= ~k_sub_masks[chosen]
            remaining -= best_gain
            
        # Post-process: Remove redundant blocks
        # This is crucial for reducing size.
        # Iterate through blocks, try to remove each one.
        # If removal doesn't uncover any set, keep it removed.
        
        # To check if removal is safe, we need to know if any t-subset in the block 
        # is covered ONLY by this block.
        
        # Build coverage map for current solution
        # Map t_idx -> list of block indices covering it? 
        # Or just count?
        # If count > 1, safe to remove? 
        # Yes, if count > 1, removing one block leaves count >= 1.
        
        # But we need to update counts as we remove.
        
        curr_cover_counts = [0] * num_t_subs
        for b in curr_blocks:
            for s_idx in k_sub_t_indices[b]:
                curr_cover_counts[s_idx] += 1
                
        # Try to remove blocks
        # We iterate multiple times because removing one might make another redundant?
        # No, removing a block only decreases counts. If a block was redundant (all its sets had count > 1),
        # removing it reduces counts. It might make *other* blocks redundant? 
        # No, if block A is redundant, removing it makes sets less covered.
        # It doesn't make block B redundant. Block B was already covering its sets.
        # So one pass is sufficient?
        # Actually, if we remove A, set S count drops from 2 to 1.
        # If block B also covers S, B is still essential for S? No, S is still covered by B.
        # B is redundant if *all* its sets are covered by others.
        # Removing A doesn't change the fact that B's other sets are covered by others.
        # So one pass is enough?
        # Wait. If we remove A, we check B.
        # If we remove B, we check C.
        # The order matters?
        # If A and B both cover S. S count = 2.
        # Remove A. S count = 1.
        # Now B is the only cover for S. B is NOT redundant.
        # If we checked B first: S count = 2. B is redundant (for S).
        # But B might cover T. T count = 1 (only B covers T).
        # So B is NOT redundant because of T.
        # So we must check ALL sets in the block.
        
        # Algorithm for Redundancy Removal:
        # 1. Calculate cover_counts for all sets.
        # 2. Iterate through blocks in curr_blocks.
        # 3. For each block B, check if there exists ANY set s in B such that cover_counts[s] == 1.
        #    If yes, B is essential. Keep it.
        #    If no (all s in B have cover_counts[s] > 1), B is redundant.
        #    Remove B. Update cover_counts for all s in B (decrement).
        # 4. Repeat this process? 
        #    Removing B reduces counts. This might make another block C redundant?
        #    No, C was already checked? Or not?
        #    If we iterate sequentially, and remove B, the counts change.
        #    Blocks already passed might now be redundant?
        #    Example: A and B cover S. C covers T. 
        #    If we remove A, S count drops. 
        #    If B was checked and found essential because of S (count 2 -> 1? No, if count 2, B is redundant for S).
        #    Wait, if count is 2, B is redundant for S.
        #    If B has no other critical sets, B is redundant.
        #    So we should loop until no more removals.
        
        changed = True
        while changed:
            changed = False
            # Recalculate counts? Or maintain?
            # Maintaining is faster.
            
            # We need to check blocks again.
            # To avoid O(N^2) if many removals, we can just do one pass per "round"
            # and repeat rounds.
            
            for i in range(len(curr_blocks) - 1, -1, -1):
                b_idx = curr_blocks[i]
                is_redundant = True
                for s_idx in k_sub_t_indices[b_idx]:
                    if curr_cover_counts[s_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove
                    for s_idx in k_sub_t_indices[b_idx]:
                        curr_cover_counts[s_idx] -= 1
                    curr_blocks.pop(i)
                    changed = True
                    
        # Check validity (should be valid if we started valid and only removed redundant)
        if is_valid(curr_blocks):
            if len(curr_blocks) < best_len:
                best_blocks = curr_blocks[:]
                best_len = len(curr_blocks)
                
        # Optional: Local Search (Swap) on the best found so far?
        # If time permits.
        
    # Final Conversion
    result = [list(all_k_subs[i]) for i in best_blocks]
    for b in result:
        b.sort()
    result.sort()
    return result
