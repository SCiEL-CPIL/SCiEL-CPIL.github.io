import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precompute t-subsets and mapping
    all_t_subs = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    all_ones_mask = (1 << num_t_subs) - 1
    
    # Helper to get mask for a block
    def get_block_mask(block):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_to_idx[ts])
        return mask

    # 2. Greedy Initialization
    covered_mask = 0
    blocks = []
    block_masks = []
    
    while covered_mask != all_ones_mask:
        if time.time() > deadline:
            break
            
        # Find an uncovered t-subset
        # Find the first set bit in ~covered_mask
        uncovered = ~covered_mask & all_ones_mask
        if uncovered == 0:
            break
            
        # Get index of one uncovered subset
        # lowbit trick
        lowbit = uncovered & -uncovered
        target_idx = lowbit.bit_length() - 1
        target_ts = all_t_subs[target_idx]
        
        # Find candidate blocks containing target_ts
        # A block contains target_ts if it is a superset of target_ts
        # We can iterate through all points not in target_ts to form the rest of the block
        # Block size k=7, t=4. Need 3 more points from v-4=15 points.
        # C(15, 3) = 455 candidates.
        
        best_gain = -1
        best_block = None
        best_mask = 0
        
        # Generate combinations of 3 points from the remaining 15
        remaining_points = [p for p in range(v) if p not in target_ts]
        for combo in itertools.combinations(remaining_points, k - t):
            candidate_block = tuple(sorted(list(target_ts) + list(combo)))
            
            # Check if this block is already in our set? 
            # Greedy usually adds new blocks. If we add a duplicate, gain is 0.
            # We can skip duplicates to save time, but 455 is small.
            
            cand_mask = get_block_mask(candidate_block)
            gain = (cand_mask & uncovered).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
                best_mask = cand_mask
        
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        block_masks.append(best_mask)
        covered_mask |= best_mask

    # 3. Local Search
    # We will alternate between removing redundant blocks and swapping.
    
    def is_redundant(idx):
        # Check if block idx covers any t-subset that no other block covers
        # Union of all other blocks
        others_mask = 0
        for j in range(len(block_masks)):
            if j != idx:
                others_mask |= block_masks[j]
        
        # Unique part of block idx
        unique_part = block_masks[idx] & ~others_mask
        return unique_part == 0

    def try_remove():
        # Try to remove any redundant block
        for i in range(len(block_masks) - 1, -1, -1):
            if is_redundant(i):
                # Remove block i
                blocks.pop(i)
                block_masks.pop(i)
                # Update covered_mask
                # We need to recompute covered_mask? 
                # Or we can just recompute it from scratch if it's cheap?
                # Recomputing covered_mask: OR of all block_masks.
                # O(N) ops. N=150. Very fast.
                new_covered = 0
                for m in block_masks:
                    new_covered |= m
                return True, new_covered
        return False, covered_mask

    # Main Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Remove redundant
        while True:
            if time.time() > deadline:
                return blocks
            removed, covered_mask = try_remove()
            if not removed:
                break
            improved = True
        
        # Pass 2: Swap
        # Pick a random block to swap
        if blocks:
            idx = random.randint(0, len(block_masks) - 1)
            old_block = blocks[idx]
            old_mask = block_masks[idx]
            
            # Compute unique part of old_block
            # Union of others
            others_mask = 0
            for j in range(len(block_masks)):
                if j != idx:
                    others_mask |= block_masks[j]
            
            unique_part = old_mask & ~others_mask
            
            # We need to find a new block B such that:
            # 1. B is not in current blocks
            # 2. B covers unique_part (so coverage is maintained)
            # 3. Ideally, B covers more than old_block, or allows other removals?
            # For now, just maintain coverage.
            
            # Strategy: Pick a t-subset from unique_part (if any) or from old_block
            # If unique_part is empty, the block was redundant, which should have been removed in Pass 1.
            # So unique_part should not be empty.
            
            # Pick a target t-subset from unique_part
            # lowbit trick
            lowbit = unique_part & -unique_part
            target_idx = lowbit.bit_length() - 1
            target_ts = all_t_subs[target_idx]
            
            # Find candidate blocks containing target_ts
            # Exclude the old_block itself
            remaining_points = [p for p in range(v) if p not in target_ts]
            found_swap = False
            
            # Randomize the order of candidates to avoid local minima?
            # Or just pick the first valid one?
            # Let's pick a few random candidates and test them.
            
            candidates = list(itertools.combinations(remaining_points, k - t))
            random.shuffle(candidates)
            
            for combo in candidates:
                if time.time() > deadline:
                    break
                
                candidate_block = tuple(sorted(list(target_ts) + list(combo)))
                
                # Check if candidate is already in blocks
                if candidate_block in blocks:
                    continue
                
                cand_mask = get_block_mask(candidate_block)
                
                # Check if cand_mask covers unique_part
                if (cand_mask & unique_part) != unique_part:
                    continue
                
                # Tentative swap
                # New covered mask = (covered_mask ^ old_mask) | cand_mask ?
                # No, covered_mask is the union.
                # New covered = (union of others) | cand_mask
                # We know union of others = covered_mask & ~unique_part? 
                # Actually, covered_mask = (union of others) | old_mask
                # unique_part = old_mask & ~(union of others)
                # So union of others = covered_mask & ~unique_part ? 
                # Not exactly, because old_mask might have parts covered by others.
                # Let's just compute new_covered = others_mask | cand_mask
                
                new_covered = others_mask | cand_mask
                
                # Check if new_covered is a valid cover (all_ones)
                if new_covered == all_ones_mask:
                    # Perform swap
                    blocks[idx] = list(candidate_block)
                    block_masks[idx] = cand_mask
                    covered_mask = new_covered
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap:
                # Try a different strategy? Or just stop?
                # If we can't swap, we might be stuck.
                # Random restart?
                # For now, break inner loop if no swap found after some tries?
                # The loop above tries all candidates for that target.
                # If no swap found, we can try a different block idx?
                # To keep it simple, we just continue the outer loop.
                # But if we always pick the same idx, we might loop.
                # Let's just break the swap pass if no swap was made in this iteration.
                # Actually, the outer while loop will repeat.
                # To avoid infinite loop with no progress, we should ensure we eventually try different things.
                # Since we shuffle candidates, it might find a swap later.
                # But if no swap is possible for this block, we should move on.
                # Let's add a counter or just rely on the fact that if no swap is found, 
                # the next iteration will pick a different random idx.
                pass

        if not improved and not blocks:
            break

    return blocks
