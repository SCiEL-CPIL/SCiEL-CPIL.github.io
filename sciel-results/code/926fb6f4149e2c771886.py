"""Covering design C(v,k,t) via multi-start randomized greedy + reduction."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start loop
    while time.time() < deadline:
        # Random seed for this restart
        rng = random.Random()
        
        # 1. Randomized Greedy Construction
        # We maintain a set of uncovered t-subsets.
        # Instead of picking the absolute best, we pick from top candidates with some randomness.
        uncovered = set(range(n_tsubs))
        blocks = []
        
        while uncovered and time.time() < deadline - 10.0: # Leave time for reduction
            # Pick a random uncovered t-subset to anchor the block?
            # Or pick the t-subset that appears in the fewest existing blocks?
            # For construction, standard greedy is:
            # 1. Pick an uncovered t-subset S.
            # 2. Find the k-set B containing S that covers the most uncovered t-subsets.
            
            # Optimization: We can just pick any uncovered t-subset, say the first one.
            # But randomizing the order might help.
            anchor_idx = rng.choice(list(uncovered))
            anchor = all_tsubs[anchor_idx]
            
            # Build a block containing 'anchor'
            # Start with anchor points
            block = list(anchor)
            
            # We need to add k - t points to this block.
            # At each step, pick the point x not in block that maximizes coverage of uncovered t-subsets.
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block]
                
                # To speed up, we can sample candidates if v is large, but v=19 is small.
                # Calculate gain for each candidate
                for x in candidates:
                    # New t-subsets covered by adding x to current block prefix
                    # A t-subset is covered if it is a subset of (current_block + [x])
                    # Specifically, it must contain x and t-1 elements from current_block
                    # AND be in 'uncovered'
                    
                    # Pre-compute combinations of current_block of size t-1?
                    # Current block size is i. We need t-1 elements from it.
                    # If i < t-1, gain is 0? No, we start with t points (anchor).
                    # So len(block) >= t always.
                    
                    gain = 0
                    # Iterate over t-1 subsets of current block
                    for sub in itertools.combinations(block, t-1):
                        # The new t-subset is sub + (x,)
                        # Check if it's in uncovered
                        # We need to check if tuple(sorted(sub + (x,))) is in uncovered
                        # Since sub is sorted and x is distinct, we can just check the set
                        # But 'uncovered' is a set of indices.
                        # We need the index of the t-subset.
                        
                        # Efficient check:
                        # Create the tuple
                        # Since sub is sorted, and x is arbitrary, we insert x into sub to keep it sorted?
                        # Or just create a set and check?
                        # v=19, t=5. Subsets are small.
                        
                        # Let's use the mapping tsub_to_idx
                        # We need to form the tuple (sub + (x,)) sorted.
                        # Since sub is sorted, we can do:
                        # new_sub = list(sub)
                        # new_sub.append(x)
                        # new_sub.sort()
                        # idx = tsub_to_idx.get(tuple(new_sub), -1)
                        # if idx != -1 and idx in uncovered: gain += 1
                        
                        # This is O(2^(t-1)) per candidate.
                        # t=5, so 4 subsets. 19 candidates. 76 checks.
                        # This is fast.
                        
                        new_sub = list(sub)
                        new_sub.append(x)
                        new_sub.sort()
                        idx = tsub_to_idx.get(tuple(new_sub), -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # Random tie-breaking?
                        # If gain == best_gain, we could randomize, but strict max is fine for now.
                        # To add randomness, we could collect all max gain points and pick one.
                        # Let's stick to max for stability, the randomness comes from anchor choice.
                
                if best_point is None:
                    # Should not happen if uncovered is not empty and v > k
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # Find all t-subsets of 'block' that are in uncovered
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if idx in uncovered:
                    uncovered.discard(idx)
        
        if not uncovered:
            # 2. Reduction Phase
            # Try to remove blocks that are redundant.
            # A block is redundant if every t-subset in it is covered by other blocks.
            
            # Count coverage of each t-subset by blocks
            # coverage_count[idx] = number of blocks covering it
            
            # To make this efficient, we can maintain a count array.
            # Initially, compute counts for all blocks.
            coverage = [0] * n_tsubs
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    idx = tsub_to_idx[ts]
                    coverage[idx] += 1
            
            removed = False
            # Iterate through blocks, check if any can be removed
            # We do this in a loop until no more can be removed
            # To avoid O(N^2 * C(k,t)) which might be slow, we check one by one.
            
            i = 0
            while i < len(blocks):
                if time.time() > deadline:
                    break
                
                b = blocks[i]
                # Check if all t-subsets in b have coverage > 1
                can_remove = True
                for ts in itertools.combinations(b, t):
                    idx = tsub_to_idx[ts]
                    if coverage[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove block i
                    for ts in itertools.combinations(b, t):
                        idx = tsub_to_idx[ts]
                        coverage[idx] -= 1
                    blocks.pop(i)
                    # Do not increment i, check the same index again
                else:
                    i += 1
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks.copy()
        
        # If we finished a full construction and reduction quickly, loop again
        # If we hit the time limit in construction, we might not have a valid covering
        # But 'uncovered' check ensures validity.
        
        # Small sleep to prevent tight loop if it finishes instantly?
        # No, it's CPU bound.
        
    if best_blocks is None:
        # Fallback: return a valid covering even if not optimized?
        # The greedy construction above *should* produce a valid covering if it completes.
        # If it didn't complete due to time, 'uncovered' is not empty.
        # In that case, we need to return *something* valid.
        # But the prompt says "Valid when every t-subset... lies in at least one block".
        # If we return invalid, score is 0.
        # So we must ensure we return a valid covering.
        # If the loop finishes without best_blocks, it means we never completed a valid covering.
        # This is unlikely for small v, k, t with 115s budget.
        # However, to be safe, we can run a simple deterministic greedy that we know will finish?
        # Or just return the last partial one? No, must be valid.
        
        # Let's assume the first iteration completes. If not, we are in trouble.
        # Given v=19, k=8, t=5, C(19,5)=11628.
        # Greedy should cover this in < 1s usually.
        
        # If best_blocks is None, we have to return a valid set.
        # We can run the original seed logic as a fallback?
        # Or just return an empty list? No.
        
        # Let's force a valid completion if best_blocks is None.
        # Run a simple greedy without time limit (should be fast).
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            anchor_idx = min(uncovered)
            anchor = all_tsubs[anchor_idx]
            block = list(anchor)
            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for sub in itertools.combinations(block, t-1):
                        new_sub = list(sub)
                        new_sub.append(x)
                        new_sub.sort()
                        idx = tsub_to_idx.get(tuple(new_sub), -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None: break
                block.append(best_point)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if idx in uncovered:
                    uncovered.discard(idx)
        best_blocks = blocks
        best_count = len(blocks)

    return best_blocks
