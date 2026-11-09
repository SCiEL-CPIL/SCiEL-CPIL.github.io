import time
import itertools
import random
import sys
from collections import defaultdict

# We keep the same entry point: construct(v, k, t) -> list of blocks.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute all t-subsets and assign them IDs
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    sub_to_id = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # Function to get mask for a block
    def get_block_mask(block):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << sub_to_id[sub])
        return mask

    # Greedy construction
    def greedy_construct(deadline, rng):
        covered_mask = 0
        blocks = []
        block_masks = []
        
        # Precompute masks for all possible blocks? 
        # C(24,7) = 346,104. 
        # Storing them might be memory heavy but fast.
        # Let's compute on the fly for now, or precompute if time allows.
        # Given 120s, precomputing 346k masks might take a few seconds.
        # Let's try to precompute if we have time, else on-the-fly.
        
        # For speed, let's just compute on the fly in the greedy step.
        # We need to find the block with max gain.
        # Gain = popcount(block_mask & ~covered_mask)
        
        # To speed up, we can iterate through all possible blocks? 
        # 346,000 blocks. For each, compute mask (35 subsets) and popcount.
        # 346k * 35 ops = 12M ops per greedy step.
        # ~70 steps. 12M * 70 = 840M ops. Too slow in Python.
        
        # Alternative: Greedy by selecting a random uncovered subset, 
        # then finding the best block containing it.
        
        uncovered_mask = (1 << n_tsubs) - 1
        
        while uncovered_mask and time.time() < deadline:
            # Pick a random uncovered subset
            # Find index of a set bit in uncovered_mask
            # Python int doesn't have efficient "find next set bit" other than bit_length tricks or loops
            # But we can just pick a random index and check if it's set?
            # Or iterate through set bits.
            
            # Let's just pick a random subset from the list of all subsets
            # and check if it's uncovered.
            # This is inefficient if most are covered.
            
            # Better: maintain a list of uncovered subset IDs?
            # Or just use the mask.
            
            # Let's try a different approach for greedy:
            # 1. Pick a random uncovered subset S.
            # 2. Consider all blocks containing S.
            # 3. Pick the one with max gain.
            
            # How to pick random uncovered subset?
            # We can store uncovered subsets in a list and remove them.
            # But removal from list is O(N).
            # Let's use a set of uncovered subset IDs.
            
            # Reset strategy: Use set of uncovered IDs.
            pass

    # Let's stick to the set-based approach for greedy, it was working.
    # The bug was in the local search.
    
    # Refined Local Search
    def local_search(blocks, v, k, t, deadline, rng):
        # Precompute masks for current blocks
        block_masks = [get_block_mask(b) for b in blocks]
        
        # Function to check if block i is redundant
        def is_redundant(i):
            # covered by others
            covered_others = 0
            for j in range(len(blocks)):
                if j != i:
                    covered_others |= block_masks[j]
            # If block i's mask is subset of covered_others, it's redundant
            return (block_masks[i] & ~covered_others) == 0

        # Pass 1: Remove redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if is_redundant(i):
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break # Restart loop

        # Pass 2: Try to swap blocks
        # For each block, try to replace it with a better one
        # This is expensive. We'll do a few iterations.
        
        for iteration in range(10): # Limit iterations
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                
                # Check if block i is "critical" (covers some unique subsets)
                covered_others = 0
                for j in range(len(blocks)):
                    if j != i:
                        covered_others |= block_masks[j]
                
                unique_to_i = block_masks[i] & ~covered_others
                
                if unique_to_i == 0:
                    # Should have been removed in Pass 1, but just in case
                    blocks.pop(i)
                    block_masks.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block that covers unique_to_i
                # and potentially allows removal of other blocks.
                # We sample some random blocks containing the subsets in unique_to_i.
                # Actually, just try to find ANY block that covers unique_to_i.
                
                # Get the subsets in unique_to_i
                # We need to iterate through bits of unique_to_i
                subs_in_unique = []
                u = unique_to_i
                while u:
                    lsb = u & -u
                    idx = lsb.bit_length() - 1
                    subs_in_unique.append(all_tsubs[idx])
                    u ^= lsb
                
                if not subs_in_unique:
                    continue

                # Try to construct a new block greedily to cover these subsets
                # Start with the first subset
                new_block = list(subs_in_unique[0])
                
                # Add points to cover more of unique_to_i and others
                # This is complex. 
                # Simpler: Try random blocks that contain all subsets in subs_in_unique?
                # If subs_in_unique has many subsets, it's hard.
                
                # Let's just try to replace with a block that covers the FIRST unique subset
                # and see if it helps.
                
                target_sub = subs_in_unique[0]
                
                # Find all blocks containing target_sub
                # This is expensive.
                
                # Given time constraints, let's just do the removal pass and 
                # a simple swap: replace block i with a random block that covers 
                # at least one of its unique subsets, and see if total count drops.
                
                # This is getting complicated. Let's stick to a robust removal pass 
                # and a few restarts of the whole process.
                
                pass
        
        return blocks

    # Main Logic
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Multiple restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        
        rng_restart = random.Random(42 + restart * 1000)
        
        # Greedy Construction
        uncovered = set(all_tsubs)
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Pick random uncovered subset
            target = rng_restart.choice(list(uncovered))
            block = list(target)
            
            # Grow block
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        if not uncovered:
            # Local Search
            blocks = local_search(blocks, v, k, t, deadline, rng_restart)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
    if best_blocks:
        return best_blocks
    else:
        return []

# Note: The above code has issues with scope and efficiency.
# I will write the final clean version below.
