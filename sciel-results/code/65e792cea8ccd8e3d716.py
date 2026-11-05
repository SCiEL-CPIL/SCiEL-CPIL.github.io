import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for faster counting? 
    # Or just use a set of tuples. Sets of tuples are reasonably fast.
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts to attempt
    # Since one run is very fast, we can try many.
    # Let's estimate: 1 run ~ 10-50ms? 
    # 1000 runs ~ 10-50s.
    # We will loop until deadline.
    
    while time.time() < deadline:
        blocks = greedy_covering(v, k, t, random_seed=random.randint(0, 2**31))
        
        # Post-processing: Remove redundant blocks
        blocks = remove_redundant_blocks(v, k, t, blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            # Optional: If we find a very good solution, we might spend time trying to improve it further
            # For now, just keep restarting to find a globally better one.
            
    if best_blocks is None:
        # Fallback: should not happen, but just in case
        return greedy_covering(v, k, t, random_seed=42)
        
    return best_blocks

def greedy_covering(v, k, t, random_seed=0):
    rng = random.Random(random_seed)
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset to start the block
        # Random selection helps escape local minima in the greedy choice of points
        start_subset = rng.choice(list(uncovered))
        block = list(start_subset)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            
            # Candidates are points not in block
            candidates = [x for x in range(v) if x not in block]
            
            # Shuffle candidates to break ties randomly
            rng.shuffle(candidates)
            
            for x in candidates:
                # Calculate gain: number of uncovered t-subsets covered by adding x to block
                # These are subsets of size t that include x and t-1 elements from block
                gain = 0
                # Generate all (t-1)-subsets of the current block
                for sub in itertools.combinations(block, t - 1):
                    # The new subset is sub + (x,)
                    # Sort to form a canonical tuple
                    new_sub = tuple(sorted(sub + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    # If we found a point that covers a lot, we might still want to check others?
                    # No, standard greedy picks max. But we shuffled, so first max is random among maxes.
                    # Actually, to ensure randomness among ties, we should collect all maxes or just rely on shuffle + strict >.
                    # With strict >, the first max encountered wins. Since shuffled, it's random.
                
                # Optimization: If gain is 0, we can break early? No, we need max.
                # But if best_gain is already high, maybe? No, keep it simple.
                
            if best_point == -1:
                # Should not happen if uncovered is not empty and k > t
                break
                
            block.append(best_point)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        # Remove all t-subsets of the new block from uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
            
    return blocks

def remove_redundant_blocks(v, k, t, blocks):
    # Count coverage for each t-subset
    # Using a dictionary: subset -> count
    cover_count = {}
    
    # Initialize counts
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] = cover_count.get(sub, 0) + 1
    
    # Try to remove blocks
    changed = True
    while changed:
        changed = False
        # Iterate through blocks in reverse or random order?
        # Random order might help avoid bias
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(blocks):
                continue
            block = blocks[i]
            
            # Check if any t-subset in this block has count 1
            # If all have count > 1, we can remove this block
            can_remove = True
            for sub in itertools.combinations(block, t):
                if cover_count.get(sub, 0) <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block i
                # Decrement counts
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                
                # Remove from list
                blocks.pop(i)
                changed = True
                # Adjust indices? Since we are iterating over a copy of indices, 
                # and popping changes indices, we should be careful.
                # Actually, if we pop i, all subsequent indices in the original list are shifted.
                # But we are iterating over `indices` which is a static list of original indices.
                # If we pop index i, the element originally at i+1 is now at i.
                # The index i+1 in `indices` now refers to the element originally at i+2.
                # This is safe as long as we don't access indices > len(blocks).
                # However, `indices` contains values up to len(blocks)-1 (original).
                # After pop, len(blocks) decreases.
                # So we check `if i >= len(blocks): continue`.
                
    return blocks
