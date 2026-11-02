import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    # For pruning, we need to know coverage counts later.
    # For now, just build.
    
    blocks = []
    
    # Helper to generate a block covering a specific target t-subset
    def generate_block(target):
        block = list(target)
        remaining = set(range(v)) - set(block)
        
        # We need to add k - t points
        # Greedy: pick point that maximizes coverage of currently uncovered t-subsets
        for _ in range(k - t):
            best_point = None
            max_gain = -1
            best_gain_set = None # To break ties or for future use
            
            # Precompute the (t-1) subsets of the current block to speed up gain calculation
            # A t-subset covered by adding x must contain x and t-1 elements from block
            block_t_minus_1 = list(itertools.combinations(block, t - 1))
            
            for x in remaining:
                gain = 0
                # Check which t-subsets formed by x + (t-1) from block are uncovered
                for rest in block_t_minus_1:
                    subset = tuple(sorted(rest + (x,)))
                    if subset in uncovered:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_point = x
                    # Early exit if we found a perfect gain? Max possible is len(block_t_minus_1)
                    if gain == len(block_t_minus_1):
                        break
                
            if best_point is None:
                # Fallback to random if something goes wrong
                best_point = random.choice(list(remaining))
            
            block.append(best_point)
            remaining.remove(best_point)
            
        block.sort()
        return tuple(block)

    # Main construction loop
    while uncovered:
        if time.time() > deadline - 10.0:
            break
            
        # Pick a target
        # Heuristic: Pick a subset that is "rare"? 
        # We don't have coverage counts yet. 
        # Just pick the first one in the set (arbitrary but deterministic).
        # To introduce variation, we could shuffle, but determinism is good for reproducibility unless we use random seed.
        # The task says "seed/value", implying determinism might be linked to seed.
        # Let's use a deterministic pick: min(uncovered)
        target = min(uncovered)
        
        new_block = generate_block(target)
        blocks.append(new_block)
        
        # Update uncovered
        # Remove all t-subsets of new_block from uncovered
        for sub in itertools.combinations(new_block, t):
            uncovered.discard(sub)
            
    # Pruning Phase
    # Calculate coverage counts
    coverage_counts = {sub: 0 for sub in all_t_subsets}
    for b in blocks:
        for sub in itertools.combinations(b, t):
            coverage_counts[sub] += 1
            
    # Iterate pruning
    changed = True
    while changed:
        if time.time() > deadline - 5.0:
            break
        changed = False
        new_blocks = []
        for b in blocks:
            # Check if b is essential
            essential = False
            for sub in itertools.combinations(b, t):
                if coverage_counts[sub] == 1:
                    essential = True
                    break
            
            if not essential:
                # Remove b
                for sub in itertools.combinations(b, t):
                    coverage_counts[sub] -= 1
                changed = True
            else:
                new_blocks.append(b)
        
        blocks = new_blocks
        
    # Convert to lists of ints as required
    return [list(b) for b in blocks]
