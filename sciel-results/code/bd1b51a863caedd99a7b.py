import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    # 1. Map triplets to indices
    triplets = list(itertools.combinations(range(v), t))
    triplet_index = {trip: i for i, trip in enumerate(triplets)}
    n_triplets = len(triplets)
    
    # 2. Generate all blocks and their masks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    block_masks = [0] * n_blocks
    
    for j, block in enumerate(blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triplet_index[triple]
            mask |= (1 << idx)
        block_masks[j] = mask
            
    # Greedy Search with Randomization
    best_covering = None
    best_size = float('inf')
    
    # Time limit check
    start_time = time.time()
    max_time = 110 # Leave buffer for evaluation overhead
    
    while time.time() - start_time < max_time:
        uncovered = (1 << n_triplets) - 1
        selected_blocks = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # Iterate through all blocks to find max gain
            # Use bit_count() for significant speedup over bin(...).count('1')
            for j in range(n_blocks):
                inter = block_masks[j] & uncovered
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [j]
                    elif gain == best_gain:
                        best_candidates.append(j)
                # If inter is 0, gain is 0. 
                # If best_gain is still -1, we might need to handle the case where no block covers anything (should not happen)
                # But if best_gain is >= 0, gain 0 is not a candidate for max.
                
            if best_gain == 0:
                # Should not happen if problem is solvable, but safe exit
                break
                
            # Randomly pick from best candidates
            chosen_j = random.choice(best_candidates)
            selected_blocks.append(blocks[chosen_j])
            uncovered &= ~block_masks[chosen_j]
            
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_covering = selected_blocks
            
        # Early termination if we hit the theoretical lower bound
        if best_size <= 69:
            break

    # Format output: list of lists of ints
    if best_covering is None:
        return []
        
    return [list(b) for b in best_covering]
