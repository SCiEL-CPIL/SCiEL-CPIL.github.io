import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy with bitset/boolean array tracking.
    """
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    
    # Map from sorted tuple to index
    # For speed, we can use a dictionary
    t_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute pairs for each point? No, we compute on the fly as block is small.
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s/150s kill
    
    # Run multiple randomized restarts
    restart_count = 0
    while time.time() - start_time < time_limit:
        restart_count += 1
        # Random seed for tie-breaking
        rng = random.Random(restart_count)
        
        # uncovered[i] is True if all_t_subsets[i] is not covered
        uncovered = np.ones(N_t, dtype=bool)
        blocks = []
        
        # Greedy construction
        while np.any(uncovered):
            # Find the first uncovered subset
            # np.argmax returns the index of the first True
            idx = np.argmax(uncovered)
            current_t = all_t_subsets[idx]
            
            # Start a new block with one element from the uncovered subset
            # To add randomness, we could pick a random element from the subset
            # But picking the first one is standard. Let's pick a random one to diversify.
            start_point = rng.choice(list(current_t))
            block = [start_point]
            
            # Grow the block
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check all pairs in the current block
                    # The new triplets formed are {x, a, b} for a, b in block
                    for a, b in itertools.combinations(block, t - 1):
                        # Form the triplet
                        # Since t=3, t-1=2, so we are forming triplets {x, a, b}
                        triplet = tuple(sorted((x, a, b)))
                        t_idx = t_to_idx[triplet]
                        if uncovered[t_idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                # Pick a candidate randomly from the best ones
                if best_candidates:
                    next_point = rng.choice(best_candidates)
                    block.append(next_point)
                else:
                    # Should not happen if uncovered is not empty and k < v
                    break
            
            # Finalize the block
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # For all t-subsets of the new block, mark as covered
            for sub in itertools.combinations(block, t):
                t_idx = t_to_idx[sub]
                uncovered[t_idx] = False
        
        # Check if this run is better
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # Optional: If we hit a very good score, we could stop early?
            # But we want to try to beat 76. 76 is the known best.
            # If we find 76, we can keep looking for 75? Unlikely.
            # Let's keep going until time runs out to ensure we find the best possible within time.
            # However, if we find a score significantly better than known best, we might stop.
            # But 76 is the target.
    
    return best_blocks
