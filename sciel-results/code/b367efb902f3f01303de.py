import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute 4-subsets for the gain calculation logic?
    # We need to check if Q (4-subset) union {x} is in uncovered.
    # We will just use the subset_to_idx lookup.
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple randomized greedy passes
    # The number of passes depends on how fast one pass is.
    # Estimate: ~0.1s per pass? Let's try 100 passes max.
    
    for seed in range(200):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        uncovered = set(range(len(all_t_subsets)))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # This helps in diversifying the search
            start_idx = random.choice(list(uncovered))
            start_subset = all_t_subsets[start_idx]
            block = list(start_subset)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                gains = [0] * v
                block_set = set(block)
                
                # Iterate over all 4-subsets of the current block
                # If t=5, we need 4 points from block to form a 5-subset with x
                # Generally, we need (t-1) points from block.
                for rest in itertools.combinations(block, t - 1):
                    rest_tuple = tuple(rest)
                    # For each x not in block
                    for x in range(v):
                        if x not in block_set:
                            # Form the t-subset
                            # Note: rest_tuple + (x,) might not be sorted
                            # We need to check if this combination is in subset_to_idx
                            # Since subset_to_idx keys are sorted tuples, we must sort
                            # To avoid sorting overhead, we can precompute?
                            # Or just use a frozenset? No, tuple is faster for lookup if sorted.
                            # Let's optimize:
                            # rest is already sorted. x is just a number.
                            # We can construct the tuple efficiently.
                            # But sorting a 5-tuple is cheap.
                            cand = tuple(sorted(rest_tuple + (x,)))
                            idx = subset_to_idx.get(cand)
                            if idx is not None and idx in uncovered:
                                gains[x] += 1
                
                # Find max gain
                max_gain = -1
                best_candidates = []
                for x in range(v):
                    if x not in block_set:
                        if gains[x] > max_gain:
                            max_gain = gains[x]
                            best_candidates = [x]
                        elif gains[x] == max_gain:
                            best_candidates.append(x)
                
                if not best_candidates:
                    # Should not happen if uncovered is not empty and k > t
                    # But just in case, pick any
                    best_candidates = [x for x in range(v) if x not in block_set]
                
                # Random tie-breaking
                block.append(random.choice(best_candidates))
            
            block.sort()
            blocks.append(block)
            
            # Remove covered subsets
            # Generate all t-subsets of the new block
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx.get(sub)
                if idx is not None:
                    uncovered.discard(idx)
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Optional: if we hit a very good number, we could stop early?
            # But we don't know the true minimum. Let's keep running until time.
            
    if best_blocks is None:
        # Fallback to deterministic if random failed or time was too tight for even 1 run
        # This shouldn't happen as seed 0 will run.
        return []
        
    return best_blocks
