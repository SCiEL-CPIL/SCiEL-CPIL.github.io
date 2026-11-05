import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    best_blocks = None
    best_size = float('inf')
    
    # Helper: Greedy Cover with Randomization
    def greedy_cover(seed_val):
        rng = random.Random(seed_val)
        # Set of tuples for O(1) membership
        uncovered_tuples = set(all_t_subsets)
        # List of indices for O(1) random selection and removal (swap with last)
        uncovered_indices = list(range(num_t_subsets))
        blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered subset
            pos = rng.randrange(len(uncovered_indices))
            first_subset_idx = uncovered_indices[pos]
            first_subset = all_t_subsets[first_subset_idx]
            
            # Remove from data structures
            uncovered_tuples.remove(first_subset)
            uncovered_indices[pos] = uncovered_indices[-1]
            uncovered_indices.pop()
            
            block = list(first_subset)
            
            # Greedily add points
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Check combinations of block of size t-1
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered_tuples:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    break
                block.append(rng.choice(best_candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                if sub in uncovered_tuples:
                    # Need to remove from uncovered_indices too
                    # Finding index in list is O(N).
                    # To avoid O(N) search, we can just rebuild the list?
                    # Or use a different structure.
                    # Actually, we only remove ONE subset at a time from the "first" selection.
                    # Here we are removing MANY subsets.
                    # So the "swap with last" trick for the list is hard to maintain if we remove arbitrary elements.
                    
                    # Alternative: Just use a set of indices for membership, and a list for selection?
                    # If we remove arbitrary indices, the list becomes stale.
                    
                    # Simplest robust approach:
                    # Just use a set of indices for everything.
                    # Random selection: `idx = rng.choice(list(uncovered_set))`.
                    # This is O(N) per block.
                    # N starts at 11k.
                    # Number of blocks ~ 300.
                    # Total O(N^2) ~ 300 * 11k = 3.3M ops.
                    # This is very fast in Python.
                    
                    pass
                    
            # Let's refactor to use a set of indices for simplicity and correctness,
            # accepting the O(N) cost of list() conversion for random choice.
            # It's much simpler and less error-prone.
            
            # Re-doing the loop logic with Set of Indices
            
            pass

    # Revised Greedy
    def greedy_cover_v2(seed_val):
        rng = random.Random(seed_val)
        uncovered_set = set(range(num_t_subsets))
        blocks = []
        
        while uncovered_set:
            # Random choice from set
            first_subset_idx = rng.choice(list(uncovered_set))
            first_subset = all_t_subsets[first_subset_idx]
            uncovered_set.remove(first_subset_idx)
            
            block = list(first_subset)
            
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        cand_idx = t_subset_to_idx[cand]
                        if cand_idx in uncovered_set:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    break
                block.append(rng.choice(best_candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered_set.discard(t_subset_to_idx[sub])
        
        return blocks

    # Helper: Deletion Pass
    def deletion_pass(blocks):
        coverage = {s: 0 for s in all_t_subsets}
        
        for block in blocks:
            for sub in itertools.combinations(block, t):
                coverage[sub] += 1
        
        changed = True
        current_blocks = list(blocks)
        while changed:
            changed = False
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                block = current_blocks[i]
                can_remove = True
                for sub in itertools.combinations(block, t):
                    if coverage[sub] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for sub in itertools.combinations(block, t):
                        coverage[sub] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break
        
        return current_blocks

    # Main Loop
    run_id = 0
    while time.time() < deadline - safety_margin:
        blocks = greedy_cover_v2(run_id)
        blocks = deletion_pass(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        run_id += 1
        
    return best_blocks if best_blocks is not None else []
