import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Total number of t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    total_subsets = len(all_subsets)
    
    # Map subset to index for faster lookups if needed, but sets of tuples are okay for 2024 items
    # We will maintain a set of uncovered subsets
    
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple randomized greedy instances
    num_trials = 1000 # Will be limited by time
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        
        # Greedy Construction
        uncovered = set(all_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # Using random choice from set is O(n) in Python, but n is small (max 2024)
            # To speed up, we can just pick the first one or use a list, but set is fine for now.
            # Actually, iterating through set to pick random is slow. 
            # Let's convert to list once? No, it changes.
            # Better: just pick min or arbitrary. Randomness comes from tie-breaking in gain.
            # Let's pick a random element from the set.
            # Python 3.7+ set iteration is ordered, but we can use next(iter(uncovered))
            # To get randomness, we can shuffle a list of uncovered? That's expensive.
            # Let's stick to deterministic start (first uncovered) but randomize gain ties.
            
            start_subset = next(iter(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break # Should not happen if k <= v
                
                # Pre-compute combinations of block of size t-1
                # If t=3, t-1=2.
                block_combos = list(itertools.combinations(block, t-1))
                
                max_gain = -1
                best_candidates = []
                
                for x in candidates:
                    gain = 0
                    for combo in block_combos:
                        # Create the t-subset
                        # combo is a tuple of t-1 elements. Add x.
                        # Need to check if this t-subset is in uncovered.
                        # Sorting is expensive. 
                        # Since combo is sorted and x is int, we can merge or just use tuple(sorted(...))
                        # For t=3, combo is (a,b). x is int.
                        # sorted((a,b,x)) is fast.
                        new_subset = tuple(sorted(combo + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [x]
                    elif gain == max_gain:
                        best_candidates.append(x)
                
                if max_gain == 0:
                    # No point improves coverage. Pick random to break deadlock?
                    # This should not happen in a valid covering search if we start from uncovered.
                    # But if we are forced to fill the block, pick random.
                    x = random.choice(candidates)
                else:
                    # Pick randomly among best candidates to add diversity
                    x = random.choice(best_candidates)
                
                block.append(x)
            
            # Add block
            blocks.append(tuple(sorted(block)))
            
            # Update uncovered
            # Remove all t-subsets of the new block from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        # Current solution size
        current_size = len(blocks)
        
        if current_size < best_size:
            best_size = current_size
            best_blocks = blocks
        
        # Optimization Phase: Try to remove redundant blocks
        # This is crucial for getting from ~80 to ~76
        if time.time() > deadline - 5.0: # Reserve time for final check
            continue
            
        opt_blocks = list(best_blocks)
        # We need to track coverage counts for optimization
        # Re-calculate coverage for the current best
        # This is expensive if done every time. 
        # Let's do optimization on the *current* trial's blocks before comparing?
        # Or optimize the best found so far?
        # Optimizing the best found so far is better.
        
        # Let's implement a simple local search on the current 'blocks'
        # 1. Compute coverage counts
        # 2. Try to remove redundant blocks
        # 3. Try swaps
        
        # For efficiency, let's only do a light optimization: 
        # Sort blocks by "uniqueness" (number of triples covered ONLY by this block)
        # If a block has 0 unique triples, it is redundant.
        
        # This calculation is O(B * C(k,t)). B~80, C(7,3)=35. 80*35 = 2800 ops. Very fast.
        
        # We'll do a loop of removals
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Compute coverage count for each triple
            cover_count = {sub: 0 for sub in all_subsets}
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    cover_count[sub] += 1
            
            # Find redundant blocks
            for i in range(len(blocks)):
                b = blocks[i]
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    if cover_count[sub] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    changed = True
                    break # Restart loop to recompute counts
        
        # Update best
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks

    # Final verification and formatting
    if best_blocks is None:
        # Fallback to a simple greedy if no time for loop (should not happen)
        uncovered = set(all_subsets)
        best_blocks = []
        while uncovered:
            start_subset = next(iter(uncovered))
            block = list(start_subset)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                max_gain = -1
                best_x = candidates[0]
                block_combos = list(itertools.combinations(block, t-1))
                for x in candidates:
                    gain = 0
                    for combo in block_combos:
                        if tuple(sorted(combo + (x,))) in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                block.append(best_x)
            best_blocks.append(tuple(sorted(block)))
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)

    # Convert to list of lists as required
    return [list(b) for b in best_blocks]
