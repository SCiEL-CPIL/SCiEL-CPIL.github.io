import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for faster lookup? 
    # Or just use a set of tuples.
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple independent greedy searches
    # Each search uses a random seed to vary the choices
    
    # Optimization: 
    # Instead of rebuilding the 'uncovered' set from scratch (which is O(N) where N=11628),
    # we can just use a set and remove elements.
    
    for seed in range(1000): # Loop until time is up
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to force the block to contain
            # This diversifies the search compared to always picking min(uncovered)
            # Alternatively, pick the one that is 'hardest' to cover? 
            # For now, random selection from uncovered.
            # But picking a random one might be slow if we iterate to find one.
            # Let's just pick the first one in a shuffled list? No, that's expensive.
            # Let's stick to picking a random element from the set.
            # Python's set doesn't support random access. 
            # We can convert to list? That's O(N). 
            # Better: maintain a list of uncovered? Or just pick min(uncovered) for determinism 
            # but randomize the *block selection* tie-breaking.
            
            # Actually, a common heuristic is to pick the uncovered subset with the 
            # fewest possible extensions to a k-block? That's complex.
            
            # Let's try: Pick the lexicographically smallest uncovered subset (deterministic core)
            # but randomize the choice among equally good blocks.
            # This ensures we make progress consistently but explores different block structures.
            
            target = min(uncovered)
            
            # We need to form a block of size k containing 'target'.
            # 'target' has t elements. We need k-t more.
            # Let fixed_points = set(target)
            # Let candidate_points = [x for x in range(v) if x not in fixed_points]
            # We need to choose k-t points from candidate_points.
            
            fixed = set(target)
            candidates = [x for x in range(v) if x not in fixed]
            
            # If k-t is small, we can iterate all combinations.
            # Here k-t = 3.
            best_gain = -1
            best_block = None
            best_tiebreak = -1
            
            # Iterate through all combinations of k-t points from candidates
            # To speed up, we can pre-compute the t-subsets of the potential block?
            # No, just generate the block and count covered uncovered subsets.
            
            # Optimization: 
            # The block will be target + chosen_points.
            # The t-subsets covered are all t-subsets of (target + chosen_points).
            # We can generate these on the fly.
            
            for comb in itertools.combinations(candidates, k - t):
                block = tuple(sorted(fixed.union(comb)))
                
                # Count how many t-subsets of this block are in 'uncovered'
                gain = 0
                for ts in itertools.combinations(block, t):
                    if ts in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early exit? No, we need max gain.
                            # But if we reach a theoretical max, we can stop.
                            # Max possible gain is C(k,t) = 56.
                            pass
                
                # Random tie-breaking
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_tiebreak = random.random()
                elif gain == best_gain:
                    r = random.random()
                    if r > best_tiebreak:
                        best_tiebreak = r
                        best_block = block
            
            if best_block is None:
                # Should not happen if uncovered is not empty
                break
                
            blocks.append(list(best_block))
            
            # Remove all t-subsets of best_block from uncovered
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)
        
        if time.time() > deadline:
            break
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
            # If we hit the lower bound (264), we can stop early
            if best_count <= 264:
                break

    if best_blocks is None:
        # Fallback to a simple deterministic greedy if time ran out before completing even one
        # Or just return whatever we have.
        # The loop above should have completed at least one if v,k,t are small.
        pass
        
    return best_blocks
