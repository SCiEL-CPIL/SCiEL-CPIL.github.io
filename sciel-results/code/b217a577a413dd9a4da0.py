import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to check if a list of blocks covers all t-subsets
    def is_valid_cover(blocks):
        if not blocks:
            return False
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        return covered == all_tsubsets_set

    # Helper to remove redundant blocks from a valid cover
    def reduce_cover(blocks):
        # Work on a copy
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                # Try removing block i
                if i >= len(current_blocks):
                    continue
                temp_blocks = current_blocks[:i] + current_blocks[i+1:]
                if is_valid_cover(temp_blocks):
                    current_blocks = temp_blocks
                    changed = True
                    break # Restart loop as indices changed
        return current_blocks

    # Greedy algorithm with randomization
    def greedy_randomized(seed):
        random.seed(seed)
        uncovered = set(all_tsubsets)
        blocks = []
        
        # Shuffle the order of initial consideration? 
        # Actually, standard greedy picks min uncovered. 
        # To randomize, we can pick a random uncovered subset first, 
        # or randomize tie-breaking in gain calculation.
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            # This diversifies the search compared to always picking min
            first_subset = random.choice(list(uncovered))
            block = list(first_subset)
            
            # Greedily add points to the block
            while len(block) < k:
                best_point = -1
                best_gain = -1
                # Randomize order of candidates to break ties randomly
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                
                for x in candidates:
                    # Calculate gain: how many new uncovered t-subsets does adding x cover?
                    # A t-subset is covered if it contains x and t-1 other points from the current block
                    # AND it is currently in 'uncovered'
                    gain = 0
                    # Optimization: only check combinations of (block) of size t-1
                    # that combined with x form a t-subset
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    # Tie-breaking is handled by random shuffle of candidates,
                    # but we should ensure we don't miss a strictly better gain.
                    # The loop above finds the max gain. If multiple have max gain,
                    # the first one encountered (randomized) is kept.
                
                if best_point == -1:
                    break # Should not happen if uncovered is not empty and v > k
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        
        return blocks

    # Run multiple randomized greedy restarts
    num_restarts = 0
    while time.time() < deadline - 10.0: # Reserve 10s for reduction
        seed = num_restarts
        blocks = greedy_randomized(seed)
        
        # Reduce the cover
        reduced_blocks = reduce_cover(blocks)
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = reduced_blocks
            # Early exit if we hit the theoretical lower bound (69)
            if best_count <= 69:
                break
        
        num_restarts += 1
        
        # Check time again
        if time.time() > deadline - 10.0:
            break

    # Final check: if we have no blocks (shouldn't happen), return a valid fallback
    if best_blocks is None:
        # Fallback to a simple deterministic greedy if no time was spent or error
        uncovered = set(all_tsubsets)
        blocks = []
        while uncovered:
            first_subset = min(uncovered)
            block = list(first_subset)
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        best_blocks = blocks
        best_count = len(best_blocks)

    return best_blocks
