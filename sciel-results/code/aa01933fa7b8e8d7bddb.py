"""
Rewritten program for C(19, 7, 4) covering design.
Uses a randomized greedy algorithm with multiple restarts to escape local optima.
"""
import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts is not fixed; we run as many as fit in the time budget.
    # Each restart is a full greedy construction.
    
    while True:
        if time.time() > deadline:
            break
            
        # Randomize the starting point for this restart
        # We can shuffle the points or just rely on internal randomization
        random.seed()  # Re-seed to get different random numbers each restart
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Optional: Randomly select the first t-subset to cover
        # Instead of always min(uncovered), pick a random one
        if not uncovered:
            break
            
        # Pick a random uncovered t-subset to start the first block
        # This helps diversify the search
        seed_subset = random.choice(tuple(uncovered))
        
        while uncovered:
            # Start a new block with the seed subset
            block = list(seed_subset)
            
            # Add points to the block until it has size k
            while len(block) < k:
                # Find the point that adds the most new coverage
                max_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block]
                
                # To speed up, we can limit candidates or just iterate all
                # For v=19, k=7, this is small enough
                for x in candidates:
                    # Calculate gain: how many uncovered t-subsets does adding x cover?
                    # A t-subset is covered if it is a subset of the new block.
                    # The new t-subsets formed are those that contain x and t-1 elements from the current block.
                    gain = 0
                    # Iterate over all (t-1)-subsets of the current block
                    for rest in itertools.combinations(block, t - 1):
                        # Check if the t-subset (rest + x) is in uncovered
                        # We need to sort to compare with the stored tuples
                        candidate_t = tuple(sorted(rest + (x,)))
                        if candidate_t in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_point = x
                        # Randomization: if gain is 0, we might pick randomly?
                        # Or we can add a small random noise to gain?
                        # Let's stick to pure greedy but with random tie-breaking
                        # Actually, if max_gain is 0, we are stuck in a local optimum for this block.
                        # We should still add a point to complete the block.
                
                if best_point is None:
                    # Fallback: pick a random point
                    best_point = random.choice(candidates)
                
                block.append(best_point)
            
            # Sort the block for consistency
            block.sort()
            blocks.append(block)
            
            # Update uncovered sets
            # Remove all t-subsets of the new block from uncovered
            for t_sub in itertools.combinations(block, t):
                uncovered.discard(t_sub)
                
            # If we just covered everything, break
            if not uncovered:
                break
                
            # Prepare for next block: pick a new random seed from remaining uncovered
            seed_subset = random.choice(tuple(uncovered))
        
        # End of one restart
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Early exit if we found a very good solution? 
            # Lower bound is 131. If we hit 131, we can stop.
            if best_count <= 131:
                break

    if best_blocks is None:
        # Fallback: return a valid (though potentially large) covering
        # The greedy always produces a valid covering
        # If we timed out before finishing even one restart, we need to ensure we return something.
        # However, the loop structure above finishes one restart before checking time at the top.
        # If the first restart takes > 110s, we might return None.
        # To be safe, let's ensure we at least complete one greedy pass if we haven't.
        # But given v=19, k=7, one greedy pass is fast (<1s).
        pass
        
    return best_blocks
