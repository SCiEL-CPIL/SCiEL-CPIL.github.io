import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    # We will use a set for fast membership checks
    # We need to run multiple randomized greedy searches
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Helper to run one randomized greedy pass
    def run_greedy():
        uncovered = set(all_t_subs)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            # Converting to list is O(N), but N is small (max 3876)
            target = random.choice(list(uncovered))
            
            block = list(target)
            
            # Extend block to size k
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets 
                    # that are subsets of (block U {x}) and contain x
                    gain = 0
                    # A t-subset U of (block U {x}) containing x 
                    # is formed by a (t-1)-subset of block plus x
                    for sub in itertools.combinations(block, t - 1):
                        # Create the t-subset
                        # sub is a tuple of t-1 points
                        # Add x and sort
                        new_sub = tuple(sorted(sub + (x,)))
                        if new_sub in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                # Pick randomly from best points to add diversity
                next_point = random.choice(best_points)
                block.append(next_point)
                
                # Early termination check: if best_gain == 0, we are stuck?
                # No, we must add a point to reach size k.
                # But if gain is 0, we are adding a point that covers nothing new.
                # This is necessary to complete the block, but might be suboptimal.
                
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            # Remove all t-subsets of the new block
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    # Main loop: run multiple times
    while time.time() < deadline:
        blocks = run_greedy()
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # If we hit the theoretical lower bound or a very good number, we might want to stop?
            # Lower bound is 131. Best known 151.
            # If we find 145, keep looking.
        
        # Optional: Local Search
        # Given the time, simple restarts are safer.
        # We can add a simple "swap" local search if time permits.
        
    return best_blocks
