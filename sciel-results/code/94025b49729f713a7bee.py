import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a randomized greedy algorithm
    with redundancy removal and multiple restarts.
    """
    # Deadline: 110 seconds to be safe
    deadline = time.time() + 110.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    # Map t-subset to index for faster lookup? 
    # Since v=19, t=4, there are 3876 subsets. 
    # Using a set of tuples is fast enough.
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts depends on time. Let's try a few.
    # Each greedy run is fast.
    # We will run until time is up or we find a very good solution.
    
    # To ensure we finish, we limit restarts or check time in loop.
    
    for seed in range(1000):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(seed)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to anchor the new block
            # This ensures we cover something new
            anchor = random.choice(list(uncovered))
            
            # Find all k-subsets containing anchor
            # anchor is a tuple of t elements.
            # Remaining points to choose: v - t
            # Points to choose: k - t
            remaining_points = [p for p in range(v) if p not in anchor]
            
            best_block = None
            best_gain = -1
            best_block_list = None # To handle ties randomly
            
            # Generate candidates
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor + combo))
                
                # Calculate gain: how many uncovered subsets are in this block
                gain = 0
                for t_sub in itertools.combinations(block, t):
                    if t_sub in uncovered:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_block_list = [block]
                elif gain == best_gain and gain > 0:
                    best_block_list.append(block)
            
            # Pick one of the best blocks randomly to introduce diversity
            if best_block_list:
                chosen_block = random.choice(best_block_list)
            else:
                # Should not happen if uncovered is not empty
                chosen_block = best_block
            
            blocks.append(list(chosen_block))
            
            # Update uncovered
            for t_sub in itertools.combinations(chosen_block, t):
                uncovered.discard(t_sub)
                
        # Refinement: Remove redundant blocks
        # We do this iteratively until no more can be removed
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                # Check if this block is redundant
                # A block is redundant if all its t-subsets are covered by other blocks
                is_redundant = True
                for t_sub in itertools.combinations(block, t):
                    # Check if t_sub is in any other block
                    covered = False
                    for j, other_block in enumerate(blocks):
                        if i == j:
                            continue
                        if t_sub in set(itertools.combinations(other_block, t)):
                            covered = True
                            break
                    if not covered:
                        is_redundant = False
                        break
                            
                if not is_redundant:
                    new_blocks.append(block)
                else:
                    changed = True
                    
            blocks = new_blocks
            
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            # Early exit if we hit the theoretical lower bound or known best?
            # Lower bound is 131. Known best is 151.
            # If we find 150, we can stop.
            if best_count <= 150:
                break

    if best_blocks is None:
        # Fallback: Return a valid covering (should not happen)
        # Just return the last one or a trivial one
        return blocks
        
    return best_blocks
