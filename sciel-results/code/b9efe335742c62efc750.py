import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    
    # Map each t-subset to an index for faster counting? 
    # Or just use the tuple itself in a dict/set.
    # Using a dict for counts is fine.
    
    uncovered = set(all_t_subsets)
    blocks = []
    counts = {t_set: 0 for t_set in all_t_subsets}
    
    # Precompute for each t-subset, the list of points that can extend it to a k-set?
    # Or just generate blocks on the fly.
    
    # Helper to generate k-subsets containing a specific t-subset
    def generate_blocks_containing(t_set):
        t_set = set(t_set)
        remaining_points = [p for p in points if p not in t_set]
        # We need to choose k-t more points from remaining_points
        for combo in itertools.combinations(remaining_points, k - t):
            yield tuple(sorted(tuple(t_set) + tuple(combo)))
            
    # Greedy Phase
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset
        # Using random choice to avoid bias and potentially improve diversity
        target_t_set = random.choice(list(uncovered))
        
        best_block = None
        best_gain = -1
        
        # Iterate through all blocks containing target_t_set
        for block in generate_blocks_containing(target_t_set):
            # Calculate gain: how many uncovered t-subsets does this block cover?
            gain = 0
            for t_sub in itertools.combinations(block, t):
                if t_sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            # Should not happen if uncovered is not empty
            break
            
        # Add best_block to solution
        blocks.append(best_block)
        
        # Update counts and uncovered
        for t_sub in itertools.combinations(best_block, t):
            counts[t_sub] += 1
            if counts[t_sub] == 1:
                uncovered.discard(t_sub)
                
    # Removal Phase (Local Search)
    # Try to remove redundant blocks
    # We repeat this process a few times to catch newly redundant blocks
    max_removal_rounds = 5
    for _ in range(max_removal_rounds):
        if time.time() > deadline:
            break
        changed = False
        # Iterate in reverse to allow removal during iteration? 
        # Or just collect candidates first.
        # Since removing a block might make another redundant, 
        # a single pass might not be enough, hence multiple rounds.
        
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = blocks[i]
            # Check if block is redundant
            # A block is redundant if for every t-subset T in block, 
            # there is another block in 'blocks' (excluding this one) that covers T.
            # i.e., counts[T] > 1 for all T in block.
            
            redundant = True
            for t_sub in itertools.combinations(block, t):
                if counts[t_sub] <= 1:
                    redundant = False
                    break
                    
            if redundant:
                # Remove block
                blocks.pop(i)
                for t_sub in itertools.combinations(block, t):
                    counts[t_sub] -= 1
                changed = True
        if not changed:
            break
            
    return blocks
