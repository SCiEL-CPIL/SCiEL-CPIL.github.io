import itertools
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a randomized greedy algorithm
    followed by local search (block deletion).
    """
    # Set a safety margin for the 150s limit. 
    # The evaluator kills at 150s. We aim to finish by 145s.
    deadline = time.time() + 145.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for faster lookup
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts to attempt. We will run as many as we can before the deadline.
    # Each run consists of greedy construction + local search.
    while time.time() < deadline:
        # 1. Greedy Construction with Randomization
        uncovered = set(range(total_t_subsets))
        blocks = []
        
        # To introduce randomness, we can shuffle the order in which we pick uncovered sets
        # or break ties in gain calculation randomly.
        # Let's shuffle the list of uncovered set indices initially, but we need to pick
        # the "most constrained" or just "first" uncovered.
        # A common strategy: pick a random uncovered set to start a block.
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_set_idx = random.choice(list(uncovered))
            start_set = all_t_subsets[start_set_idx]
            
            block = list(start_set)
            
            # Greedily add points to the block
            while len(block) < k:
                # Calculate gain for each candidate point
                max_gain = -1
                best_points = []
                
                # Optimization: Precompute which t-subsets are formed by current block + x
                # For t=3, k=7, block size grows from 3 to 7.
                # Gains are determined by how many uncovered t-subsets contain x and 
                # (t-1) elements from the current block.
                
                # To speed this up, we can maintain a count of uncovered subsets for each (t-1)-subset?
                # Or just iterate. v=24, k=7 is small.
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Count how many uncovered t-subsets are covered by adding x
                    gain = 0
                    # A t-subset covered by adding x must contain x and t-1 elements from block
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(s, -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_points = [x]
                    elif gain == max_gain:
                        best_points.append(x)
                
                if not best_points:
                    # Should not happen if uncovered is not empty and v > k
                    break
                    
                # Randomly choose from best points to introduce diversity
                block.append(random.choice(best_points))
            
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            for s in itertools.combinations(block, t):
                idx = subset_to_idx[s]
                if idx in uncovered:
                    uncovered.discard(idx)
        
        # 2. Local Search: Block Deletion
        # Try to remove blocks that are redundant
        # We need to know how many times each t-subset is covered.
        # Let's do this iteratively.
        
        # Helper function to check if a block can be removed
        # and perform the deletion loop
        def try_delete_blocks(blocks, all_t_subsets, subset_to_idx):
            # Calculate coverage counts
            # count[i] = number of blocks covering all_t_subsets[i]
            count = [0] * len(all_t_subsets)
            for block in blocks:
                for s in itertools.combinations(block, t):
                    idx = subset_to_idx[s]
                    count[idx] += 1
            
            changed = True
            while changed:
                changed = False
                # Try to remove each block
                for i in range(len(blocks) - 1, -1, -1):
                    block = blocks[i]
                    # Check if all t-subsets in this block are covered by at least one other block
                    # i.e., count[idx] > 1 for all s in block
                    can_remove = True
                    for s in itertools.combinations(block, t):
                        idx = subset_to_idx[s]
                        if count[idx] <= 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        # Remove the block
                        # Update counts
                        for s in itertools.combinations(block, t):
                            idx = subset_to_idx[s]
                            count[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        # Restart the loop or continue? 
                        # Since counts changed, other blocks might now be removable or not.
                        # Breaking and restarting the outer while loop is safer.
                        break
            
            return blocks

        blocks = try_delete_blocks(blocks, all_t_subsets, subset_to_idx)
        
        # 3. Optional: Simple Local Search Improvement (Swap)
        # This can be expensive. We'll try a limited number of swaps.
        # For each block, try swapping one point with another.
        # If the new block allows the deletion of another block, accept.
        # This is complex to check quickly. 
        # Given time constraints, we might skip complex swaps and rely on restarts.
        # However, a simple "improve" step: 
        # Try to replace a block with a "better" one? 
        # Actually, the deletion step is the most impactful for reducing block count.
        
        current_count = len(blocks)
        if current_count < best_count:
            best_count = current_count
            best_blocks = [list(b) for b in blocks]
            
        # If we found a very good solution, we might want to break early?
        # The lower bound is 69. If we hit 69, we can break.
        if best_count <= 69:
            break
            
        # Check time
        if time.time() > deadline - 5: # 5s margin for cleanup
            break

    # If no blocks were found (should not happen), return empty
    if best_blocks is None:
        return []
        
    return best_blocks

# EVOLVE-BLOCK-END
