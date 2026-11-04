import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_t_subsets.append(mask)
        
    # Initial uncovered set
    initial_uncovered = set(all_t_subsets)
    
    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = set(initial_uncovered)
        blocks = []
        
        while uncovered:
            # Pick random anchor
            anchor_mask = random.choice(list(uncovered))
            
            # Convert mask to list of points for the block
            # We need the actual points to form the block
            # Find the t points corresponding to anchor_mask
            # We can precompute a mapping from mask to tuple?
            # Or just iterate.
            
            # To get points from mask:
            anchor_points = []
            temp_mask = anchor_mask
            for i in range(v):
                if temp_mask & (1 << i):
                    anchor_points.append(i)
                    temp_mask &= ~(1 << i)
                    
            block = list(anchor_points)
            uncovered.discard(anchor_mask)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                best_newly_covered = set()
                
                candidates = [x for x in range(v) if x not in block]
                
                if not candidates:
                    break
                    
                # Precompute 4-subsets of block as masks
                block_4_subsets_masks = []
                for sub4 in itertools.combinations(block, t - 1):
                    mask = 0
                    for x in sub4:
                        mask |= (1 << x)
                    block_4_subsets_masks.append(mask)
                    
                for x in candidates:
                    gain = 0
                    newly_covered = set()
                    x_bit = 1 << x
                    
                    for sub4_mask in block_4_subsets_masks:
                        candidate_mask = sub4_mask | x_bit
                        if candidate_mask in uncovered:
                            gain += 1
                            newly_covered.add(candidate_mask)
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_newly_covered = newly_covered
                        if gain == len(block_4_subsets_masks):
                            break
                            
                if best_x is None:
                    break
                    
                block.append(best_x)
                
                if best_newly_covered:
                    uncovered.difference_update(best_newly_covered)
                    
            # Convert block to sorted list of ints for output
            block.sort()
            blocks.append(block)
            
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        blocks = run_greedy(random_seed=i)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = run_greedy(random_seed=0)
        
    return best_blocks
