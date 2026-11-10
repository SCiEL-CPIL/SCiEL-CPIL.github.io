import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design using a hybrid of weighted greedy initialization 
    and iterative local search (redundancy removal and repair).
    
    Parameters:
    v: Number of points
    k: Size of each block
    t: Size of subsets to cover
    
    Returns:
    List of blocks (each a list of k distinct integers in 0..v-1)
    """
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for efficient counting
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Helper to get indices of t-subsets contained in a block
    def get_block_t_indices(block):
        block_set = set(block)
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[ts])
        return indices

    # 1. Initial Greedy Construction
    # We use a simple greedy: pick the block that covers the most uncovered t-subsets.
    # To break ties and add some diversity, we can randomize slightly or use point degrees.
    
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    # Precompute for each point, the t-subsets it belongs to? 
    # That's too big. 
    # Instead, for each candidate block, count covered uncovered subsets.
    
    # Optimization: We can't iterate all C(19,8) blocks every time.
    # We use the standard greedy: 
    # Pick the first uncovered subset S.
    # We must cover S. 
    # We extend S to a k-subset B.
    # To maximize coverage, we pick the extension that covers the most OTHER uncovered subsets.
    
    while uncovered:
        # Pick a random uncovered subset to break symmetry and avoid getting stuck in a specific order
        # Or pick the one with the "highest bottleneck" (fewest ways to cover it in future?).
        # For now, random pick from uncovered for diversity, or min for determinism.
        # Let's use random to help local search later.
        first_uncovered_idx = random.choice(list(uncovered))
        first_uncovered = all_t_subsets[first_uncovered_idx]
        
        # We need to form a block containing first_uncovered.
        # The block must have k points. first_uncovered has t points.
        # We need to choose k-t more points from the remaining v-t points.
        
        # To find the best extension, we can use a greedy step for the extension points.
        current_block = list(first_uncovered)
        
        # Calculate current coverage of this partial block? 
        # No, we build the block first, then add it.
        
        # Greedy extension:
        # Add points one by one to maximize the number of uncovered t-subsets 
        # that are subsets of the final block.
        
        # This is still expensive if done naively.
        # Approximation: 
        # Pick the point that maximizes the number of uncovered t-subsets 
        # that contain the current block's points (minus 1) and the new point.
        
        while len(current_block) < k:
            best_point = -1
            best_gain = -1
            candidates = [x for x in range(v) if x not in current_block]
            
            # Sort candidates to allow for some randomness or deterministic order
            # For speed, we might not check all candidates if v is large, but v=19 is small.
            
            for x in candidates:
                # Calculate gain: how many uncovered t-subsets are formed by 
                # combining x with t-1 points from current_block?
                # This is an underestimate of the final block's coverage, but a good heuristic.
                gain = 0
                # Iterate over t-1 combinations of current_block
                for combo in itertools.combinations(current_block, t - 1):
                    ts = tuple(sorted(combo + (x,)))
                    if ts in t_subset_to_idx:
                        idx = t_subset_to_idx[ts]
                        if idx in uncovered:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    # Tie-breaking: random
                    # if gain == best_gain:
                    #     if random.random() < 0.5: best_point = x
            
            current_block.append(best_point)
        
        current_block.sort()
        blocks.append(current_block)
        
        # Update uncovered
        block_indices = get_block_t_indices(current_block)
        for idx in block_indices:
            uncovered.discard(idx)
            
    # 2. Local Search / Refinement
    # We now have a valid covering. We try to reduce the number of blocks.
    
    # Data structure for fast redundancy check:
    # count[idx] = number of blocks covering t-subset idx
    
    def calculate_counts(blocks):
        counts = [0] * num_t_subsets
        for b in blocks:
            for idx in get_block_t_indices(b):
                counts[idx] += 1
        return counts
    
    current_counts = calculate_counts(blocks)
    
    # Helper to check if a block is removable
    def is_removable(block, counts):
        for idx in get_block_t_indices(block):
            if counts[idx] == 1:
                return False
        return True
    
    # Local Search Loop
    start_time = time.time()
    time_limit = 110  # Leave some buffer for the evaluator
    
    while time.time() - start_time < time_limit:
        # Step 1: Remove redundant blocks
        removed = False
        new_blocks = []
        for b in blocks:
            if is_removable(b, current_counts):
                # Remove it
                for idx in get_block_t_indices(b):
                    current_counts[idx] -= 1
                removed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
        if not removed:
            # No blocks were removable. Perform a "kick" to escape local optimum.
            # Pick a random block, remove it, and repair.
            if blocks:
                kick_idx = random.randint(0, len(blocks) - 1)
                kicked_block = blocks[kick_idx]
                
                # Update counts for removal
                for idx in get_block_t_indices(kicked_block):
                    current_counts[idx] -= 1
                    if current_counts[idx] < 0:
                        current_counts[idx] = 0 # Safety
                
                blocks.pop(kick_idx)
                
                # Repair: Cover the subsets that are now uncovered (count == 0)
                uncovered = set()
                for i in range(num_t_subsets):
                    if current_counts[i] == 0:
                        uncovered.add(i)
                
                # Greedily cover the uncovered subsets
                while uncovered:
                    # Pick a random uncovered subset
                    target_idx = random.choice(list(uncovered))
                    target_subset = all_t_subsets[target_idx]
                    
                    # Construct a block containing target_subset
                    current_block = list(target_subset)
                    
                    # Greedy extension
                    while len(current_block) < k:
                        best_point = -1
                        best_gain = -1
                        candidates = [x for x in range(v) if x not in current_block]
                        
                        for x in candidates:
                            gain = 0
                            for combo in itertools.combinations(current_block, t - 1):
                                ts = tuple(sorted(combo + (x,)))
                                if ts in t_subset_to_idx:
                                    idx = t_subset_to_idx[ts]
                                    if idx in uncovered:
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = x
                        current_block.append(best_point)
                    
                    current_block.sort()
                    blocks.append(current_block)
                    
                    # Update counts
                    for idx in get_block_t_indices(current_block):
                        current_counts[idx] += 1
                        uncovered.discard(idx)
                        
    return blocks

# EVOLVE-BLOCK-END
