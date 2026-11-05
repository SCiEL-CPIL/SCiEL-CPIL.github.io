import itertools
import time
import random

def construct(v, k, t):
    # Parameters for the specific task C(19, 8, 5)
    # v=19, k=8, t=5
    
    all_points = list(range(v))
    
    # Pre-compute all t-subset masks
    # Mask is an integer where bit i is 1 if point i is in the subset
    all_t_masks = []
    for combo in itertools.combinations(all_points, t):
        m = 0
        for p in combo:
            m |= (1 << p)
        all_t_masks.append(m)
        
    num_t_subsets = len(all_t_masks)
    
    def get_gain(block_mask):
        """
        Calculate the number of currently uncovered t-subsets covered by block_mask.
        block_mask is an integer mask of the k points.
        """
        gain = 0
        # Iterate over all t-subsets of the block
        # Since k=8, t=5, there are C(8,5)=56 subsets per block.
        # We need to extract points from mask to generate subsets?
        # Or precompute t-subsets for all possible k-blocks? 
        # C(19,8) is ~75k. 75k * 56 masks is ~4M operations. Might be heavy to precompute fully.
        # Better: On the fly.
        
        # Extract points from block_mask
        pts = []
        for i in range(v):
            if (block_mask >> i) & 1:
                pts.append(i)
                
        for sub in itertools.combinations(pts, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            if cover_count[m] == 0:
                gain += 1
        return gain

    def run_greedy():
        blocks = []
        # cover_count[m] = number of blocks covering t-subset m
        cover_count = {m: 0 for m in all_t_masks}
        uncovered = set(all_t_masks)
        
        # To speed up, we can keep a list of uncovered masks
        # But checking `if cover_count[m] == 0` is fast.
        
        while uncovered:
            # Pick a random uncovered subset to break symmetry
            # This helps in finding diverse solutions
            start_mask = random.choice(list(uncovered))
            
            # Generate all k-subsets containing start_mask
            # start_mask has t bits. We need to add k-t more points.
            # Points in start_mask:
            start_pts = []
            for i in range(v):
                if (start_mask >> i) & 1:
                    start_pts.append(i)
            
            # Available points to add:
            available = [p for p in all_points if p not in start_pts]
            
            # We need to choose k-t points from available
            # v=19, t=5, k=8 => need 3 points from 14 available.
            # C(14, 3) = 364 candidates. This is small enough.
            
            best_gain = -1
            best_block_mask = None
            best_block_pts = None
            
            for extra in itertools.combinations(available, k - t):
                full_pts = start_pts + list(extra)
                # Sort to ensure canonical form? Not strictly necessary for mask, but good for distinctness check if we used tuples.
                # We use masks for identity.
                b_mask = 0
                for p in full_pts:
                    b_mask |= (1 << p)
                
                # Calculate gain
                # Optimization: We can precompute the t-subsets of the block?
                # No, just compute gain directly.
                gain = 0
                # The block covers start_mask itself.
                # It covers C(k, t) subsets.
                # We only care about those with cover_count 0.
                
                # Fast check: 
                # We can iterate through all t-subsets of full_pts
                for sub in itertools.combinations(full_pts, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    if cover_count[m] == 0:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we found a perfect gain? 
                            # Max possible gain is number of uncovered subsets in this block.
                            # We can't easily know max without iterating.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_mask = b_mask
                    best_block_pts = full_pts
            
            if best_block_mask is None:
                # Fallback: shouldn't happen if uncovered is not empty
                break
            
            # Add the block
            blocks.append(best_block_pts)
            # Update cover counts
            for sub in itertools.combinations(best_block_pts, t):
                m = 0
                for p in sub:
                    m |= (1 << p)
                cover_count[m] += 1
                if cover_count[m] == 1:
                    uncovered.discard(m)
                    
        return blocks, cover_count

    def prune(blocks, cover_count):
        """
        Remove blocks that are not essential.
        A block is essential if it is the *only* block covering some t-subset.
        """
        changed = True
        while changed:
            changed = False
            # Iterate backwards
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                essential = False
                for sub in itertools.combinations(b, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    if cover_count[m] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    blocks.pop(i)
                    # Update cover counts
                    for sub in itertools.combinations(b, t):
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        cover_count[m] -= 1
                    changed = True
        return blocks, cover_count

    def local_search(blocks, cover_count):
        """
        Simple local search: Try to swap a block with another if it reduces the number of blocks 
        or allows further pruning.
        This is complex to implement correctly and efficiently within time limits.
        A simpler improvement: "Remove and Re-add".
        For each block B, remove it. If coverage breaks, find the best block B' to add back.
        If B' covers more unique subsets than B did (relative to the rest), or if it allows pruning of another block, keep it.
        
        Given time constraints, we will rely on multiple randomized greedy runs + pruning.
        """
        pass

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    # Run multiple times
    # C(19,8,5) is hard. Randomized greedy might find ~330-340.
    # The seed is 1.0549. If seed is ~350, 1.05 means child is ~332?
    # Or is the score seed/value?
    # "Score: the number of blocks, minimized... Score: seed/value".
    # If seed is 350, and value is 332, score = 350/332 = 1.054.
    # So we want to minimize blocks.
    
    iterations = 0
    while time.time() - start_time < time_limit:
        blocks, cover_count = run_greedy()
        blocks, cover_count = prune(blocks, cover_count)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Optional: Try a simple local search on the best found so far?
        # Let's just do more greedy runs.
        
        iterations += 1
        # If we find a very good solution, we can spend more time on it?
        # For now, keep running.

    return best_blocks
