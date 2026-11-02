import itertools
import time
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a randomized weighted greedy 
    algorithm followed by redundancy removal.
    """
    points = range(v)
    all_t_subsets = set(itertools.combinations(points, t))
    
    # Pre-compute combinations of sizes t-1 for gain calculation efficiency?
    # Actually, generating them on the fly is fine for small k.
    
    def greedy_run(seed):
        random.seed(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            block = []
            # We build the block greedily
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                # Calculate gain for each candidate point
                for x in points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # If we have enough points to form a t-subset with x
                    if len(block) >= t - 1:
                        # Iterate over all (t-1)-subsets of the current block
                        # and check if adding x creates an uncovered t-subset
                        for rest in itertools.combinations(block, t - 1):
                            # rest is a tuple
                            # Create the new t-subset
                            # We need to merge rest and x, then sort
                            # Since rest is sorted and x is int, we can do this efficiently
                            # But for t=4, t-1=3, it's small.
                            new_sub = tuple(sorted(rest + (x,)))
                            if new_sub in uncovered:
                                gain += 1
                    
                    # Add small random noise to break ties and explore different paths
                    # The noise should be small enough not to override significant gain differences
                    # but large enough to randomize ties
                    noise = random.uniform(0, 0.1)
                    total_score = gain + noise
                    
                    if total_score > best_gain:
                        best_gain = total_score
                        best_candidates = [x]
                    elif total_score == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    break # Should not happen if uncovered is not empty and k < v
                    
                # Pick randomly from best candidates
                next_point = random.choice(best_candidates)
                block.append(next_point)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                # Remove covered subsets
                # Optimization: Only remove if not already removed? 
                # 'uncovered' is a set, so difference_update is fast.
                uncovered.difference_update(itertools.combinations(block, t))
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Map each t-subset to the list of block indices that cover it
        # Or just count
        cover_count = {s: 0 for s in all_t_subsets}
        
        for b in blocks:
            for sub in itertools.combinations(b, t):
                cover_count[sub] += 1
        
        # Identify redundant blocks
        # A block is redundant if all its t-subsets are covered by at least one other block
        # i.e., cover_count[sub] > 1 for all sub in block
        
        to_remove = set()
        for i, b in enumerate(blocks):
            is_redundant = True
            for sub in itertools.combinations(b, t):
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.add(i)
        
        if not to_remove:
            return blocks
        
        new_blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
        
        # After removal, some previously non-redundant blocks might become redundant
        # because the cover counts for their subsets have decreased.
        # So we loop.
        if len(new_blocks) < len(blocks):
            return remove_redundant(new_blocks)
            
        return new_blocks

    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_blocks = None
    best_len = float('inf')
    seed = 42
    
    # Run multiple iterations
    while time.time() < deadline:
        blocks = greedy_run(seed)
        # Post-process
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we've reached the theoretical lower bound or a very good known bound, we can stop early?
        # Lower bound is 111. If we hit 111, we are done.
        if best_len <= 111:
            break
            
        seed += 1
        
    return best_blocks if best_blocks is not None else []
