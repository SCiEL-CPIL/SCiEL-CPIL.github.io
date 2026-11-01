import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets and their indices
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    idx_map = {subset: i for i, subset in enumerate(all_subsets)}
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        random.seed(seed)
        np.random.seed(seed)
        
        # Greedy construction
        covered = np.zeros(n_subsets, dtype=bool)
        uncovered_indices = list(range(n_subsets))
        blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered subset to start the block
            idx = random.choice(uncovered_indices)
            start_subset = all_subsets[idx]
            block = list(start_subset)
            
            # Extend block to size k
            while len(block) < k:
                best_gain = -1
                best_x = None
                # Candidates are points not in block
                # To speed up, we can just iterate all v points and skip those in block
                # But v=19 is small.
                
                # Pre-compute 4-subsets of current block for gain calculation
                # We need T such that x in T and T\{x} in block.
                # T\{x} is a 4-subset of block.
                # If len(block) < 4, gain is 0 for all x.
                if len(block) >= 4:
                    block_4_subsets = list(itertools.combinations(block, 4))
                else:
                    block_4_subsets = []
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    if block_4_subsets:
                        for rest in block_4_subsets:
                            # Form the 5-subset
                            # rest is a tuple of 4 points from block
                            # x is the new point
                            # We need to form a sorted tuple to lookup in idx_map
                            # Optimization: rest is sorted. x might not be.
                            # We can merge x into rest efficiently or just sort.
                            t_tuple = tuple(sorted((x,) + rest))
                            t_idx = idx_map.get(t_tuple)
                            if t_idx is not None and not covered[t_idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
                        # Tie break randomly? Or just keep first.
                        # Random tie break adds diversity
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x is None:
                    # Should not happen if uncovered is not empty
                    break
                block.append(best_x)
            
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update covered status
            # Find all 5-subsets of the new block
            new_subsets = list(itertools.combinations(block, t))
            newly_covered = []
            for sub in new_subsets:
                sub_idx = idx_map[sub]
                if not covered[sub_idx]:
                    covered[sub_idx] = True
                    newly_covered.append(sub_idx)
            
            # Remove newly covered from uncovered_indices
            if newly_covered:
                # Convert to set for fast removal? Or just filter.
                # Filtering is O(N). N=11628.
                # Set difference is O(N) too.
                newly_covered_set = set(newly_covered)
                uncovered_indices = [i for i in uncovered_indices if i not in newly_covered_set]
        
        # Post-processing: Remove redundant blocks
        # Count coverage for each subset
        counts = np.zeros(n_subsets, dtype=int)
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[idx_map[sub]] += 1
        
        # Pass 1: Identify redundant blocks
        redundant = []
        for b in blocks:
            is_redundant = True
            for sub in itertools.combinations(b, t):
                if counts[idx_map[sub]] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant.append(b)
        
        if redundant:
            # Decrement counts for redundant blocks
            for b in redundant:
                for sub in itertools.combinations(b, t):
                    counts[idx_map[sub]] -= 1
            # Remove them
            blocks = [b for b in blocks if b not in redundant]
            
            # Pass 2: Check again (blocks might have become redundant due to removal)
            redundant2 = []
            for b in blocks:
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    if counts[idx_map[sub]] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant2.append(b)
            
            if redundant2:
                for b in redundant2:
                    for sub in itertools.combinations(b, t):
                        counts[idx_map[sub]] -= 1
                blocks = [b for b in blocks if b not in redundant2]

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # Early stop if we hit the lower bound or a very good score?
            # Lower bound is 264.
            if best_len <= 264:
                break

    if best_blocks is None:
        # Fallback: return a valid covering (maybe the last one generated)
        # If loop didn't finish or failed, we need to return something valid.
        # The greedy always produces a valid covering.
        # If time ran out before any completion, we might have a partial one?
        # No, the while loop inside greedy completes.
        # So best_blocks should be set after the first iteration.
        pass
        
    return best_blocks
