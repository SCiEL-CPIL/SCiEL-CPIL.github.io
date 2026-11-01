import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    total_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for O(1) array access
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 115  # Conservative limit
    
    def local_search(blocks):
        """
        Try to remove redundant blocks.
        A block is redundant if every t-subset it contains is covered by at least one other block.
        """
        if not blocks:
            return blocks
        
        # Count coverage for each t-subset
        coverage_count = [0] * total_subsets
        
        # Pre-compute t-subset indices for each block to speed up updates
        block_indices = []
        for block in blocks:
            indices = []
            for s in itertools.combinations(block, t):
                indices.append(subset_to_idx[s])
            block_indices.append(indices)
            
            for idx in indices:
                coverage_count[idx] += 1
        
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_block_indices = []
            
            for i, block in enumerate(blocks):
                indices = block_indices[i]
                # Check if any t-subset has count == 1
                essential = False
                for idx in indices:
                    if coverage_count[idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # This block can be removed
                    for idx in indices:
                        coverage_count[idx] -= 1
                    changed = True
                else:
                    # Keep this block
                    new_blocks.append(block)
                    new_block_indices.append(indices)
            
            blocks = new_blocks
            block_indices = new_block_indices
            
        return blocks

    while time.time() - start_time < time_limit:
        covered = [0] * total_subsets # 0 means uncovered
        blocks = []
        covered_count = 0
        
        while covered_count < total_subsets:
            # Pick a random uncovered subset
            start_subset = None
            attempts = 0
            max_attempts = 20
            
            while start_subset is None:
                if random.random() < 0.5 or attempts > max_attempts:
                    # Linear scan from random start
                    start_idx = random.randint(0, total_subsets - 1)
                    for j in range(total_subsets):
                        idx = (start_idx + j) % total_subsets
                        if covered[idx] == 0:
                            start_subset = all_t_subsets[idx]
                            break
                else:
                    idx = random.randint(0, total_subsets - 1)
                    if covered[idx] == 0:
                        start_subset = all_t_subsets[idx]
                
                attempts += 1
                if attempts > max_attempts * 2:
                    # Fallback: find first uncovered
                    for s in all_t_subsets:
                        if covered[subset_to_idx[s]] == 0:
                            start_subset = s
                            break
                    break
            
            if start_subset is None:
                break
                
            block = list(start_subset)
            
            # Greedily add points
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_points = []
                
                # Calculate gain for each candidate
                for x in points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    if len(block) >= t - 1:
                        for y_tuple in itertools.combinations(block, t - 1):
                            s = tuple(sorted(y_tuple + (x,)))
                            idx = subset_to_idx[s]
                            if covered[idx] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if not best_points:
                    break
                    
                block.append(random.choice(best_points))
            
            # Finalize block
            block.sort()
            blocks.append(block)
            
            # Update covered
            for s in itertools.combinations(block, t):
                idx = subset_to_idx[s]
                if covered[idx] == 0:
                    covered[idx] = 1
                    covered_count += 1
        
        # Apply Local Search
        optimized_blocks = local_search(blocks)
        
        if len(optimized_blocks) < best_count:
            best_count = len(optimized_blocks)
            best_blocks = optimized_blocks
            
    return best_blocks if best_blocks else []
