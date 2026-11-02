import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    best_blocks = None
    best_size = float('inf')
    
    # We will run multiple randomized greedy restarts
    # Each restart:
    # 1. Greedy construction with random anchor selection
    # 2. Redundancy removal
    
    max_restarts = 50 # Adjust based on performance
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Use a deterministic seed based on restart index for reproducibility
        # but varied enough to explore different paths
        random.seed(restart * 7919 + 13)
        
        uncovered_indices = set(range(num_t_subsets))
        blocks = []
        
        while uncovered_indices:
            if time.time() > deadline:
                break
                
            # Pick a random uncovered t-subset as anchor
            anchor_subset = tuple(random.choice(list(uncovered_indices)))
            
            # Generate candidate blocks containing this anchor
            # Anchor has t points. Need k-t more from v-t remaining.
            remaining_points = [p for p in points if p not in anchor_subset]
            
            candidates = []
            # Pre-compute the t-subsets of the anchor? No, anchor IS a t-subset.
            # We need to iterate over combinations of remaining points.
            
            # Optimization: Pre-calculate which t-subsets are in uncovered_indices?
            # We have a set of indices.
            
            for extra_points in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor_subset + extra_points))
                
                # Calculate gain
                gain = 0
                for t_sub in itertools.combinations(block, t):
                    idx = t_subset_index[t_sub]
                    if idx in uncovered_indices:
                        gain += 1
                
                if gain > 0:
                    candidates.append((gain, block))
            
            if not candidates:
                # This shouldn't happen if uncovered_indices is not empty
                # because the anchor itself is in some block
                break
            
            # Sort by gain
            candidates.sort(key=lambda x: -x[0])
            
            # Randomized selection from top candidates
            # Pick from top 5 with probability
            top_k = min(5, len(candidates))
            # Weighted choice: 1st has weight 5, 2nd 4, etc.
            weights = [top_k - i for i in range(top_k)]
            chosen_block = random.choices([c[1] for c in candidates[:top_k]], weights=weights)[0]
            
            blocks.append(chosen_block)
            
            # Update uncovered
            for t_sub in itertools.combinations(chosen_block, t):
                idx = t_subset_index[t_sub]
                if idx in uncovered_indices:
                    uncovered_indices.remove(idx)
        
        if uncovered_indices:
            continue # Incomplete
        
        # Redundancy Removal
        # Count coverage
        counts = [0] * num_t_subsets
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                idx = t_subset_index[t_sub]
                counts[idx] += 1
        
        # Remove redundant blocks
        i = 0
        while i < len(blocks):
            b = blocks[i]
            redundant = True
            for t_sub in itertools.combinations(b, t):
                idx = t_subset_index[t_sub]
                if counts[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                for t_sub in itertools.combinations(b, t):
                    idx = t_subset_index[t_sub]
                    counts[idx] -= 1
                blocks.pop(i)
                # Do not increment i
            else:
                i += 1
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    if best_blocks is None:
        # Fallback: Simple greedy without randomization
        # Just in case all randomized runs failed
        uncovered_indices = set(range(num_t_subsets))
        blocks = []
        while uncovered_indices:
            anchor_subset = tuple(min(uncovered_indices))
            remaining_points = [p for p in points if p not in anchor_subset]
            candidates = []
            for extra_points in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor_subset + extra_points))
                gain = sum(1 for t_sub in itertools.combinations(block, t) if t_subset_index[t_sub] in uncovered_indices)
                if gain > 0:
                    candidates.append((gain, block))
            if not candidates: break
            candidates.sort(key=lambda x: -x[0])
            best_block = candidates[0][1]
            blocks.append(best_block)
            for t_sub in itertools.combinations(best_block, t):
                idx = t_subset_index[t_sub]
                if idx in uncovered_indices:
                    uncovered_indices.remove(idx)
        
        # Redundancy removal
        counts = [0] * num_t_subsets
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                idx = t_subset_index[t_sub]
                counts[idx] += 1
        
        i = 0
        while i < len(blocks):
            b = blocks[i]
            redundant = True
            for t_sub in itertools.combinations(b, t):
                idx = t_subset_index[t_sub]
                if counts[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for t_sub in itertools.combinations(b, t):
                    idx = t_subset_index[t_sub]
                    counts[idx] -= 1
                blocks.pop(i)
            else:
                i += 1
        best_blocks = [list(b) for b in blocks]

    return best_blocks
