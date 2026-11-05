import itertools
import random
import time

def construct(v, k, t):
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute combinations for gain calculation?
    # For a given set S of size m, and point p not in S:
    # Subsets of S U {p} of size t are:
    # 1. C(S, t) if m >= t
    # 2. {p} U C(S, t-1)
    
    while time.time() - start_time < time_limit:
        cover_count = [0] * n_subsets
        blocks = []
        
        # Greedy Construction
        while True:
            # Find uncovered
            uncovered = [i for i in range(n_subsets) if cover_count[i] == 0]
            if not uncovered:
                break
            
            start_idx = random.choice(uncovered)
            start_subset = all_subsets[start_idx]
            current_block = list(start_subset)
            
            while len(current_block) < k:
                best_gain = -1
                best_points = []
                
                # Candidates
                for p in range(v):
                    if p in current_block:
                        continue
                    
                    gain = 0
                    # Subsets containing p
                    # t-1 elements from current_block
                    for rest in itertools.combinations(current_block, t - 1):
                        sub = tuple(sorted(rest + (p,)))
                        idx = subset_to_idx[sub]
                        if cover_count[idx] == 0:
                            gain += 1
                            
                    # Subsets not containing p
                    if len(current_block) >= t:
                        for sub in itertools.combinations(current_block, t):
                            idx = subset_to_idx[sub]
                            if cover_count[idx] == 0:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [p]
                    elif gain == best_gain:
                        best_points.append(p)
                        
                if not best_points:
                    break
                p = random.choice(best_points)
                current_block.append(p)
                
            # Add block
            block_tuple = tuple(sorted(current_block))
            blocks.append(frozenset(block_tuple))
            for sub in itertools.combinations(block_tuple, t):
                cover_count[subset_to_idx[sub]] += 1
                
        # Local Search: Removal
        improved = True
        while improved:
            improved = False
            # Shuffle to break symmetry?
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                block = blocks[i]
                can_remove = True
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove
                    blocks.pop(i)
                    for sub in itertools.combinations(block, t):
                        cover_count[subset_to_idx[sub]] -= 1
                    improved = True
                    
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    return best_blocks
