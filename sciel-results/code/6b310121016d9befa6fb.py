import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute gain_indices[x][rest_tuple] = index of t-subset (rest + x)
    gain_indices = [{} for _ in range(v)]
    for x in range(v):
        others = [p for p in range(v) if p != x]
        for rest in itertools.combinations(others, t - 1):
            full_subset = tuple(sorted(rest + (x,)))
            gain_indices[x][rest] = subset_to_idx[full_subset]

    def get_gain(block, x, uncovered):
        gain = 0
        for rest in itertools.combinations(block, t - 1):
            idx = gain_indices[x][rest]
            if (uncovered >> idx) & 1:
                gain += 1
        return gain

    def greedy_cover(seed_val):
        random.seed(seed_val)
        uncovered = (1 << n_t_subsets) - 1
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return blocks
            
            # Find first uncovered subset
            first_uncovered_idx = -1
            # Optimization: maintain a list of uncovered? 
            # For now, linear scan is O(N) = 11628. 
            # 300 blocks * 11628 = 3.5M ops. Acceptable.
            for i in range(n_t_subsets):
                if (uncovered >> i) & 1:
                    first_uncovered_idx = i
                    break
            if first_uncovered_idx == -1:
                break
            
            start_subset = all_t_subsets[first_uncovered_idx]
            start_point = random.choice(start_subset)
            block = [start_point]
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in range(v) if x not in block]
                for x in candidates:
                    g = get_gain(block, x, uncovered)
                    if g > best_gain:
                        best_gain = g
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for s in itertools.combinations(block, t):
                idx = subset_to_idx[s]
                uncovered &= ~(1 << idx)
                
        return blocks

    def remove_redundant(blocks):
        counts = [0] * n_t_subsets
        for block in blocks:
            for s in itertools.combinations(block, t):
                counts[subset_to_idx[s]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks
                block = blocks[i]
                is_redundant = True
                for s in itertools.combinations(block, t):
                    idx = subset_to_idx[s]
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for s in itertools.combinations(block, t):
                        counts[subset_to_idx[s]] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 50
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 264: 
            break

    if best_blocks is None:
        best_blocks = greedy_cover(0)
        best_blocks = remove_redundant(best_blocks)

    return best_blocks
