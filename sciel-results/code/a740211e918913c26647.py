import numpy as np
import itertools
import time

def construct(v, k, t):
    # Specialized fast path for C(19, 7, 4)
    if v == 19 and k == 7 and t == 4:
        return _construct_19_7_4()
    
    # Generic fallback
    return _generic_greedy(v, k, t)

def _construct_19_7_4():
    start_time = time.time()
    deadline = start_time + 110.0
    
    v, k, t = 19, 7, 4
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_4subs = list(itertools.combinations(points, t))
    combo_to_idx = {c: i for i, c in enumerate(all_4subs)}
    n_4subs = len(all_4subs)
    
    num_4subs_per_block = 35 # C(7,4)
    block_4subs = np.zeros((n_blocks, num_4subs_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_4subs[i, j] = combo_to_idx[sub]
            
    best_blocks_found = None
    best_count = float('inf')
    
    for seed in range(500):
        if time.time() > deadline:
            break
            
        selected_indices, coverage = _run_greedy_numpy(block_4subs, n_4subs, seed)
        reduced_indices = _reduce_blocks(selected_indices, coverage, block_4subs)
        
        if len(reduced_indices) < best_count:
            best_count = len(reduced_indices)
            best_blocks_found = [list(all_blocks[b]) for b in reduced_indices]
            
            # If we found 151, it's a known best. 
            # We can continue to try for 150, but it's rare.
            # If we found 150, we stop.
            if best_count <= 150:
                break

    if best_blocks_found is None:
        return _generic_greedy(v, k, t)
        
    return best_blocks_found

def _run_greedy_numpy(block_4subs, n_4subs, seed):
    np.random.seed(seed)
    coverage = np.zeros(n_4subs, dtype=np.int32)
    selected = []
    n_blocks = block_4subs.shape[0]
    uncovered = n_4subs
    
    while uncovered > 0:
        gains = (coverage[block_4subs] == 0).sum(axis=1)
        noise = np.random.random(n_blocks) * 0.5
        scores = gains + noise
        best_idx = np.argmax(scores)
        selected.append(best_idx)
        
        subs = block_4subs[best_idx]
        newly_covered = (coverage[subs] == 0).sum()
        uncovered -= newly_covered
        coverage[subs] += 1
        
    return selected, coverage

def _reduce_blocks(selected_indices, coverage, block_4subs):
    current_blocks = list(selected_indices)
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for idx in current_blocks:
            subs = block_4subs[idx]
            if np.min(coverage[subs]) > 1:
                coverage[subs] -= 1
                changed = True
            else:
                new_blocks.append(idx)
        current_blocks = new_blocks
    return current_blocks

def _generic_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
