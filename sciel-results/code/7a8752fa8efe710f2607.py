import itertools
import time
import sys
import math

sys.setrecursionlimit(10000)

def construct(v, k, t):
    points = range(v)
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    block_masks = [0] * n_blocks
    t_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            t_to_blocks[idx].append(i)
        block_masks[i] = mask
        
    t_counts = [len(lst) for lst in t_to_blocks]
    max_gain = math.comb(k, t)

    def greedy_solve():
        uncovered = all_mask
        used = set()
        path = []
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(n_blocks):
                if i in used:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            path.append(best_idx)
            used.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        return path, len(path)

    initial_path, current_best = greedy_solve()
    best_path = initial_path[:]
    
    start_time = time.time()
    timeout = 110.0

    def lower_bound(uncovered):
        if uncovered == 0:
            return 0
        return (uncovered.bit_count() + max_gain - 1) // max_gain

    def find_mvc(uncovered):
        min_count = float('inf')
        min_idx = -1
        mask = uncovered
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            if t_counts[idx] < min_count:
                min_count = t_counts[idx]
                min_idx = idx
            mask ^= lsb
        return min_idx

    def branch_and_bound(uncovered, depth, path, used):
        nonlocal current_best, best_path, start_time, timeout
        
        if time.time() - start_time > timeout:
            return
            
        if not uncovered:
            if depth < current_best:
                current_best = depth
                best_path = path[:]
            return
            
        lb = lower_bound(uncovered)
        if depth + lb >= current_best:
            return
            
        target_idx = find_mvc(uncovered)
        if target_idx == -1:
            return
            
        candidates = t_to_blocks[target_idx]
        
        valid_candidates = []
        for b in candidates:
            if b in used:
                continue
            gain = (block_masks[b] & uncovered).bit_count()
            if gain > 0:
                valid_candidates.append((gain, b))
                
        if not valid_candidates:
            return
            
        valid_candidates.sort(reverse=True)
        
        for gain, b in valid_candidates:
            if depth + 1 >= current_best:
                break
                
            new_uncovered = uncovered & ~block_masks[b]
            new_used = used | {b}
            new_path = path + [b]
            
            branch_and_bound(new_uncovered, depth + 1, new_path, new_used)

    branch_and_bound(all_mask, 0, [], set())
    
    return [list(k_subsets[i]) for i in best_path]
