import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    points = range(v)
    all_t_subs = list(itertools.combinations(points, t))
    n_subs = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_k_blocks)

    # Precompute coverage masks and lists
    cov_masks = [0] * n_blocks
    cov_lists = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        lst = []
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            mask |= (1 << idx)
            lst.append(idx)
        cov_masks[i] = mask
        cov_lists[i] = lst

    best_blocks = None
    best_count = n_blocks + 1
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # 1. Greedy Construction
        uncovered_mask = (1 << n_subs) - 1
        cover_count = [0] * n_subs
        selected = set()
        selected_list = []
        
        while uncovered_mask != 0:
            best_gain = -1
            best_candidates = []
            
            for i in range(n_blocks):
                if i in selected:
                    continue
                gain = (cov_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
                
            chosen = rng.choice(best_candidates)
            selected.add(chosen)
            selected_list.append(chosen)
            
            uncovered_mask &= ~cov_masks[chosen]
            
            for idx in cov_lists[chosen]:
                cover_count[idx] += 1
        
        # 2. Deletion Phase
        i = 0
        while i < len(selected_list):
            b_idx = selected_list[i]
            is_essential = False
            for idx in cov_lists[b_idx]:
                if cover_count[idx] == 1:
                    is_essential = True
                    break
            
            if not is_essential:
                selected.remove(b_idx)
                for idx in cov_lists[b_idx]:
                    cover_count[idx] -= 1
                selected_list[i] = selected_list[-1]
                selected_list.pop()
            else:
                i += 1
        
        # 3. Update Best
        if len(selected_list) < best_count:
            best_count = len(selected_list)
            best_blocks = [list(all_k_blocks[i]) for i in selected_list]

    if best_blocks is None:
        best_blocks = []
        
    return best_blocks
