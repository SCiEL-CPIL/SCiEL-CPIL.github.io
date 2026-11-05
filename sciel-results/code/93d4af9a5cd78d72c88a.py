import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    mask_to_idx = {}
    points_by_idx = [None] * n_subsets
    for i, sub in enumerate(all_t_subsets):
        m = 0
        for p in sub:
            m |= (1 << p)
        mask_to_idx[m] = i
        points_by_idx[i] = sub
        
    # Precompute masks for (t-1)-subsets
    mask_cache = {}
    if t - 1 > 0:
        for sub in itertools.combinations(range(v), t - 1):
            m = 0
            for p in sub:
                m |= (1 << p)
            mask_cache[sub] = m
    else:
        # t=1 case? Not relevant for t=5, but safe.
        mask_cache[()] = 0

    def run_greedy(randomize_start=False, seed=42):
        if randomize_start:
            random.seed(seed)
            
        is_covered = [False] * n_subsets
        uncovered_list = list(range(n_subsets))
        blocks = []
        
        while uncovered_list:
            if time.time() > deadline:
                return None # Timeout, discard this run
                
            # Pick a target
            if randomize_start:
                idx_pos = random.randint(0, len(uncovered_list) - 1)
            else:
                idx_pos = 0
                
            target_idx = uncovered_list[idx_pos]
            
            # Check if it's still uncovered (might have been covered by a previous block in this run)
            if is_covered[target_idx]:
                # Remove from list and continue
                last_idx = uncovered_list[-1]
                uncovered_list[idx_pos] = last_idx
                uncovered_list.pop()
                continue
            
            # Remove target from list
            last_idx = uncovered_list[-1]
            uncovered_list[idx_pos] = last_idx
            uncovered_list.pop()
            
            block_points = list(points_by_idx[target_idx])
            
            while len(block_points) < k:
                best_p = -1
                max_gain = -1
                
                for p in range(v):
                    if p in block_points:
                        continue
                    
                    gain = 0
                    # Iterate (t-1)-subsets of current block
                    for comb in itertools.combinations(block_points, t - 1):
                        m_comb = mask_cache[comb]
                        m_full = m_comb | (1 << p)
                        idx = mask_to_idx.get(m_full)
                        if idx is not None and not is_covered[idx]:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_p = p
                        
                if best_p == -1:
                    # Fallback: pick any valid point
                    for p in range(v):
                        if p not in block_points:
                            best_p = p
                            break
                block_points.append(best_p)
            
            # Mark covered
            for sub in itertools.combinations(block_points, t):
                m = 0
                for p in sub:
                    m |= (1 << p)
                idx = mask_to_idx.get(m)
                if idx is not None and not is_covered[idx]:
                    is_covered[idx] = True
                    # Note: We do NOT remove idx from uncovered_list here.
                    # It will be filtered out when selected.
            
            blocks.append(sorted(block_points))
            
        return blocks

    best_blocks = None
    
    # Deterministic run
    res = run_greedy(randomize_start=False)
    if res:
        best_blocks = res
        
    # Random restarts
    # Allow up to 15 restarts or until time limit
    for i in range(15):
        if time.time() > deadline - 5:
            break
        res = run_greedy(randomize_start=True, seed=i)
        if res:
            if best_blocks is None or len(res) < len(best_blocks):
                best_blocks = res
                
    if best_blocks is None:
        return []
        
    return best_blocks
