import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    k_subs = list(itertools.combinations(range(v), k))
    num_k_subs = len(k_subs)
    
    block_masks = [0] * num_k_subs
    subsets_to_blocks = [[] for _ in range(num_t_subs)]
    
    for b_idx, k_sub in enumerate(k_subs):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            t_idx = t_sub_to_idx[t_sub]
            mask |= (1 << t_idx)
            subsets_to_blocks[t_idx].append(b_idx)
        block_masks[b_idx] = mask
        
    # Map k_sub tuple to index
    k_sub_to_idx = {k: i for i, k in enumerate(k_subs)}

    def greedy_run(random_seed):
        random.seed(random_seed)
        uncovered_mask = (1 << num_t_subs) - 1
        blocks_used = []
        
        while uncovered_mask:
            # Get list of uncovered indices
            uncovered_indices = []
            m = uncovered_mask
            while m:
                low = m & -m
                idx = low.bit_length() - 1
                uncovered_indices.append(idx)
                m ^= low
                
            # Pick a random uncovered index
            u_idx = random.choice(uncovered_indices)
            
            # Get candidate blocks covering u_idx
            candidates = subsets_to_blocks[u_idx]
            
            max_score = -1
            best_candidates = []
            
            for b_idx in candidates:
                score = (uncovered_mask & block_masks[b_idx]).bit_count()
                if score > max_score:
                    max_score = score
                    best_candidates = [b_idx]
                elif score == max_score:
                    best_candidates.append(b_idx)
            
            chosen_block_idx = random.choice(best_candidates)
            blocks_used.append(k_subs[chosen_block_idx])
            uncovered_mask &= ~block_masks[chosen_block_idx]
            
        return blocks_used

    def remove_redundant(blocks):
        block_indices = [k_sub_to_idx[b] for b in blocks]
        
        cover_counts = [0] * num_t_subs
        for b_idx in block_indices:
            mask = block_masks[b_idx]
            m = mask
            while m:
                low = m & -m
                idx = low.bit_length() - 1
                cover_counts[idx] += 1
                m ^= low
                
        changed = True
        while changed:
            changed = False
            for i in range(len(block_indices) - 1, -1, -1):
                b_idx = block_indices[i]
                mask = block_masks[b_idx]
                
                safe = True
                m = mask
                while m:
                    low = m & -m
                    idx = low.bit_length() - 1
                    if cover_counts[idx] <= 1:
                        safe = False
                        break
                    m ^= low
                
                if safe:
                    m = mask
                    while m:
                        low = m & -m
                        idx = low.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= low
                    block_indices.pop(i)
                    changed = True
                    if time.time() > deadline:
                        return [k_subs[j] for j in block_indices]
            
            if time.time() > deadline:
                break

        return [k_subs[i] for i in block_indices]

    best_blocks = None
    best_len = float('inf')
    
    run_id = 0
    while time.time() < deadline:
        blocks = greedy_run(run_id)
        cleaned = remove_redundant(blocks)
        
        if len(cleaned) < best_len:
            best_len = len(cleaned)
            best_blocks = cleaned
            
        run_id += 1
        
    if best_blocks is None:
        blocks = greedy_run(0)
        return [list(b) for b in blocks]
        
    return [list(b) for b in best_blocks]
