import itertools
import random
import time

def construct(v, k, t):
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(all_k_blocks)
    
    block_cover_ids = [[] for _ in range(num_k_blocks)]
    block_cover_mask = [0] * num_k_blocks
    
    # Precompute coverage for each block
    for bid, b in enumerate(all_k_blocks):
        # Generate all t-subsets of this block
        for s in itertools.combinations(b, t):
            tid = t_subset_to_id[s]
            block_cover_ids[bid].append(tid)
            block_cover_mask[bid] |= (1 << tid)
            
    full_mask = (1 << num_t_subsets) - 1
    
    def greedy_once(seed_val):
        random.seed(seed_val)
        uncovered_mask = full_mask
        blocks_used = []
        cover_counts = [0] * num_t_subsets
        
        while uncovered_mask:
            best_gain = -1
            best_candidates = []
            
            for b_idx in range(num_k_blocks):
                common = block_cover_mask[b_idx] & uncovered_mask
                if common:
                    gain = common.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [b_idx]
                    elif gain == best_gain:
                        best_candidates.append(b_idx)
            
            if not best_candidates:
                break
            
            chosen_idx = random.choice(best_candidates)
            blocks_used.append(chosen_idx)
            
            uncovered_mask &= ~block_cover_mask[chosen_idx]
            
            for idx in block_cover_ids[chosen_idx]:
                cover_counts[idx] += 1
        
        # Local Search: Delete redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_used) - 1, -1, -1):
                b_idx = blocks_used[i]
                is_redundant = True
                for ts_idx in block_cover_ids[b_idx]:
                    if cover_counts[ts_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for ts_idx in block_cover_ids[b_idx]:
                        cover_counts[ts_idx] -= 1
                    blocks_used.pop(i)
                    changed = True
                    break
        
        return [list(all_k_blocks[i]) for i in blocks_used]

    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    
    num_runs = 10
    for i in range(num_runs):
        blocks = greedy_once(i)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs
    
    remaining_time = 115 - elapsed
    if remaining_time > 0:
        additional_runs = int(remaining_time / avg_time)
        for i in range(num_runs, num_runs + additional_runs):
            blocks = greedy_once(i)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                
    return best_blocks
