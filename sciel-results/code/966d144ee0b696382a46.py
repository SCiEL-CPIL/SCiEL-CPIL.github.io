import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    block_covers = [None] * num_blocks
    block_masks = [0] * num_blocks
    
    for b, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[b] = mask
        block_covers[b] = indices

    def greedy_local_search(seed):
        rng = random.Random(seed)
        order = list(range(num_blocks))
        rng.shuffle(order)
        
        uncovered_mask = (1 << num_t_subsets) - 1
        selected_blocks = []
        used_blocks = [False] * num_blocks
        
        while uncovered_mask:
            best_block = -1
            max_gain = 0
            
            for b in order:
                if used_blocks[b]:
                    continue
                gain = (uncovered_mask & block_masks[b]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
            
            if best_block == -1:
                break
                
            selected_blocks.append(best_block)
            used_blocks[best_block] = True
            uncovered_mask &= ~block_masks[best_block]
            
        coverage_count = [0] * num_t_subsets
        for b in selected_blocks:
            for idx in block_covers[b]:
                coverage_count[idx] += 1
                
        improved = True
        while improved:
            if time.time() > deadline:
                return selected_blocks
            
            improved = False
            for i in range(len(selected_blocks)):
                if time.time() > deadline:
                    return selected_blocks
                    
                b = selected_blocks[i]
                
                essential = []
                for idx in block_covers[b]:
                    coverage_count[idx] -= 1
                    if coverage_count[idx] == 0:
                        essential.append(idx)
                        
                if not essential:
                    selected_blocks.pop(i)
                    improved = True
                    break
                else:
                    for idx in block_covers[b]:
                        coverage_count[idx] += 1
                    
                    essential_mask = 0
                    for idx in essential:
                        essential_mask |= (1 << idx)
                    
                    found = False
                    for b_new in order:
                        if used_blocks[b_new]:
                            continue
                        if (essential_mask & ~block_masks[b_new]) == 0:
                            used_blocks[b] = False
                            selected_blocks[i] = b_new
                            used_blocks[b_new] = True
                            
                            for idx in block_covers[b]:
                                coverage_count[idx] -= 1
                            for idx in block_covers[b_new]:
                                coverage_count[idx] += 1
                                
                            improved = True
                            found = True
                            break
                            
                    if found:
                        break

        return selected_blocks

    best_blocks = None
    best_count = float('inf')
    
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
        res = greedy_local_search(i)
        if len(res) < best_count:
            best_count = len(res)
            best_blocks = res
        if best_count <= 151:
            break

    if best_blocks is None:
        return []
        
    return [list(all_blocks[b]) for b in best_blocks]
