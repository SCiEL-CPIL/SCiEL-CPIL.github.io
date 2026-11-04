import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets and their t-subset indices
    all_blocks = []
    block_t_indices = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        t_inds = []
        for ts in itertools.combinations(block, t):
            t_inds.append(t_subset_indices[ts])
        block_t_indices.append(t_inds)
        
    num_blocks_total = len(all_blocks)
    
    # Map from t-subset index to list of block indices containing it
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, t_inds in enumerate(block_t_indices):
        for t_idx in t_inds:
            t_to_blocks[t_idx].append(b_idx)
            
    def greedy_cover():
        covered = [0] * num_t
        blocks = []
        uncovered_count = num_t
        
        while uncovered_count > 0:
            # Find first uncovered t-subset
            target_t_idx = -1
            for i in range(num_t):
                if covered[i] == 0:
                    target_t_idx = i
                    break
            
            if target_t_idx == -1:
                break
                
            candidates = t_to_blocks[target_t_idx]
            best_gain = 0
            best_b_idx = -1
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if covered[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            if best_b_idx == -1:
                break
                
            blocks.append(best_b_idx)
            for t_idx in block_t_indices[best_b_idx]:
                if covered[t_idx] == 0:
                    covered[t_idx] = 1
                    uncovered_count -= 1
            
        return blocks

    def optimize(blocks):
        cov_count = [0] * num_t
        for b_idx in blocks:
            for t_idx in block_t_indices[b_idx]:
                cov_count[t_idx] += 1
                
        improved = True
        while improved:
            improved = False
            
            # 1. Try to remove any block
            for i in range(len(blocks)):
                b_idx = blocks[i]
                removable = True
                for t_idx in block_t_indices[b_idx]:
                    if cov_count[t_idx] <= 1:
                        removable = False
                        break
                if removable:
                    for t_idx in block_t_indices[b_idx]:
                        cov_count[t_idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try Swaps
            found_swap = False
            for i in range(len(blocks)):
                b_i = blocks[i]
                for t_idx in block_t_indices[b_i]:
                    cov_count[t_idx] -= 1
                
                uncovered = [t_idx for t_idx in range(num_t) if cov_count[t_idx] == 0]
                
                if not uncovered:
                    for t_idx in block_t_indices[b_i]:
                        cov_count[t_idx] += 1
                    continue
                
                # Find candidate blocks covering all uncovered
                cand_set = set(t_to_blocks[uncovered[0]])
                for t_idx in uncovered[1:]:
                    cand_set.intersection_update(t_to_blocks[t_idx])
                    
                for b_j in cand_set:
                    if b_j == b_i:
                        continue
                    if b_j in blocks:
                        continue
                        
                    for t_idx in block_t_indices[b_j]:
                        cov_count[t_idx] += 1
                        
                    removable_k = -1
                    for k in range(len(blocks)):
                        if k == i:
                            continue
                        b_k = blocks[k]
                        is_removable = True
                        for t_idx in block_t_indices[b_k]:
                            if cov_count[t_idx] <= 1:
                                is_removable = False
                                break
                        if is_removable:
                            removable_k = k
                            break
                            
                    if removable_k != -1:
                        for t_idx in block_t_indices[blocks[removable_k]]:
                            cov_count[t_idx] -= 1
                            
                        blocks.pop(i)
                        if removable_k > i:
                            removable_k -= 1
                        blocks.pop(removable_k)
                        blocks.append(b_j)
                        
                        improved = True
                        found_swap = True
                        break
                    else:
                        for t_idx in block_t_indices[b_j]:
                            cov_count[t_idx] -= 1
                
                for t_idx in block_t_indices[b_i]:
                    cov_count[t_idx] += 1
                
                if found_swap:
                    break
            
            if found_swap:
                continue
            
            # 3. Perturbation
            if len(blocks) > 0:
                i = random.randint(0, len(blocks)-1)
                b_i = blocks[i]
                
                for t_idx in block_t_indices[b_i]:
                    cov_count[t_idx] -= 1
                blocks.pop(i)
                
                uncovered = [t_idx for t_idx in range(num_t) if cov_count[t_idx] == 0]
                
                while uncovered:
                    t_idx = random.choice(uncovered)
                    candidates = t_to_blocks[t_idx]
                    if not candidates:
                        break
                    b_new = random.choice(candidates)
                    
                    blocks.append(b_new)
                    for t_idx2 in block_t_indices[b_new]:
                        if cov_count[t_idx2] == 0:
                            cov_count[t_idx2] = 1
                        else:
                            cov_count[t_idx2] += 1
                            
                    uncovered = [t_idx2 for t_idx2 in range(num_t) if cov_count[t_idx2] == 0]
                    
                improved = True

        return blocks

    start_time = time.time()
    best_blocks = greedy_cover()
    best_len = len(best_blocks)
    
    while time.time() - start_time < 115:
        current_blocks = optimize(best_blocks)
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks
            if best_len <= 69:
                break
                
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
