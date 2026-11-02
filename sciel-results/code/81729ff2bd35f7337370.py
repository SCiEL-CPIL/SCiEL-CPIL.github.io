import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115  # 5s margin
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    
    # Map block to list of t-subset indices it covers
    block_t_idx = {}
    for b in all_k_blocks:
        b_set = set(b)
        idxs = []
        for s in all_t_subsets:
            if all(x in b_set for x in s):
                idxs.append(t_subset_to_idx[s])
        block_t_idx[b] = idxs
        
    # Map t-subset to list of blocks covering it
    t_to_blocks = {s: [] for s in all_t_subsets}
    for b in all_k_blocks:
        for idx in block_t_idx[b]:
            s = all_t_subsets[idx]
            t_to_blocks[s].append(b)
            
    uncovered_set = set(all_t_subsets)
    blocks = []
    coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
    
    # Greedy
    while uncovered_set and time.time() - start_time < timeout:
        s = random.choice(list(uncovered_set))
        candidates = t_to_blocks[s]
        
        best_block = None
        best_gain = -1
        
        # Optimize gain calculation
        # We can cache gains? No, they change.
        for b in candidates:
            gain = 0
            for idx in block_t_idx[b]:
                if all_t_subsets[idx] in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        for idx in block_t_idx[best_block]:
            coverage_count[idx] += 1
            if coverage_count[idx] == 1:
                uncovered_set.discard(all_t_subsets[idx])
                
    # VND
    def is_redundant(block):
        for idx in block_t_idx[block]:
            if coverage_count[idx] == 1:
                return False
        return True
        
    def try_remove():
        for i in range(len(blocks)):
            if is_redundant(blocks[i]):
                for idx in block_t_idx[blocks[i]]:
                    coverage_count[idx] -= 1
                blocks.pop(i)
                return True
        return False
        
    def try_2_for_1():
        n = len(blocks)
        for i in range(n):
            for j in range(i+1, n):
                b1 = blocks[i]
                b2 = blocks[j]
                
                points_in_u = set()
                has_u = False
                for idx in block_t_idx[b1]:
                    if coverage_count[idx] == 1:
                        has_u = True
                        points_in_u.update(all_t_subsets[idx])
                for idx in block_t_idx[b2]:
                    if coverage_count[idx] == 1:
                        has_u = True
                        points_in_u.update(all_t_subsets[idx])
                        
                if not has_u:
                    continue
                    
                if len(points_in_u) <= k:
                    remaining = set(range(v)) - points_in_u
                    if len(remaining) < k - len(points_in_u):
                        continue
                    extra = sorted(remaining)[:k - len(points_in_u)]
                    new_block = tuple(sorted(list(points_in_u) + extra))
                    
                    # Update counts
                    for idx in block_t_idx[b1]:
                        coverage_count[idx] -= 1
                    for idx in block_t_idx[b2]:
                        coverage_count[idx] -= 1
                    for idx in block_t_idx[new_block]:
                        coverage_count[idx] += 1
                        
                    # Update blocks
                    blocks.pop(i)
                    blocks.pop(j-1) # j shifts down by 1 after popping i
                    blocks.append(new_block)
                    return True
        return False

    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        while try_remove():
            improved = True
        if try_2_for_1():
            improved = True

    return blocks
