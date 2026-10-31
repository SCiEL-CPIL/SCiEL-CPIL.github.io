import numpy as np
import time
import itertools
import random

def construct(v, k, t):
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute: for each point x, the numpy array of t-subset indices containing x
    point_to_subsets = [np.array([], dtype=np.int32) for _ in range(v)]
    for idx, s in enumerate(all_t_subsets):
        for x in s:
            point_to_subsets[x] = np.append(point_to_subsets[x], idx)
    
    # Precompute: for each (t-1)-subset (as frozenset), map to list of (completion_point, t_subset_idx)
    t1_map = {}
    for idx, s in enumerate(all_t_subsets):
        for j in range(t):
            t1 = frozenset(s[:j] + s[j+1:])
            pt = s[j]
            if t1 not in t1_map:
                t1_map[t1] = []
            t1_map[t1].append((pt, idx))
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(uncovered_mask):
        current_blocks = []
        uncovered = uncovered_mask.copy()
        
        uncovered_indices = np.where(uncovered)[0]
        random.shuffle(uncovered_indices)
        
        for idx_s in uncovered_indices:
            if not uncovered[idx_s]:
                continue
            
            s = all_t_subsets[idx_s]
            block = list(s)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                t1_subs = [frozenset(c) for c in itertools.combinations(block, t-1)]
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for t1 in t1_subs:
                        if t1 in t1_map:
                            for (pt, idx) in t1_map[t1]:
                                if pt == x and uncovered[idx]:
                                    gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    break
                block.append(best_point)
            
            block.sort()
            current_blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
            
            if not uncovered.any():
                break
        
        return current_blocks
    
    def local_search(blocks, time_limit):
        blocks = [b[:] for b in blocks]
        search_start = time.time()
        
        coverage_count = np.zeros(num_subsets, dtype=np.int32)
        block_subsets = []
        
        for block in blocks:
            subs = []
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                coverage_count[idx] += 1
                subs.append(idx)
            block_subsets.append(subs)
        
        improved = True
        while improved and (time.time() - search_start) < time_limit:
            improved = False
            
            for i in range(len(blocks) - 1, -1, -1):
                if (time.time() - search_start) > time_limit:
                    break
                redundant = True
                for idx in block_subsets[i]:
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_subsets[i]:
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    block_subsets.pop(i)
                    improved = True
            
            if (time.time() - search_start) > time_limit:
                break
            
            for i in range(len(blocks)):
                if (time.time() - search_start) > time_limit:
                    break
                unique_count = sum(1 for idx in block_subsets[i] if coverage_count[idx] == 1)
                if unique_count == 0:
                    continue
                
                for _ in range(3):
                    if (time.time() - search_start) > time_limit:
                        break
                    new_block = random.sample(range(v), k)
                    new_block.sort()
                    new_subsets = [subset_to_idx[sub] for sub in itertools.combinations(new_block, t)]
                    new_set = set(new_subsets)
                    old_set = set(block_subsets[i])
                    lost = sum(1 for idx in old_set if coverage_count[idx] == 1 and idx not in new_set)
                    if lost == 0:
                        for idx in block_subsets[i]:
                            coverage_count[idx] -= 1
                        for idx in new_subsets:
                            coverage_count[idx] += 1
                        blocks[i] = new_block
                        block_subsets[i] = new_subsets
                        improved = True
                        break
        
        return blocks
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = greedy_construct(uncovered)
        
        remaining = deadline - time.time()
        if remaining > 5:
            ls_time = min(remaining * 0.3, 20.0)
            current_blocks = local_search(current_blocks, ls_time)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [b[:] for b in current_blocks]
            if best_count <= 131:
                break
    
    if best_blocks is None:
        uncovered = np.ones(num_subsets, dtype=bool)
        best_blocks = greedy_construct(uncovered)
    
    return best_blocks
