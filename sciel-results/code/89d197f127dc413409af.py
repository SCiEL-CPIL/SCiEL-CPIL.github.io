import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = list(range(v))
    tsets = list(itertools.combinations(points, t))
    tset_to_id = {ts: i for i, ts in enumerate(tsets)}
    
    blocks = list(itertools.combinations(points, k))
    
    block_tsets = []
    for b in blocks:
        ts_ids = [tset_to_id[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(ts_ids)
        
    num_tsets = len(tsets)
    num_blocks = len(blocks)
    
    def greedy_cover_indices():
        covered = [False] * num_tsets
        used_indices = []
        used_set = set()
        remaining = num_tsets
        
        while remaining > 0 and time.time() < deadline - 20: # Leave time for LS
            best_block_idx = -1
            max_gain = 0
            
            for i in range(num_blocks):
                if i in used_set:
                    continue
                gain = 0
                for tid in block_tsets[i]:
                    if not covered[tid]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            
            if max_gain == 0:
                break
                
            used_indices.append(best_block_idx)
            used_set.add(best_block_idx)
            
            for tid in block_tsets[best_block_idx]:
                if not covered[tid]:
                    covered[tid] = True
                    remaining -= 1
                    
        return used_indices

    def remove_redundant_indices(block_indices):
        if not block_indices:
            return []
        
        cover_count = [0] * num_tsets
        for b_idx in block_indices:
            for tid in block_tsets[b_idx]:
                cover_count[tid] += 1
                
        new_indices = []
        for b_idx in block_indices:
            is_redundant = True
            for tid in block_tsets[b_idx]:
                if cover_count[tid] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for tid in block_tsets[b_idx]:
                    cover_count[tid] -= 1
            else:
                new_indices.append(b_idx)
                
        return new_indices

    def local_search(block_indices):
        current_best = list(block_indices)
        best_size = len(current_best)
        
        while time.time() < deadline:
            if len(current_best) < best_size:
                best_size = len(current_best)
                current_best = list(current_best)
                
            if not current_best:
                break
                
            rem_idx = random.randint(0, len(current_best) - 1)
            rem_block_idx = current_best[rem_idx]
            
            temp_indices = current_best[:rem_idx] + current_best[rem_idx+1:]
            
            cover_count = [0] * num_tsets
            for b_idx in temp_indices:
                for tid in block_tsets[b_idx]:
                    cover_count[tid] += 1
                    
            uncovered_set = {tid for tid in range(num_tsets) if cover_count[tid] == 0}
            
            if not uncovered_set:
                current_best = remove_redundant_indices(temp_indices)
                continue
            
            used_set = set(temp_indices)
            
            while uncovered_set and time.time() < deadline:
                best_new_block = -1
                max_gain = 0
                
                for b_idx in range(num_blocks):
                    if b_idx in used_set:
                        continue
                    gain = 0
                    for tid in block_tsets[b_idx]:
                        if tid in uncovered_set:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_new_block = b_idx
                        
                if best_new_block == -1 or max_gain == 0:
                    break
                    
                temp_indices.append(best_new_block)
                used_set.add(best_new_block)
                
                for tid in block_tsets[best_new_block]:
                    if tid in uncovered_set:
                        uncovered_set.remove(tid)
                        
            temp_indices = remove_redundant_indices(temp_indices)
            
            if len(temp_indices) < len(current_best):
                current_best = temp_indices
                
        return current_best

    indices = greedy_cover_indices()
    indices = remove_redundant_indices(indices)
    indices = local_search(indices)
    
    return [blocks[i] for i in indices]
