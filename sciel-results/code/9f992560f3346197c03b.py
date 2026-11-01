import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # Precompute all k-subsets and their t-subset coverage as bitmasks
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Map each t-subset to an index
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # For each block, store the bitmask of t-subsets it covers
    # Also store the list of t-indices for quick iteration
    block_t_mask = [0] * num_blocks
    block_t_list = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        t_list = []
        for t_sub in itertools.combinations(block, t):
            idx = t_to_idx[t_sub]
            mask |= (1 << idx)
            t_list.append(idx)
        block_t_mask[b_idx] = mask
        block_t_list[b_idx] = t_list
    
    # For each t-subset, store which blocks cover it
    t_blocks = [[] for _ in range(num_t)]
    for b_idx in range(num_blocks):
        for t_idx in block_t_list[b_idx]:
            t_blocks[t_idx].append(b_idx)
    
    # Degree of each t-subset
    t_degree = [len(lst) for lst in t_blocks]
    
    # Greedy set cover with weighting
    def greedy_cover():
        uncovered_mask = (1 << num_t) - 1
        active = set()
        
        while uncovered_mask and time_remaining() > 5.0:
            best_b = -1
            best_score = -1.0
            
            for b_idx in range(num_blocks):
                if b_idx in active:
                    continue
                # Count uncovered t-subsets covered by this block
                new_cover = block_t_mask[b_idx] & uncovered_mask
                if new_cover == 0:
                    continue
                
                # Score: number of new covers + scarcity bonus
                count = bin(new_cover).count('1')
                scarcity = 0.0
                for t_idx in block_t_list[b_idx]:
                    if (uncovered_mask >> t_idx) & 1:
                        scarcity += 1.0 / t_degree[t_idx]
                
                score = count * 1000.0 + scarcity
                if score > best_score:
                    best_score = score
                    best_b = b_idx
            
            if best_b == -1:
                break
            
            active.add(best_b)
            uncovered_mask &= ~block_t_mask[best_b]
        
        return active, uncovered_mask
    
    # Minimize: remove redundant blocks
    def minimize(active_set):
        if not active_set:
            return active_set
        
        cover_count = [0] * num_t
        for b_idx in active_set:
            for t_idx in block_t_list[b_idx]:
                cover_count[t_idx] += 1
        
        changed = True
        while changed and time_remaining() > 5.0:
            changed = False
            items = list(active_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                can_remove = True
                for t_idx in block_t_list[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_set.remove(b_idx)
                    for t_idx in block_t_list[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return active_set
    
    # Local search: try remove and swap moves
    def local_search(active_set):
        if not active_set:
            return active_set
        
        cover_count = [0] * num_t
        for b_idx in active_set:
            for t_idx in block_t_list[b_idx]:
                cover_count[t_idx] += 1
        
        improved = True
        while improved and time_remaining() > 5.0:
            improved = False
            items = list(active_set)
            random.shuffle(items)
            
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                
                # Try remove
                can_remove = True
                for t_idx in block_t_list[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_set.remove(b_idx)
                    for t_idx in block_t_list[b_idx]:
                        cover_count[t_idx] -= 1
                    improved = True
                    continue
                
                # Try swap: find a block that covers all uniquely-covered t-subsets of b_idx
                unique_ts = [t_idx for t_idx in block_t_list[b_idx] if cover_count[t_idx] == 1]
                
                if unique_ts:
                    # Find candidate blocks that cover all unique_ts
                    candidates = None
                    for t_idx in unique_ts:
                        if candidates is None:
                            candidates = set(t_blocks[t_idx])
                        else:
                            candidates.intersection_update(t_blocks[t_idx])
                        if not candidates:
                            break
                    
                    if candidates:
                        valid_cands = [c for c in candidates if c != b_idx and c not in active_set]
                        if valid_cands:
                            new_b = random.choice(valid_cands)
                            # Perform swap
                            active_set.remove(b_idx)
                            for t_idx in block_t_list[b_idx]:
                                cover_count[t_idx] -= 1
                            active_set.add(new_b)
                            for t_idx in block_t_list[new_b]:
                                cover_count[t_idx] += 1
                            improved = True
                            # Re-minimize after swap
                            active_set = minimize(active_set)
                            # Rebuild cover_count
                            cover_count = [0] * num_t
                            for b_idx2 in active_set:
                                for t_idx in block_t_list[b_idx2]:
                                    cover_count[t_idx] += 1
        
        return active_set
    
    # Run multiple restarts
    best_result = None
    best_size = float('inf')
    
    num_restarts = 10  # Will be limited by time
    
    for restart in range(num_restarts):
        if time_remaining() < 10.0:
            break
        
        random.seed(restart * 12345 + 67890)
        
        # Greedy initialization
        active, uncovered = greedy_cover()
        
        if uncovered != 0:
            # Fallback: add blocks to cover remaining
            while uncovered and time_remaining() > 5.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(num_blocks):
                    if b_idx in active:
                        continue
                    gain = bin(block_t_mask[b_idx] & uncovered).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                active.add(best_b)
                uncovered &= ~block_t_mask[best_b]
        
        if uncovered != 0:
            continue  # Invalid
        
        # Minimize
        active = minimize(active)
        
        # Local search
        active = local_search(active)
        
        if len(active) < best_size:
            best_size = len(active)
            best_result = active
        
        if time_remaining() < 15.0:
            break
    
    if best_result is None:
        # Last resort: simple greedy
        active, _ = greedy_cover()
        best_result = active
    
    # Final minimize
    best_result = minimize(best_result)
    
    # Convert to list of blocks
    result = [list(all_k_subsets[b_idx]) for b_idx in best_result]
    return result
