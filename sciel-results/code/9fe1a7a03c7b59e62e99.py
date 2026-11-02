import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
    
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
    def indices_to_blocks(indices):
        return [list(all_k_blocks[i]) for i in indices]
    
    def blocks_to_indices(blocks):
        return [get_block_idx(b) for b in blocks]
    
    # Greedy with configurable randomization
    def greedy_run(seed, strategy='uniform'):
        random.seed(seed)
        rng = np.random.RandomState(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol_indices = []
        
        while True:
            if time.time() > deadline - 12.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            
            if strategy == 'uniform':
                b_idx = int(candidates[rng.randint(0, len(candidates))])
            elif strategy == 'first':
                b_idx = int(candidates[0])
            elif strategy == 'last':
                b_idx = int(candidates[-1])
            elif strategy == 'weighted':
                # Weight by how many uncovered ts each candidate covers
                # (all have same gain here, so use position as tiebreaker)
                b_idx = int(candidates[rng.randint(0, len(candidates))])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates))])
            
            sol_indices.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol_indices

    # Run diverse greedy instances
    best_sol_indices = None
    best_len = float('inf')
    
    # Deterministic greedy (strategy='first')
    det_sol = greedy_run(0, strategy='first')
    if det_sol is not None:
        best_sol_indices = det_sol
        best_len = len(det_sol)
    
    # Randomized greedy with different strategies and seeds
    strategies = ['uniform', 'last', 'weighted']
    num_runs = 60
    for i in range(num_runs):
        if time.time() > deadline - 15.0:
            break
        seed = i + 1
        strat = strategies[i % len(strategies)]
        sol = greedy_run(seed, strategy=strat)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_sol_indices = sol
    
    if best_sol_indices is None:
        # Fallback: deterministic greedy
        best_sol_indices = greedy_run(0, strategy='first')
        best_len = len(best_sol_indices)
    
    best_solution = indices_to_blocks(best_sol_indices)
    
    # Redundancy removal
    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
                    
        return blocks

    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)
    
    # Local search: single-block swap (replace one block with another)
    def local_search_swap(blocks, max_passes=100):
        if not blocks:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for _ in range(max_passes):
            if time.time() > deadline - 3.0:
                break
                
            found_swap = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    break
                    
                unique_ts = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)
                        
                if not unique_ts:
                    continue
                    
                # Find candidate blocks that cover ALL unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                
                current_idx = get_block_idx(blocks[i])
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                
                if not valid_candidates:
                    continue
                    
                new_block_idx = random.choice(valid_candidates)
                new_block = list(all_k_blocks[new_block_idx])
                
                old_block = blocks[i]
                old_block_idx = get_block_idx(old_block)
                
                # Perform swap
                blocks[i] = new_block
                sol_block_indices.discard(old_block_idx)
                sol_block_indices.add(new_block_idx)
                
                for ts in itertools.combinations(old_block, t):
                    cover_count[ts_index_map[ts]] -= 1
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1
                    
                improved = True
                found_swap = True
                break
                
            if not found_swap:
                break
                
        return blocks, improved

    # 2-exchange: replace two blocks with one block (reduces count by 1)
    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        n = len(blocks)
        improved = False
        
        for i in range(n):
            if time.time() > deadline - 3.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 3.0:
                    break
                    
                # Find t-subsets uniquely covered by block i or block j
                # (i.e., only covered by i or only covered by j)
                unique_to_i = []
                unique_to_j = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_to_i.append(idx)
                for ts in itertools.combinations(blocks[j], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_to_j.append(idx)
                        
                if not unique_to_i and not unique_to_j:
                    # Both blocks are fully redundant - just remove one
                    # But we want to replace 2 with 1
                    pass
                    
                # Find a single block that covers all unique_to_i AND all unique_to_j
                all_unique = unique_to_i + unique_to_j
                if not all_unique:
                    continue
                    
                # Find candidates that cover all these unique t-subsets
                candidates = set(ts_to_blocks[all_unique[0]])
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                
                # Remove i and j from candidates
                idx_i = get_block_idx(blocks[i])
                idx_j = get_block_idx(blocks[j])
                candidates.discard(idx_i)
                candidates.discard(idx_j)
                
                # Also exclude blocks already in solution
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                
                if not valid_candidates:
                    continue
                    
                new_block_idx = random.choice(valid_candidates)
                new_block = list(all_k_blocks[new_block_idx])
                
                # Perform the 2->1 exchange
                old_block_i = blocks[i]
                old_block_j = blocks[j]
                
                # Update cover counts
                for ts in itertools.combinations(old_block_i, t):
                    cover_count[ts_index_map[ts]] -= 1
                for ts in itertools.combinations(old_block_j, t):
                    cover_count[ts_index_map[ts]] -= 1
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1
                    
                # Verify validity
                if np.min(cover_count) >= 1:
                    # Remove blocks i and j, add new block
                    if i < j:
                        blocks.pop(j)
                        blocks.pop(i)
                    else:
                        blocks.pop(i)
                        blocks.pop(j)
                    blocks.append(new_block)
                    
                    sol_block_indices.discard(idx_i)
                    sol_block_indices.discard(idx_j)
                    sol_block_indices.add(new_block_idx)
                    
                    improved = True
                    return blocks, improved
                else:
                    # Revert
                    for ts in itertools.combinations(new_block, t):
                        cover_count[ts_index_map[ts]] -= 1
                    for ts in itertools.combinations(old_block_i, t):
                        cover_count[ts_index_map[ts]] += 1
                    for ts in itertools.combinations(old_block_j, t):
                        cover_count[ts_index_map[ts]] += 1
                        
        return blocks, improved

    # Apply local search
    best_solution, improved = local_search_swap(best_solution)
    while improved and time.time() < deadline - 5.0:
        best_solution, improved = local_search_swap(best_solution)
    
    # Apply 2-exchange
    best_solution, improved = local_search_2exchange(best_solution)
    while improved and time.time() < deadline - 3.0:
        best_solution, improved = local_search_2exchange(best_solution)
    
    # Final redundancy removal
    best_solution = optimize_redundancy(best_solution)
    
    return best_solution
