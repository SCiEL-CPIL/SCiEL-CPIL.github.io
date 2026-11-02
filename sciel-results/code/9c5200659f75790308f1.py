import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin well before 120s

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
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures
    # block_to_ts[j]: list of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    # Map block tuple to index for O(1) lookup later
    block_index_map = {block: i for i, block in enumerate(all_k_blocks)}

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # State for greedy
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Greedy Phase (Deterministic)
    while True:
        if time.time() > deadline - 10.0:
            break
            
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
            
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
                    
    # Map solution blocks to their global indices for fast lookup using the precomputed map
    sol_indices = [block_index_map[tuple(b)] for b in solution_blocks]
    
    def remove_redundant(sol_indices, cover_count):
        changed = True
        while changed:
            changed = False
            for i in range(len(sol_indices)):
                if time.time() > deadline - 2.0:
                    return sol_indices, cover_count, True
                b_idx = sol_indices[i]
                redundant = True
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    sol_indices.pop(i)
                    changed = True
                    break
        return sol_indices, cover_count, changed

    def local_search(sol_indices):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in sol_indices:
            for ts_idx in block_to_ts[b_idx]:
                cover_count[ts_idx] += 1
        
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant blocks
            sol_indices, cover_count, changed = remove_redundant(sol_indices, cover_count)
            if changed:
                improved = True
                continue
            
            # 2. Try 2-for-1 merge
            n = len(sol_indices)
            found_merge = False
            # Limit the search if time is running out
            max_i = n if time.time() < deadline - 5.0 else min(10, n)
            
            for i in range(max_i):
                if time.time() > deadline - 2.0:
                    return sol_indices, cover_count, True
                b_i = sol_indices[i]
                unique_i = [ts for ts in block_to_ts[b_i] if cover_count[ts] == 1]
                
                for j in range(i + 1, n):
                    if time.time() > deadline - 2.0:
                        return sol_indices, cover_count, True
                    b_j = sol_indices[j]
                    unique_j = [ts for ts in block_to_ts[b_j] if cover_count[ts] == 1]
                    
                    all_unique = unique_i + unique_j
                    if not all_unique:
                        continue
                        
                    sorted_uniques = sorted(all_unique, key=lambda x: len(ts_to_blocks[x]))
                    
                    # Use sets for intersection
                    candidates = set(ts_to_blocks[sorted_uniques[0]])
                    for ts_idx in sorted_uniques[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                            
                    if candidates:
                        sol_set = set(sol_indices)
                        valid_candidates = [c for c in candidates if c not in sol_set]
                        if valid_candidates:
                            new_b_idx = random.choice(valid_candidates)
                            
                            for ts_idx in block_to_ts[b_i]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[b_j]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[new_b_idx]:
                                cover_count[ts_idx] += 1
                                
                            sol_indices.pop(j)
                            sol_indices.pop(i)
                            sol_indices.append(new_b_idx)
                            
                            found_merge = True
                            improved = True
                            break
                if found_merge:
                    break
                    
            # 3. Try 1-for-1 swap
            if not improved:
                n = len(sol_indices)
                for i in range(n):
                    if time.time() > deadline - 2.0:
                        return sol_indices, cover_count, True
                    b_i = sol_indices[i]
                    unique_i = [ts for ts in block_to_ts[b_i] if cover_count[ts] == 1]
                    if not unique_i:
                        continue
                        
                    sorted_uniques = sorted(unique_i, key=lambda x: len(ts_to_blocks[x]))
                    candidates = set(ts_to_blocks[sorted_uniques[0]])
                    for ts_idx in sorted_uniques[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                            
                    if candidates:
                        sol_set = set(sol_indices)
                        valid_candidates = [c for c in candidates if c not in sol_set]
                        if valid_candidates:
                            new_b_idx = random.choice(valid_candidates)
                            
                            for ts_idx in block_to_ts[b_i]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[new_b_idx]:
                                cover_count[ts_idx] += 1
                                
                            sol_indices[i] = new_b_idx
                            improved = True
                            break

        return sol_indices, cover_count, improved

    # Run local search
    sol_indices, cover_count, _ = local_search(sol_indices)
    
    # Convert back to list of lists
    best_solution = [list(all_k_blocks[idx]) for idx in sol_indices]
    
    return best_solution
