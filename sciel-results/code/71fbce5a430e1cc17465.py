import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

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

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures
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

    def time_left():
        return deadline - time.time()

    # Phase 1: Build a maximal t-packing (blocks with disjoint t-subsets)
    def build_maximal_packing(seed=42):
        random.seed(seed)
        packing = []
        used_ts = set()
        remaining_blocks = list(range(num_k_blocks))
        random.shuffle(remaining_blocks)
        
        for b_idx in remaining_blocks:
            if time_left() < 30.0:
                break
            block_ts = set(block_to_ts[b_idx])
            if block_ts.isdisjoint(used_ts):
                packing.append(b_idx)
                used_ts.update(block_ts)
        
        return packing, used_ts

    # Phase 2: Packing-guided greedy completion
    def packing_guided_greedy(packing, used_ts, seed=0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = list(packing)
        
        # Mark packing blocks as selected and update state
        for b_idx in packing:
            is_selected[b_idx] = True
            gain[b_idx] = -1
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        
        # Now greedily add blocks to cover remaining t-subsets
        while True:
            if time_left() < 10.0:
                break
            if not is_uncovered.any():
                break
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Prefer blocks that cover many uncovered t-subsets
            # with a slight bias toward blocks that overlap with packing structure
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        
        return sol

    # Phase 3: Standard greedy (for comparison)
    def standard_greedy(seed=0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time_left() < 10.0:
                return None
            if not is_uncovered.any():
                break
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        
        return sol

    # Local search: Remove redundant blocks
    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
        
        changed = True
        while changed and time_left() > 2.0:
            changed = False
            for i in range(len(blocks)):
                if time_left() < 1.0:
                    return blocks
                idx = get_block_idx(blocks[i])
                if idx < 0:
                    continue
                redundant = True
                for ts_idx in block_to_ts[idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] -= 1
                    changed = True
                    break
        return blocks

    # Local search: 1-move swap
    def local_search_swap(blocks, max_iterations=50):
        if not blocks:
            return blocks, False
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
        
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
        
        improved = False
        for _ in range(max_iterations):
            if time_left() < 2.0:
                break
            made_swap = False
            for i in range(len(blocks)):
                if time_left() < 1.0:
                    return blocks, improved
                idx = get_block_idx(blocks[i])
                if idx < 0:
                    continue
                
                unique_ts = []
                for ts_idx in block_to_ts[idx]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                
                if not unique_ts:
                    continue
                
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                candidates.discard(idx)
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                
                if not valid_candidates:
                    continue
                
                new_block_idx = random.choice(valid_candidates)
                
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] += 1
                
                blocks[i] = list(all_k_blocks[new_block_idx])
                sol_block_indices.discard(idx)
                sol_block_indices.add(new_block_idx)
                
                improved = True
                made_swap = True
                break
            
            if not made_swap:
                break
            
            # Check for redundancy after swap
            blocks = optimize_redundancy(blocks)
            if len(blocks) < max_iterations:
                # Rebuild cover_count after redundancy removal
                cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                for b in blocks:
                    idx = get_block_idx(b)
                    if idx >= 0:
                        for ts_idx in block_to_ts[idx]:
                            cover_count[ts_idx] += 1
                sol_block_indices = set()
                for b in blocks:
                    idx = get_block_idx(b)
                    if idx >= 0:
                        sol_block_indices.add(idx)
        
        return blocks, improved

    # Local search: 2-exchange (remove 2, add 1)
    def local_search_2exchange(blocks, max_rounds=20):
        if not blocks or len(blocks) < 3:
            return blocks, False
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
        
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
        
        improved = False
        for _ in range(max_rounds):
            if time_left() < 2.0:
                break
            found = False
            n = len(blocks)
            for i in range(n):
                if time_left() < 1.0:
                    return blocks, improved
                idx_i = get_block_idx(blocks[i])
                if idx_i < 0:
                    continue
                for j in range(i + 1, n):
                    if time_left() < 0.5:
                        return blocks, improved
                    idx_j = get_block_idx(blocks[j])
                    if idx_j < 0:
                        continue
                    
                    unique_ts = set()
                    for ts_idx in block_to_ts[idx_i]:
                        if cover_count[ts_idx] == 1:
                            unique_ts.add(ts_idx)
                    for ts_idx in block_to_ts[idx_j]:
                        if cover_count[ts_idx] == 1:
                            unique_ts.add(ts_idx)
                    
                    if not unique_ts:
                        continue
                    
                    unique_ts_list = list(unique_ts)
                    candidates = set(ts_to_blocks[unique_ts_list[0]])
                    for ts_idx in unique_ts_list[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    
                    candidates.discard(idx_i)
                    candidates.discard(idx_j)
                    valid_candidates = [c for c in candidates if c not in sol_block_indices]
                    
                    if not valid_candidates:
                        continue
                    
                    new_block_idx = random.choice(valid_candidates)
                    
                    # Try the exchange
                    for ts_idx in block_to_ts[idx_i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[idx_j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_block_idx]:
                        cover_count[ts_idx] += 1
                    
                    valid = True
                    for ts_idx in range(num_t_subsets):
                        if cover_count[ts_idx] < 1:
                            valid = False
                            break
                    
                    if valid:
                        blocks.pop(j)
                        blocks.pop(i)
                        blocks.append(list(all_k_blocks[new_block_idx]))
                        sol_block_indices.discard(idx_i)
                        sol_block_indices.discard(idx_j)
                        sol_block_indices.add(new_block_idx)
                        improved = True
                        found = True
                        break
                    else:
                        # Rollback
                        for ts_idx in block_to_ts[new_block_idx]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[idx_i]:
                            cover_count[ts_idx] += 1
                        for ts_idx in block_to_ts[idx_j]:
                            cover_count[ts_idx] += 1
                    
                    if found:
                        break
                
                if found:
                    break
            
            if not found:
                break
            
            blocks = optimize_redundancy(blocks)
        
        return blocks, improved

    # Main search loop
    best_solution = None
    best_len = float('inf')
    
    # Try multiple packing-guided approaches
    num_packings = 5
    for p in range(num_packings):
        if time_left() < 20.0:
            break
        packing, used_ts = build_maximal_packing(seed=100 + p)
        sol = packing_guided_greedy(packing, used_ts, seed=p)
        if sol is not None:
            sol_lists = [list(all_k_blocks[b]) for b in sol]
            sol_lists = optimize_redundancy(sol_lists)
            if len(sol_lists) < best_len:
                best_len = len(sol_lists)
                best_solution = sol_lists
    
    # Try standard greedy with different seeds
    for s in range(20):
        if time_left() < 15.0:
            break
        sol = standard_greedy(seed=s)
        if sol is not None:
            sol_lists = [list(all_k_blocks[b]) for b in sol]
            sol_lists = optimize_redundancy(sol_lists)
            if len(sol_lists) < best_len:
                best_len = len(sol_lists)
                best_solution = sol_lists
    
    if best_solution is None:
        # Fallback: simple greedy
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            if time_left() < 5.0:
                break
            if not is_uncovered.any():
                break
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            b_idx = candidates[0]
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        best_solution = [list(all_k_blocks[b]) for b in sol]
        best_len = len(best_solution)
    
    # Local search refinement
    for _ in range(50):
        if time_left() < 5.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_iterations=30)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
    
    for _ in range(20):
        if time_left() < 5.0:
            break
        best_solution, improved = local_search_2exchange(best_solution, max_rounds=10)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
    
    # Final cleanup
    best_solution = optimize_redundancy(best_solution)
    
    return best_solution
