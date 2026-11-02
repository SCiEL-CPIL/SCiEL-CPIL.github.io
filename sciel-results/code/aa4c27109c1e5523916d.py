import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

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

    def greedy_construction(seed=None, weight_temp=0.0):
        if seed is not None:
            random.seed(seed)
        
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 8.0:
                return None
            
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
            
            if weight_temp > 0:
                top_k = 20
                top_indices = np.argsort(gain)[-top_k:]
                gains = gain[top_indices]
                max_g = np.max(gains)
                log_probs = (gains - max_g) / weight_temp
                log_probs -= np.max(log_probs)
                probs = np.exp(log_probs)
                probs /= np.sum(probs)
                chosen_local = np.random.choice(len(top_indices), p=probs)
                b_idx = top_indices[chosen_local]
            else:
                b_idx = max_idx
            
            if is_selected[b_idx]:
                continue
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    def remove_redundant(blocks):
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

    def local_search(blocks, max_passes=5):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        
        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)
        
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            if time.time() > deadline - 3.0:
                break
            improved = False
            passes += 1
            
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 3.0:
                    break
                
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
                    improved = True
                    break
            else:
                for i in range(len(blocks)):
                    if time.time() > deadline - 3.0:
                        break
                    
                    unique_ts = []
                    for ts in itertools.combinations(blocks[i], t):
                        idx = ts_index_map[ts]
                        if cover_count[idx] == 1:
                            unique_ts.append(idx)
                            
                    if not unique_ts:
                        block_to_remove = blocks[i]
                        blocks.pop(i)
                        for ts in itertools.combinations(block_to_remove, t):
                            cover_count[ts_index_map[ts]] -= 1
                        improved = True
                        break
                    
                    candidates = set(ts_to_blocks[unique_ts[0]])
                    for ts_idx in unique_ts[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    
                    current_idx = get_block_idx(blocks[i])
                    candidates.discard(current_idx)
                    
                    valid_candidates = [c for c in candidates if c not in sol_block_set]
                    
                    if not valid_candidates:
                        continue
                    
                    random.shuffle(valid_candidates)
                    for new_idx in valid_candidates[:10]:
                        old_block = blocks[i]
                        new_block = list(all_k_blocks[new_idx])
                        
                        old_counts = cover_count.copy()
                        
                        for ts in itertools.combinations(old_block, t):
                            cover_count[ts_index_map[ts]] -= 1
                        
                        if np.min(cover_count) < 0 or np.any(cover_count == 0):
                            cover_count[:] = old_counts
                            continue
                        
                        for ts in itertools.combinations(new_block, t):
                            cover_count[ts_index_map[ts]] += 1
                        
                        blocks[i] = new_block
                        sol_block_set.remove(current_idx)
                        sol_block_set.add(new_idx)
                        improved = True
                        break
                        
            if improved:
                cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                for b in blocks:
                    for ts in itertools.combinations(b, t):
                        cover_count[ts_index_map[ts]] += 1
                sol_block_set = set()
                for b in blocks:
                    idx = get_block_idx(b)
                    if idx >= 0:
                        sol_block_set.add(idx)
        
        return blocks

    def iterative_compression(initial_solution, seed=0, max_iterations=20):
        """
        Iterative compression: repeatedly remove a random block and try to re-cover
        the uncovered t-subsets by adding fewer blocks.
        """
        if seed is not None:
            random.seed(seed)
        
        current_solution = list(initial_solution)
        if not current_solution:
            return current_solution
        
        best_solution = list(current_solution)
        best_len = len(current_solution)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Remove a random block
            remove_idx = random.randint(0, len(current_solution) - 1)
            removed_block = current_solution.pop(remove_idx)
            
            # Find uncovered t-subsets
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            for b in current_solution:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
            
            uncovered_ts = [ts_idx for ts_idx in range(num_t_subsets) if cover_count[ts_idx] == 0]
            
            if not uncovered_ts:
                # Removing the block didn't uncover anything, keep it removed
                if len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = list(current_solution)
                current_solution.append(removed_block)
                continue
            
            # Try to cover uncovered t-subsets with greedy
            new_blocks = []
            remaining_uncovered = set(uncovered_ts)
            
            # Build a local gain array for blocks that cover at least one uncovered ts
            local_gain = np.zeros(num_k_blocks, dtype=np.int32)
            for ts_idx in remaining_uncovered:
                for b_idx in ts_to_blocks[ts_idx]:
                    if b_idx != get_block_idx(removed_block):
                        local_gain[b_idx] += 1
            
            selected_local = set()
            while remaining_uncovered:
                if time.time() > deadline - 5.0:
                    break
                
                max_gain = np.max(local_gain)
                if max_gain <= 0:
                    break
                
                # Get all blocks with max gain
                max_candidates = np.where(local_gain == max_gain)[0]
                chosen = random.choice(max_candidates.tolist())
                
                if chosen in selected_local:
                    continue
                
                selected_local.add(chosen)
                new_blocks.append(list(all_k_blocks[chosen]))
                
                # Update remaining uncovered
                for ts_idx in block_to_ts[chosen]:
                    if ts_idx in remaining_uncovered:
                        remaining_uncovered.discard(ts_idx)
                        for nb_idx in ts_to_blocks[ts_idx]:
                            if nb_idx not in selected_local:
                                local_gain[nb_idx] -= 1
            
            if remaining_uncovered:
                # Failed to cover all, restore
                current_solution.append(removed_block)
                continue
            
            # Check if we saved blocks
            if len(new_blocks) < 1:
                # We covered everything with 0 new blocks (shouldn't happen if uncovered_ts was non-empty)
                pass
            elif len(new_blocks) <= 1:
                # Net saving or break-even
                current_solution.extend(new_blocks)
                if len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = list(current_solution)
            else:
                # More blocks added, revert
                current_solution.append(removed_block)
        
        return best_solution

    # Main algorithm
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Run multiple greedy constructions
    num_greedy_runs = 30
    for i in range(num_greedy_runs):
        if time.time() > deadline - 15.0:
            break
        
        sol = greedy_construction(seed=i, weight_temp=0.0)
        if sol is not None:
            sol = remove_redundant(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
    
    # Phase 2: Local search on best
    if best_solution is not None and time.time() < deadline - 10.0:
        improved_sol = local_search(best_solution, max_passes=5)
        if len(improved_sol) < best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol
    
    # Phase 3: Iterative compression
    if best_solution is not None and time.time() < deadline - 10.0:
        compressed_sol = iterative_compression(best_solution, seed=42, max_iterations=50)
        if len(compressed_sol) < best_len:
            best_len = len(compressed_sol)
            best_solution = compressed_sol
    
    # Phase 4: Final local search
    if best_solution is not None and time.time() < deadline - 5.0:
        improved_sol = local_search(best_solution, max_passes=10)
        if len(improved_sol) < best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol
    
    # Phase 5: Final redundancy removal
    if best_solution is not None:
        best_solution = remove_redundant(best_solution)
    
    if best_solution is None:
        sol = greedy_construction(seed=0, weight_temp=0.0)
        best_solution = sol if sol is not None else []
    
    return best_solution
