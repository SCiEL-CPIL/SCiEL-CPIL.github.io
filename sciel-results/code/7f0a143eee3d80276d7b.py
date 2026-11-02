import time
import random
import itertools
import numpy as np
import sys

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
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # For each block, list of t-subset indices it covers
    block_to_ts = []
    # For each t-subset, list of block indices that cover it
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Greedy Phase
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
            
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        solution_blocks = []
        
        while True:
            if time.time() > deadline - 10.0:
                break
            max_idx = int(np.argmax(gain))
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
        return solution_blocks

    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        solution_blocks = []
        
        while True:
            if time.time() > deadline - 5.0:
                break
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            # Add some randomness: pick from top candidates
            # Instead of just argmax, pick from those with gain close to max
            # But for speed, just random among max is okay, or weighted
            b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            solution_blocks.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        return solution_blocks

    # Local Search Optimization
    def optimize_blocks(blocks):
        # Returns a potentially better set of blocks
        # 1. Remove redundant blocks
        # 2. Try 1-for-1 swaps
        # 3. Try 2-for-1 swaps
        
        if not blocks:
            return blocks
            
        # Map block tuple to index for fast lookup
        block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
        
        def get_idx(b):
            return block_tuple_to_idx.get(tuple(b), -1)

        # State: cover count of each t-subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts_idx in block_to_ts[get_idx(b)]:
                cover_count[ts_idx] += 1
        
        sol_indices = set(get_idx(b) for b in blocks)
        
        changed = True
        while changed:
            changed = False
            
            # Phase 1: Remove redundant blocks
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    break
                b_idx = get_idx(blocks[i])
                redundant = True
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    sol_indices.discard(b_idx)
                    changed = True
                    break
            
            if changed:
                continue

            # Phase 2: 1-for-1 Swaps
            # For each block, find its unique t-subsets
            # If there is another block covering all of them, swap
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    break
                b_idx = get_idx(blocks[i])
                unique_ts = []
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                
                if not unique_ts:
                    continue
                
                # Find candidates covering all unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if candidates:
                    # Filter out current solution blocks
                    valid_candidates = [c for c in candidates if c not in sol_indices]
                    if valid_candidates:
                        new_b_idx = random.choice(valid_candidates)
                        new_block = list(all_k_blocks[new_b_idx])
                        
                        # Apply swap
                        for ts_idx in block_to_ts[b_idx]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[new_b_idx]:
                            cover_count[ts_idx] += 1
                        
                        blocks[i] = new_block
                        sol_indices.discard(b_idx)
                        sol_indices.add(new_b_idx)
                        changed = True
                        break
            
            if changed:
                continue

            # Phase 3: 2-for-1 Swaps
            # Try to replace two blocks with one
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 2.0:
                        break
                    
                    b_idx_i = get_idx(blocks[i])
                    b_idx_j = get_idx(blocks[j])
                    
                    unique_ts_i = []
                    for ts_idx in block_to_ts[b_idx_i]:
                        if cover_count[ts_idx] == 1:
                            unique_ts_i.append(ts_idx)
                            
                    unique_ts_j = []
                    for ts_idx in block_to_ts[b_idx_j]:
                        if cover_count[ts_idx] == 1:
                            unique_ts_j.append(ts_idx)
                            
                    all_unique = unique_ts_i + unique_ts_j
                    if not all_unique:
                        continue
                        
                    candidates = set(ts_to_blocks[all_unique[0]])
                    for ts_idx in all_unique[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                            
                    if candidates:
                        valid_candidates = [c for c in candidates if c not in sol_indices]
                        if valid_candidates:
                            new_b_idx = random.choice(valid_candidates)
                            new_block = list(all_k_blocks[new_b_idx])
                            
                            # Remove i and j, add new
                            for ts_idx in block_to_ts[b_idx_i]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[b_idx_j]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts[new_b_idx]:
                                cover_count[ts_idx] += 1
                                
                            sol_indices.discard(b_idx_i)
                            sol_indices.discard(b_idx_j)
                            sol_indices.add(new_b_idx)
                            
                            blocks.pop(j)
                            blocks[i] = new_block
                            changed = True
                            break
                if changed:
                    break
        
        return blocks

    # Run initial greedy
    best_solution = greedy_cover()
    best_len = len(best_solution)
    
    # Run multiple randomized greeds to find a better start
    # Each randomized greedy is fast
    for i in range(50):
        if time.time() > deadline - 20.0:
            break
        sol = randomized_greedy(i)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
        # Optimize the new solution immediately if it's promising
        if len(sol) <= best_len + 5:
            optimized = optimize_blocks(sol)
            if len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized

    # Deep local search on the best solution
    # Iterate optimization multiple times to ensure convergence
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        current = optimize_blocks(best_solution)
        if len(current) < best_len:
            best_len = len(current)
            best_solution = current
        else:
            # If no improvement, try a randomized perturbation?
            # Or just break if we've hit a stable point
            pass

    return best_solution
