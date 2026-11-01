import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

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
        
    # Precompute a map from tuple(block) to its index in all_k_blocks for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
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
                    
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Function to run a randomized greedy
    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
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

    # Run randomized greedy multiple times to find a better starting point
    num_runs = 100
    for i in range(num_runs):
        if time.time() > deadline - 5.0:
            break
            
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Local Search: Remove redundant blocks (efficient version)
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Use block_to_ts for speed
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                    
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 1.0:
                    return blocks
                
                idx = get_block_idx(blocks[i])
                if idx >= 0:
                    redundant = True
                    for ts_idx in block_to_ts[idx]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                else:
                    redundant = True
                    for ts in itertools.combinations(blocks[i], t):
                        if cover_count[ts_index_map[ts]] == 1:
                            redundant = False
                            break
                        
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    idx = get_block_idx(block_to_remove)
                    if idx >= 0:
                        for ts_idx in block_to_ts[idx]:
                            cover_count[ts_idx] -= 1
                    else:
                        for ts in itertools.combinations(block_to_remove, t):
                            cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
                    
        return blocks

    best_solution = optimize_redundancy(best_solution)
    
    # Local Search: 1-move swap optimization
    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                    
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            idx = get_block_idx(blocks[i])
            if idx < 0:
                continue
                
            # Find t-subsets uniquely covered by blocks[i]
            unique_ts = []
            for ts_idx in block_to_ts[idx]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
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
            
            # Remove current block from candidates
            candidates.discard(idx)
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            
            if not valid_candidates:
                continue
                
            # Pick one randomly
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            # Perform swap
            old_block = blocks[i]
            blocks[i] = new_block
            
            # Update cover counts
            for ts_idx in block_to_ts[idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            
            # Update solution block indices
            sol_block_indices.discard(idx)
            sol_block_indices.add(new_block_idx)
            
            improved = True
            
        return blocks, improved

    # Local Search: 2-exchange optimization
    def local_search_2exchange(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                    
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        
        # Try removing 2 blocks and adding 1 that covers all their unique ts
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            idx_i = get_block_idx(blocks[i])
            if idx_i < 0:
                continue
                
            for j in range(i + 1, len(blocks)):
                if time.time() > deadline - 1.0:
                    return blocks, improved
                    
                idx_j = get_block_idx(blocks[j])
                if idx_j < 0:
                    continue
                    
                # Find ts uniquely covered by blocks[i] or blocks[j]
                # These are ts where cover_count == 1 AND the only covering block is i or j
                unique_ts = set()
                for ts_idx in block_to_ts[idx_i]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.add(ts_idx)
                for ts_idx in block_to_ts[idx_j]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.add(ts_idx)
                        
                if not unique_ts:
                    continue
                    
                # Find blocks that cover ALL these unique ts
                candidates = set(ts_to_blocks[list(unique_ts)[0]])
                for ts_idx in list(unique_ts)[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                # Remove i and j from candidates
                candidates.discard(idx_i)
                candidates.discard(idx_j)
                
                # Filter out blocks already in solution
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                
                if not valid_candidates:
                    continue
                    
                # Try the swap: remove i and j, add new block
                new_block_idx = random.choice(valid_candidates)
                new_block = list(all_k_blocks[new_block_idx])
                
                # Check if this reduces size (it does: -2 + 1 = -1)
                # Perform the exchange
                old_blocks = [blocks[i], blocks[j]]
                
                # Update cover counts
                for ts_idx in block_to_ts[idx_i]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[idx_j]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] += 1
                    
                # Verify validity: all ts should still be covered
                valid = True
                for ts_idx in range(num_t_subsets):
                    if cover_count[ts_idx] < 1:
                        valid = False
                        break
                        
                if valid:
                    # Perform the swap
                    blocks.pop(j)
                    blocks.pop(i)
                    blocks.append(new_block)
                    
                    sol_block_indices.discard(idx_i)
                    sol_block_indices.discard(idx_j)
                    sol_block_indices.add(new_block_idx)
                    
                    improved = True
                    return blocks, improved  # Restart after a 2-exchange
                    
                # Rollback
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[idx_i]:
                    cover_count[ts_idx] += 1
                for ts_idx in block_to_ts[idx_j]:
                    cover_count[ts_idx] += 1
                    
        return blocks, improved

    # Run local search for a few iterations
    for _ in range(30):
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Run 2-exchange local search
    for _ in range(10):
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_2exchange(best_solution)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Randomized Swap Search (WalkSAT-style)
    def local_search_random_swap(blocks, num_iterations=2000):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                    
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for _ in range(num_iterations):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            if len(blocks) == 0:
                break
                
            # Pick a random block to remove
            i = random.randint(0, len(blocks) - 1)
            idx_out = get_block_idx(blocks[i])
            if idx_out < 0:
                continue
                
            # Temporarily remove
            for ts_idx in block_to_ts[idx_out]:
                cover_count[ts_idx] -= 1
                
            # Check if still valid
            if np.all(cover_count >= 1):
                # Valid removal!
                blocks.pop(i)
                sol_block_indices.discard(idx_out)
                improved = True
                return blocks, improved
                
            # Find uncovered ts
            uncovered_ts = np.where(cover_count == 0)[0]
            
            # Try to find a block to add that covers all uncovered_ts
            if len(uncovered_ts) > 100: # Heuristic limit
                for ts_idx in block_to_ts[idx_out]:
                    cover_count[ts_idx] += 1
                continue
                
            # Pick a random uncovered ts to guide the search
            target_ts_idx = uncovered_ts[random.randint(0, len(uncovered_ts) - 1)]
            
            candidates = ts_to_blocks[target_ts_idx]
            random.shuffle(candidates)
            
            found = False
            for b_idx_in in candidates:
                if b_idx_in in sol_block_indices or b_idx_in == idx_out:
                    continue
                    
                # Check if this block covers ALL uncovered_ts
                valid = True
                for ts_idx in uncovered_ts:
                    if ts_idx not in block_to_ts[b_idx_in]:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    blocks[i] = list(all_k_blocks[b_idx_in])
                    sol_block_indices.discard(idx_out)
                    sol_block_indices.add(b_idx_in)
                    
                    # Update cover counts
                    for ts_idx in block_to_ts[idx_out]:
                        cover_count[ts_idx] += 1
                    for ts_idx in block_to_ts[b_idx_in]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    found = True
                    break
                    
            if found:
                return blocks, improved
            else:
                # Rollback
                for ts_idx in block_to_ts[idx_out]:
                    cover_count[ts_idx] += 1

        return blocks, improved

    # Run randomized swap search
    for _ in range(20):
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_random_swap(best_solution, 500)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Final redundancy cleanup
    best_solution = optimize_redundancy(best_solution)

    return best_solution
