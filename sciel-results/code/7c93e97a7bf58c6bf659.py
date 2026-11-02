import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Precompute block indices as numpy arrays for fast operations
    block_ts_arrays = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    
    # Precompute tuple-to-index map for fast block lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # State for greedy
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Greedy Phase
    while True:
        if time.time() > deadline - 20.0:
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
            if time.time() > deadline - 10.0:
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
    num_runs = 50
    for i in range(num_runs):
        if time.time() > deadline - 10.0:
            break
            
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Local Search Functions
    
    def build_cover_state(blocks):
        """Builds cover_count and block_ts_indices for a given list of blocks."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                ts_list = block_to_ts[idx]
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
            block_ts_indices.append(ts_list)
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
        return cover_count, block_ts_indices

    def optimize_redundancy(blocks):
        """Removes blocks that do not uniquely cover any t-subset."""
        cover_count, block_ts_indices = build_cover_state(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks, cover_count, block_ts_indices
                
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break
        return blocks, cover_count, block_ts_indices

    def local_search_1swap(blocks, cover_count, block_ts_indices):
        """Tries to swap one block with another to enable further reductions or direct improvement."""
        improved = False
        sol_block_indices = set()
        for i in range(len(blocks)):
            idx = get_block_idx(blocks[i])
            if idx >= 0:
                sol_block_indices.add(idx)
                
        for i in range(len(blocks)):
            if time.time() > deadline - 2.0:
                return blocks, cover_count, block_ts_indices, False
            
            unique_ts = []
            for ts_idx in block_ts_indices[i]:
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
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
                
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block_idx = get_block_idx(blocks[i])
            
            # Check if this swap actually helps (e.g., allows removal of another block later)
            # For now, we just do the swap if it covers the unique parts.
            # We'll rely on optimize_redundancy to clean up.
            
            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            
            # Update cover counts
            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            block_ts_indices[i] = block_to_ts[new_block_idx]
            
            improved = True
            break # Restart local search after a change
            
        return blocks, cover_count, block_ts_indices, improved

    def local_search_2swap(blocks, cover_count, block_ts_indices):
        """Tries to swap two blocks with two other blocks."""
        improved = False
        n = len(blocks)
        sol_block_indices = set()
        for i in range(n):
            idx = get_block_idx(blocks[i])
            if idx >= 0:
                sol_block_indices.add(idx)
                
        for i in range(n):
            if time.time() > deadline - 2.0:
                return blocks, cover_count, block_ts_indices, False
            for j in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    return blocks, cover_count, block_ts_indices, False
                    
                unique_ts_i = []
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        unique_ts_i.append(ts_idx)
                        
                unique_ts_j = []
                for ts_idx in block_ts_indices[j]:
                    if cover_count[ts_idx] == 1:
                        unique_ts_j.append(ts_idx)
                        
                if not unique_ts_i and not unique_ts_j:
                    continue
                    
                # Try to find two blocks B', B'' not in solution such that
                # B' covers unique_ts_i and B'' covers unique_ts_j (or vice versa, or mixed)
                # Simple heuristic: B' covers unique_ts_i, B'' covers unique_ts_j
                
                if unique_ts_i:
                    cand_i = set(ts_to_blocks[unique_ts_i[0]])
                    for ts_idx in unique_ts_i[1:]:
                        cand_i.intersection_update(ts_to_blocks[ts_idx])
                        if not cand_i:
                            break
                else:
                    cand_i = set(range(num_k_blocks))
                    
                if unique_ts_j:
                    cand_j = set(ts_to_blocks[unique_ts_j[0]])
                    for ts_idx in unique_ts_j[1:]:
                        cand_j.intersection_update(ts_to_blocks[ts_idx])
                        if not cand_j:
                            break
                else:
                    cand_j = set(range(num_k_blocks))
                    
                valid_i = [c for c in cand_i if c not in sol_block_indices]
                valid_j = [c for c in cand_j if c not in sol_block_indices and c not in cand_i]
                
                if valid_i and valid_j:
                    new_i_idx = random.choice(valid_i)
                    new_j_idx = random.choice(valid_j)
                    new_i = list(all_k_blocks[new_i_idx])
                    new_j = list(all_k_blocks[new_j_idx])
                    
                    old_i_idx = get_block_idx(blocks[i])
                    old_j_idx = get_block_idx(blocks[j])
                    
                    blocks[i] = new_i
                    blocks[j] = new_j
                    sol_block_indices.discard(old_i_idx)
                    sol_block_indices.discard(old_j_idx)
                    sol_block_indices.add(new_i_idx)
                    sol_block_indices.add(new_j_idx)
                    
                    # Update cover counts
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_ts_indices[j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_i_idx]:
                        cover_count[ts_idx] += 1
                    for ts_idx in block_to_ts[new_j_idx]:
                        cover_count[ts_idx] += 1
                    block_ts_indices[i] = block_to_ts[new_i_idx]
                    block_ts_indices[j] = block_to_ts[new_j_idx]
                    
                    improved = True
                    break
            if improved:
                break
                
        return blocks, cover_count, block_ts_indices, improved

    def local_search_add_remove(blocks, cover_count, block_ts_indices):
        """Tries to add a block and then remove two blocks (net -1)."""
        improved = False
        n = len(blocks)
        sol_block_indices = set()
        for i in range(n):
            idx = get_block_idx(blocks[i])
            if idx >= 0:
                sol_block_indices.add(idx)
                
        # Randomly select a candidate block to add
        # We prefer blocks that cover many currently single-covered t-subsets? 
        # Or just random.
        
        # To be efficient, we'll try a few random candidates
        attempts = 10
        for _ in range(attempts):
            if time.time() > deadline - 2.0:
                return blocks, cover_count, block_ts_indices, False
                
            new_block_idx = random.randint(0, num_k_blocks - 1)
            if new_block_idx in sol_block_indices:
                continue
                
            new_block = list(all_k_blocks[new_block_idx])
            
            # Temporarily add
            blocks.append(new_block)
            sol_block_indices.add(new_block_idx)
            
            # Update cover counts
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            block_ts_indices.append(block_to_ts[new_block_idx])
            
            # Try to remove two blocks
            # We look for pairs of blocks that are now redundant or become redundant
            # Specifically, if adding the new block made some t-subsets double-covered,
            # maybe we can remove a block that was only covering those.
            
            # Run a limited redundancy optimization
            removed_any = False
            for _ in range(2):
                found_redundant = False
                for i in range(len(blocks)):
                    redundant = True
                    for ts_idx in block_ts_indices[i]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove block i
                        for ts_idx in block_ts_indices[i]:
                            cover_count[ts_idx] -= 1
                        block_ts_indices.pop(i)
                        blocks.pop(i)
                        idx = get_block_idx(blocks[i-1]) # Careful with indices, but we popped
                        # Actually, we need to track the index of the removed block to update sol_block_indices
                        # Let's do this more carefully.
                        
                        # Re-implementing removal logic for this specific context
                        # Since we are in a nested function, let's just break and let the main loop handle it?
                        # No, we need to be atomic.
                        
                        # Let's restart the removal logic properly.
                        pass
                break # Simplified: just do one pass of redundancy check
                
            # Proper redundancy check after addition
            cover_count, block_ts_indices = build_cover_state(blocks)
            new_len = len(blocks)
            
            # Try to remove blocks
            for _ in range(2):
                removed = False
                for i in range(len(blocks)):
                    redundant = True
                    for ts_idx in block_ts_indices[i]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove
                        idx = get_block_idx(blocks[i])
                        sol_block_indices.discard(idx)
                        blocks.pop(i)
                        for ts_idx in block_ts_indices[i]:
                            cover_count[ts_idx] -= 1
                        block_ts_indices.pop(i)
                        removed = True
                        break
                if not removed:
                    break
                    
            if len(blocks) < n:
                improved = True
                return blocks, cover_count, block_ts_indices, improved
            else:
                # Revert
                blocks.pop()
                sol_block_indices.discard(new_block_idx)
                # Rebuild state to be safe
                cover_count, block_ts_indices = build_cover_state(blocks)
                
        return blocks, cover_count, block_ts_indices, improved

    # Main Local Search Loop
    cover_count, block_ts_indices = build_cover_state(best_solution)
    best_solution, cover_count, block_ts_indices = optimize_redundancy(best_solution)
    if len(best_solution) < best_len:
        best_len = len(best_solution)
        
    while time.time() < deadline - 5.0:
        improved = False
        
        # 1. Redundancy
        best_solution, cover_count, block_ts_indices = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
            improved = True
            
        # 2. 1-Swap
        best_solution, cover_count, block_ts_indices, imp = local_search_1swap(best_solution, cover_count, block_ts_indices)
        if imp:
            improved = True
            # Re-optimize after swap
            best_solution, cover_count, block_ts_indices = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
            continue # Restart loop
            
        # 3. 2-Swap
        best_solution, cover_count, block_ts_indices, imp = local_search_2swap(best_solution, cover_count, block_ts_indices)
        if imp:
            improved = True
            best_solution, cover_count, block_ts_indices = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
            continue # Restart loop
            
        # 4. Add-Remove
        best_solution, cover_count, block_ts_indices, imp = local_search_add_remove(best_solution, cover_count, block_ts_indices)
        if imp:
            improved = True
            best_solution, cover_count, block_ts_indices = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
            continue # Restart loop
            
        if not improved:
            break

    return best_solution
