import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

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

    # For C(19, 8, 5): use cyclic structure
    # We'll build blocks using a difference family approach on Z_19
    # A cyclic (v, k, t) difference family: sets of k-subsets of Z_v such that
    # every t-subset of Z_v appears as a difference (mod v) in some block.
    
    # Actually, for covering, we need every t-subset to be contained in some block.
    # Let's use a simpler approach: start with a good greedy, then do intensive local search.
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Build incidence
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
    
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 10.0:
                return sol
            
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
                    removed = blocks.pop(i)
                    for ts in itertools.combinations(removed, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
        
        return blocks
    
    def local_search(blocks, max_iters=50):
        """Iteratively try to remove a block and replace it with a better one, or just remove if redundant."""
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
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 2.0:
                break
            
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    break
                
                # Find t-subsets uniquely covered by blocks[i]
                unique_ts = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)
                
                if not unique_ts:
                    # Block is redundant, remove it
                    removed = blocks.pop(i)
                    for ts in itertools.combinations(removed, t):
                        cover_count[ts_index_map[ts]] -= 1
                    sol_block_set.discard(get_block_idx(removed))
                    improved = True
                    break
                
                # Try to find a replacement block that covers all unique_ts
                # and is not already in the solution
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                current_idx = get_block_idx(blocks[i])
                valid_candidates = [c for c in candidates if c not in sol_block_set or c == current_idx]
                
                if not valid_candidates:
                    continue
                
                # Pick the candidate that maximizes additional coverage (covers most non-unique t-subsets)
                best_candidate = None
                best_score = -1
                for cand_idx in valid_candidates:
                    # Score: number of t-subsets in cand that are NOT in unique_ts
                    # (i.e., how much extra coverage it provides)
                    extra = 0
                    for ts_idx in block_to_ts[cand_idx]:
                        if ts_idx not in unique_ts and cover_count[ts_idx] == 0:
                            extra += 1
                    # Prefer candidates that are already in solution? No, we want new ones.
                    # Actually, we want to minimize the loss of coverage from removing blocks[i]
                    # and maximize the gain from adding the new block.
                    # The unique_ts must be covered. The rest is bonus.
                    if extra > best_score:
                        best_score = extra
                        best_candidate = cand_idx
                
                if best_candidate is None or best_candidate == current_idx:
                    continue
                
                # Perform swap
                old_block = blocks[i]
                new_block = list(all_k_blocks[best_candidate])
                
                # Update cover counts
                for ts in itertools.combinations(old_block, t):
                    cover_count[ts_index_map[ts]] -= 1
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1
                
                # Check validity
                if np.min(cover_count) >= 1:
                    blocks[i] = new_block
                    sol_block_set.discard(current_idx)
                    sol_block_set.add(best_candidate)
                    improved = True
                    break
                else:
                    # Revert
                    for ts in itertools.combinations(new_block, t):
                        cover_count[ts_index_map[ts]] -= 1
                    for ts in itertools.combinations(old_block, t):
                        cover_count[ts_index_map[ts]] += 1
                    sol_block_set.add(current_idx)
                    sol_block_set.discard(best_candidate)
        
        return blocks
    
    def add_missing(blocks):
        """Add blocks to cover any missing t-subsets."""
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        
        missing = [i for i in range(num_t_subsets) if cover_count[i] == 0]
        if not missing:
            return blocks
        
        # Greedily add blocks to cover missing t-subsets
        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)
        
        while missing and time.time() < deadline - 2.0:
            # Find block that covers the most missing t-subsets
            best_block = None
            best_count = 0
            for b_idx in range(num_k_blocks):
                if b_idx in sol_block_set:
                    continue
                count = sum(1 for ts_idx in block_to_ts[b_idx] if cover_count[ts_idx] == 0)
                if count > best_count:
                    best_count = count
                    best_block = b_idx
            
            if best_block is None or best_count == 0:
                break
            
            blocks.append(list(all_k_blocks[best_block]))
            sol_block_set.add(best_block)
            for ts_idx in block_to_ts[best_block]:
                cover_count[ts_idx] += 1
            
            missing = [i for i in range(num_t_subsets) if cover_count[i] == 0]
        
        return blocks
    
    # Multi-start greedy + local search
    best_solution = None
    best_len = float('inf')
    
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 15.0:
            break
        
        sol = greedy_cover(seed=i * 12345 + 678)
        sol = remove_redundant(sol)
        sol = local_search(sol, max_iters=30)
        sol = add_missing(sol)
        sol = remove_redundant(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Final local search on the best solution
    if best_solution is not None:
        best_solution = local_search(best_solution, max_iters=50)
        best_solution = remove_redundant(best_solution)
        best_solution = add_missing(best_solution)
        best_solution = remove_redundant(best_solution)
    
    if best_solution is None:
        # Fallback: pure greedy
        best_solution = greedy_cover(seed=0)
        best_solution = remove_redundant(best_solution)
    
    return best_solution
