import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

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
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Precompute as numpy arrays
    block_ts_arrays = [np.array(block_to_ts[j], dtype=np.int32) for j in range(num_k_blocks)]
    ts_to_blocks_arrays = [np.array(ts_to_blocks[i], dtype=np.int32) for i in range(num_t_subsets)]
    
    # Precompute bitsets for fast coverage checks
    num_t_subsets_bits = (num_t_subsets + 63) // 64
    block_bitsets = np.zeros((num_k_blocks, num_t_subsets_bits), dtype=np.uint64)
    for j in range(num_k_blocks):
        for ts_idx in block_to_ts[j]:
            block_bitsets[j, ts_idx // 64] |= (1 << (ts_idx % 64))
    
    # Convert ts_to_blocks to numpy arrays for faster intersection
    ts_to_blocks_np = [np.array(ts_to_blocks[i], dtype=np.int32) for i in range(num_t_subsets)]
    
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, weight=None):
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
            
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    # Phase 1: Multiple greedy runs
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    num_greedy_runs = 200
    for i in range(1, num_greedy_runs):
        if time.time() > deadline - 60.0:
            break
        sol = run_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Phase 2: Local search with swaps
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks
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
        return blocks

    def local_search_swap(blocks, max_iters=300):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            
            # Pick a random block to try replacing
            i = random.randint(0, len(blocks) - 1)
            
            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
            if not unique_ts:
                # Block is redundant, remove it
                blocks.pop(i)
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                block_ts_indices.pop(i)
                improved = True
                continue
            
            # Find candidate blocks that cover all unique_ts using numpy intersection
            if len(unique_ts) == 1:
                candidate_array = ts_to_blocks_np[unique_ts[0]]
            else:
                # Use set intersection for multiple unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                if not candidates:
                    continue
                candidate_array = np.array(list(candidates), dtype=np.int32)
            
            valid_candidates = candidate_array[~np.isin(candidate_array, list(sol_block_indices))]
            if len(valid_candidates) == 0:
                continue
                
            new_block_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
            
            # Check if this swap improves or maintains the solution
            # Remove old block's coverage
            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            # Add new block's coverage
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            
            # Verify no t-subset is uncovered
            if np.min(cover_count) >= 1:
                old_idx = get_block_idx(blocks[i])
                blocks[i] = list(all_k_blocks[new_block_idx])
                if old_idx >= 0:
                    sol_block_indices.discard(old_idx)
                sol_block_indices.add(new_block_idx)
                block_ts_indices[i] = block_to_ts[new_block_idx]
                improved = True
            else:
                # Revert
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] += 1
                    
        return blocks, improved

    def local_search_2exchange(blocks, max_iters=80):
        if len(blocks) < 2:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
                
            i = random.randint(0, len(blocks) - 1)
            j = random.randint(0, len(blocks) - 1)
            if i == j:
                continue
                
            unique_ts_i = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts_i.append(ts_idx)
                    
            unique_ts_j = []
            for ts_idx in block_ts_indices[j]:
                if cover_count[ts_idx] == 1:
                    unique_ts_j.append(ts_idx)
                    
            if not unique_ts_i and not unique_ts_j:
                # Both blocks are redundant
                hi = max(i, j)
                lo = min(i, j)
                blocks.pop(hi)
                blocks.pop(lo)
                for ts_idx in block_ts_indices[hi]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_ts_indices[lo]:
                    cover_count[ts_idx] -= 1
                block_ts_indices.pop(hi)
                block_ts_indices.pop(lo)
                improved = True
                continue
                
            all_unique = unique_ts_i + unique_ts_j
            if not all_unique:
                continue
                
            candidates = set(ts_to_blocks[all_unique[0]])
            for ts_idx in all_unique[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
                    
            if candidates:
                valid = [c for c in candidates if c not in sol_block_indices]
                if valid:
                    new_block_idx = random.choice(valid)
                    
                    # Try replacing both blocks with one
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_ts_indices[j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_block_idx]:
                        cover_count[ts_idx] += 1
                        
                    if np.min(cover_count) >= 1:
                        hi = max(i, j)
                        lo = min(i, j)
                        blocks.pop(hi)
                        blocks[lo] = list(all_k_blocks[new_block_idx])
                        block_ts_indices.pop(hi)
                        block_ts_indices[lo] = block_to_ts[new_block_idx]
                        sol_block_indices.discard(get_block_idx(blocks[lo]))
                        sol_block_indices.add(new_block_idx)
                        improved = True
                    else:
                        for ts_idx in block_to_ts[new_block_idx]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_ts_indices[i]:
                            cover_count[ts_idx] += 1
                        for ts_idx in block_ts_indices[j]:
                            cover_count[ts_idx] += 1
                        
        return blocks, improved

    # Phase 3: Iterative local search
    for _ in range(150):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_iters=150)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            best_solution, improved2 = local_search_2exchange(best_solution, max_iters=50)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                # Try perturbation: remove a random block and re-greedy
                if len(best_solution) > 1 and time.time() < deadline - 20.0:
                    perturbed = list(best_solution)
                    removed_idx = random.randint(0, len(perturbed) - 1)
                    removed_block = perturbed.pop(removed_idx)
                    
                    # Check if still valid
                    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                    for b in perturbed:
                        idx = get_block_idx(b)
                        if idx >= 0:
                            for ts_idx in block_to_ts[idx]:
                                cover_count[ts_idx] += 1
                    if np.min(cover_count) >= 1:
                        best_solution = perturbed
                        best_len = len(best_solution)
                    else:
                        # Re-greedy to fill gaps
                        sol = run_greedy(random.randint(0, 10000))
                        if sol is not None:
                            combined = perturbed + sol
                            combined = optimize_redundancy(combined)
                            if len(combined) < best_len:
                                best_solution = combined
                                best_len = len(combined)

    # Final polish: run optimize_redundancy one more time
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    return best_solution
