import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for the 120s limit (or 150s kill)

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute tset indices for each block
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # Convert to sets for faster intersection checks in local search
    block_to_tsets_set = [set(lst) for lst in block_to_tsets]

    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def greedy_cover(seed):
        """
        Standard greedy set cover.
        Uses dynamic weights to prioritize rare t-sets.
        """
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        
        # Weights: prioritize rare t-sets
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts
        
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])
                
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        while uncovered_count > 0 and time.time() < deadline - 20:
            best_idx = -1
            best_gain = -1.0
            
            # Scan for best block
            # Optimization: We could use a heap, but rebuilding it is expensive.
            # Linear scan is O(N_blocks). N_blocks ~ 75k. ~300 iterations. 22M ops.
            # In Python, this is ~1-2s per greedy run. Acceptable for a few runs.
            for i in range(num_blocks):
                if not selected_mask[i] and gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
                    
            if best_idx == -1 or best_gain <= 0:
                break
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w
                            
        return selected

    def run_local_search(initial_indices, max_time_per_run=10.0, seed=42):
        """
        Local search with:
        1. Redundancy removal
        2. 1-for-1 swaps (Dual-guided: prefer blocks that cover many 'critical' or 'tight' t-sets)
        3. 2-for-1 moves (Reduction)
        4. Random kicks to escape local minima
        """
        rng = random.Random(seed)
        current = list(initial_indices)
        if not current:
            return []
        
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 50 # After this many stalls, try a kick or restart?
        
        while (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # Randomize order to avoid bias
            indices = list(range(len(current)))
            rng.shuffle(indices)
            for i in indices:
                if (time.time() - local_start) > max_time_per_run:
                    break
                if i >= len(current):
                    continue
                b_idx = current[i]
                is_redundant = True
                for ts_idx in block_to_tsets[b_idx]:
                    if cover_counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for ts_idx in block_to_tsets[b_idx]:
                        cover_counts[ts_idx] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    # Restart loop to rebuild indices safely
                    break
            
            if improved:
                no_improve_count = 0
                continue

            # Phase 2: 1-for-1 Swaps
            # Dual-guided heuristic:
            # We want to replace block B_out with B_in.
            # B_in must cover all unique t-sets of B_out.
            # Score B_in by how many "tight" t-sets (count == 1 or 2) it covers.
            # Covering tight t-sets increases robustness.
            
            swap_attempts = 0
            max_swap_attempts = 30 # Increased attempts for better exploration
            while swap_attempts < max_swap_attempts:
                if (time.time() - local_start) > max_time_per_run:
                    break
                swap_attempts += 1
                
                if not current:
                    break
                    
                i = rng.randint(0, len(current) - 1)
                b_out = current[i]
                
                # Find unique t-sets of b_out
                unique_tsets = [ts for ts in block_to_tsets[b_out] if cover_counts[ts] == 1]
                
                if not unique_tsets:
                    # This block is redundant. Should have been caught in Phase 1.
                    # Remove it.
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    break
                
                # Find candidates that cover all unique_tsets
                # Start with the rarest unique tset to minimize candidate pool
                min_count_ts = unique_tsets[0]
                min_count = len(tset_to_blocks[min_count_ts])
                for ts in unique_tsets[1:]:
                    c = len(tset_to_blocks[ts])
                    if c < min_count:
                        min_count = c
                        min_count_ts = ts
                        
                candidates = tset_to_blocks[min_count_ts]
                
                best_cand = -1
                best_score = -1.0
                
                # Precompute set of b_out's t-sets for fast checking? 
                # block_to_tsets_set[b_out] is available.
                b_out_tsets = block_to_tsets_set[b_out]
                
                for b_cand in candidates:
                    if b_cand == b_out:
                        continue
                    # Check if it covers all unique tsets
                    valid = True
                    cand_tsets = block_to_tsets_set[b_cand]
                    for ts in unique_tsets:
                        if ts not in cand_tsets:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score the candidate
                    # 1. Number of tsets with count 1 (that are NOT in b_out) covered by b_cand?
                    #    If b_cand covers a tset with count 1 that is NOT in b_out, that tset becomes count 2. Good.
                    #    If b_cand covers a tset with count 1 that IS in b_out, it becomes count 2 (since b_out removed). Good.
                    #    Actually, any tset with count 1 covered by b_cand becomes count >= 2 (unless it's the only one? No, b_cand adds 1).
                    #    Wait, if ts is in b_cand and count[ts]==1, and ts is NOT in b_out, then count becomes 2.
                    #    If ts is in b_cand and count[ts]==1, and ts IS in b_out, then count stays 1 (remove b_out, add b_cand).
                    #    
                    # 2. Number of tsets with count 2 covered by b_cand?
                    #    If count is 2, and we add b_cand, it becomes 3. More robust.
                    
                    # Let's score by:
                    # +2 for each tset with count 1 covered by b_cand (increases min count)
                    # +1 for each tset with count 2 covered by b_cand (increases robustness)
                    # -1 for each tset with count 1 in b_out that is NOT covered by b_cand? (These become 0! Wait, unique_tsets are covered by b_cand by definition. So this is 0.)
                    
                    score = 0.0
                    for ts in block_to_tsets[b_cand]:
                        c = cover_counts[ts]
                        if c == 1:
                            score += 2.0
                        elif c == 2:
                            score += 1.0
                        # If c >= 3, no change in "tightness" score, but robustness increases slightly.
                        
                    # Add small noise to break ties
                    score += rng.random() * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_cand = b_cand
                        
                if best_cand != -1:
                    # Perform swap
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_cand]:
                        cover_counts[ts_idx] += 1
                    current[i] = best_cand
                    improved = True
                    break

            if improved:
                no_improve_count = 0
                continue

            # Phase 3: 2-for-1 moves (Reduction)
            # Try to find two blocks A, B such that their union of unique t-sets
            # can be covered by a single block C.
            
            if len(current) > 2:
                pair_attempts = 0
                max_pair_attempts = 15
                while pair_attempts < max_pair_attempts:
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    pair_attempts += 1
                    
                    # Pick two distinct blocks
                    i1 = rng.randint(0, len(current) - 1)
                    i2 = rng.randint(0, len(current) - 1)
                    if i1 == i2:
                        continue
                    
                    b_out1 = current[i1]
                    b_out2 = current[i2]
                    
                    # Calculate lost tsets
                    tsets_A = block_to_tsets_set[b_out1]
                    tsets_B = block_to_tsets_set[b_out2]
                    lost_tsets = set()
                    
                    for ts in tsets_A:
                        if cover_counts[ts] == 1:
                            lost_tsets.add(ts)
                        elif cover_counts[ts] == 2 and ts in tsets_B:
                            lost_tsets.add(ts)
                    
                    for ts in tsets_B:
                        if cover_counts[ts] == 1:
                            lost_tsets.add(ts)
                        # If count==2 and in A, already handled.
                        # If count==2 and NOT in A, removing B makes count 1, not 0.
                    
                    if not lost_tsets:
                        continue

                    # Find a block C that covers all lost_tsets
                    if len(lost_tsets) > 12:
                        continue # Unlikely to find a single block covering >12 specific tsets
                    
                    # Use the rarest lost tset to find candidates
                    min_lost_ts = list(lost_tsets)[0]
                    min_count = len(tset_to_blocks[min_lost_ts])
                    for ts in list(lost_tsets)[1:]:
                        c = len(tset_to_blocks[ts])
                        if c < min_count:
                            min_count = c
                            min_lost_ts = ts
                            
                    candidates = tset_to_blocks[min_lost_ts]
                    
                    best_c = -1
                    best_add = -1.0
                    for c in candidates:
                        if c == b_out1 or c == b_out2:
                            continue
                        # Check if c covers all lost_tsets
                        valid = True
                        cand_tsets = block_to_tsets_set[c]
                        for ts in lost_tsets:
                            if ts not in cand_tsets:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Score: Maximize coverage of count-2 tsets
                        add = 0.0
                        for ts in block_to_tsets[c]:
                            if cover_counts[ts] == 2:
                                add += 1.0
                            elif cover_counts[ts] == 1:
                                add += 2.0
                                
                        add += rng.random() * 0.1
                        if add > best_add:
                            best_add = add
                            best_c = c
                    
                    if best_c != -1:
                        # Apply: Remove b_out1, b_out2, Add best_c
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b_out2]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[best_c]:
                            cover_counts[ts_idx] += 1
                        
                        # Update current list
                        if i1 < i2:
                            current.pop(i2)
                            current.pop(i1)
                        else:
                            current.pop(i1)
                            current.pop(i2)
                        current.append(best_c)
                        
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break

            if not improved:
                no_improve_count += 1
                # If stuck for a while, try a "kick": random swap even if it doesn't improve score?
                # Or just continue. The randomization in swap attempts usually helps.
                # If no_improve_count > 100, we might break and rely on other restarts?
                # For now, just continue.

        return best_solution

    best_global_solution = None
    best_global_size = float('inf')
    
    # Run multiple instances
    # We have ~115s.
    # Greedy takes ~1-2s. Local search takes the rest.
    # Let's run 5-6 full cycles.
    seeds = [42, 123, 777, 1234, 5678, 9999]
    
    for seed in seeds:
        if time.time() > deadline - 25:
            break
        t_start = time.time()
        init = greedy_cover(seed)
        if init:
            ls_time = min(15.0, deadline - t_start - 10) # Reserve 10s for final polish
            if ls_time > 3:
                sol = run_local_search(init, max_time_per_run=ls_time, seed=seed)
                if sol and len(sol) < best_global_size:
                    best_global_size = len(sol)
                    best_global_solution = sol
    
    # Final polish on the best solution found
    if best_global_solution and time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 1
        if remaining_time > 2:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time, seed=99999)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol

    if not best_global_solution:
        return []
        
    return get_solution(best_global_solution)

if __name__ == "__main__":
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
