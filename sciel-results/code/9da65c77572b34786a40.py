import time
import random
import itertools
import numpy as np
import sys
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

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
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
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
        block_to_ts.append(np.array(ts_list, dtype=np.int32))
        
    # Convert ts_to_blocks to numpy arrays for faster intersection logic if needed, 
    # but sets are better for intersection.
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]
    
    # Map block tuple to index for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # 1. Greedy Construction
    def greedy_construct():
        # gain[b] = number of uncovered t-subsets covered by block b
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        # Use a heap for max-heap (store negative gain)
        # ( -gain, block_index )
        heap = [(-gain[i], i) for i in range(num_k_blocks)]
        heapq.heapify(heap)
        
        while True:
            if time.time() > deadline - 20.0:
                break
            if not heap:
                break
            neg_gain, b_idx = heapq.heappop(heap)
            if is_selected[b_idx]:
                continue
            if -neg_gain <= 0:
                break
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            
            # Update gains for blocks that cover newly covered t-subsets
            newly_covered = block_to_ts[b_idx][is_uncovered[block_to_ts[b_idx]]]
            if len(newly_covered) > 0:
                is_uncovered[newly_covered] = False
                # For each newly covered t-subset, decrement gain of all blocks covering it
                # This is expensive. Optimization: only push updated gains to heap.
                # We don't need to update all gains immediately, just mark them dirty or push new values.
                # Since we are doing lazy deletion, we can just push new (gain, idx) pairs.
                # But we need to know the current gain.
                # Let's maintain an array `current_gain`.
                # Actually, the standard greedy update is:
                # For each t in newly_covered:
                #   for each b in ts_to_blocks[t]:
                #     if not selected[b]:
                #       gain[b] -= 1
                #       push (-gain[b], b) to heap
                for ts_idx in newly_covered:
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            heapq.heappush(heap, (-gain[nb_idx], nb_idx))
                            
        return sol

    # 2. Redundancy Removal
    def remove_redundancy(blocks):
        if not blocks:
            return []
        block_indices = [get_block_idx(b) for b in blocks]
        # Filter out invalid indices just in case
        block_indices = [i for i in block_indices if i >= 0]
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for bidx in block_indices:
            cover_count[block_to_ts[bidx]] += 1
            
        changed = True
        while changed:
            if time.time() > deadline - 10.0:
                break
            changed = False
            # Find blocks with no unique t-subsets
            to_remove = []
            for bidx in block_indices:
                ts_arr = block_to_ts[bidx]
                # Check if any ts in ts_arr has cover_count == 1
                # vectorized check
                if not np.any(cover_count[ts_arr] == 1):
                    to_remove.append(bidx)
                    
            if not to_remove:
                break
                
            for bidx in to_remove:
                cover_count[block_to_ts[bidx]] -= 1
                block_indices.remove(bidx)
                changed = True
                
        return [list(all_k_blocks[idx]) for idx in block_indices]

    # 3. Local Search: 1-Swap
    def local_search_1swap(blocks, max_rounds=50):
        if not blocks:
            return blocks, False
        
        block_indices = [get_block_idx(b) for b in blocks]
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for bidx in block_indices:
            cover_count[block_to_ts[bidx]] += 1
            
        sol_set = set(block_indices)
        improved = False
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            round_improved = False
            # Shuffle to avoid local minima bias
            order = list(range(len(block_indices)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                old_idx = block_indices[i]
                ts_arr = block_to_ts[old_idx]
                unique_ts = ts_arr[cover_count[ts_arr] == 1]
                
                if len(unique_ts) == 0:
                    # This block is redundant, remove it
                    cover_count[ts_arr] -= 1
                    block_indices.pop(i)
                    sol_set.discard(old_idx)
                    round_improved = True
                    improved = True
                    # Adjust order indices if needed, but we are iterating over a copy
                    # Actually, popping from list while iterating is tricky. 
                    # Let's rebuild list if we pop.
                    # Or better: mark for removal and filter later.
                    # For simplicity, let's just break and restart if we remove.
                    break
                else:
                    # Find candidate blocks that cover all unique_ts
                    if len(unique_ts) > 0:
                        candidates = ts_to_blocks_set[unique_ts[0]].copy()
                        for ts_idx in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks_set[ts_idx])
                            if not candidates:
                                break
                        
                        if candidates:
                            valid_candidates = [c for c in candidates if c not in sol_set]
                            if valid_candidates:
                                # Pick best candidate: max additional coverage
                                best_cand = None
                                best_gain = -1
                                for c in valid_candidates[:10]: # Limit search
                                    c_ts = block_to_ts[c]
                                    # Gain = number of ts covered by c that are not covered by old_idx?
                                    # No, gain is number of ts with cover_count 0 that c covers.
                                    # Since we are swapping, the unique_ts of old_idx will be covered by c.
                                    # The new gain is how many *other* ts c covers that were 0.
                                    gain_val = np.sum(cover_count[c_ts] == 0)
                                    if gain_val > best_gain:
                                        best_gain = gain_val
                                        best_cand = c
                                        
                                if best_cand is not None:
                                    # Perform swap
                                    new_idx = best_cand
                                    # Remove old
                                    cover_count[ts_arr] -= 1
                                    # Add new
                                    new_ts = block_to_ts[new_idx]
                                    cover_count[new_ts] += 1
                                    
                                    block_indices[i] = new_idx
                                    sol_set.discard(old_idx)
                                    sol_set.add(new_idx)
                                    round_improved = True
                                    improved = True
                                    
            if not round_improved:
                break
            if improved:
                # Restart loop to catch cascading improvements
                # But we need to update block_indices list if we popped
                # Let's just break and return, caller will re-optimize
                break
                
        # Reconstruct blocks
        return [list(all_k_blocks[idx]) for idx in block_indices], improved

    # 4. Local Search: 2-Exchange (Replace 2 blocks with 1)
    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False
            
        block_indices = [get_block_idx(b) for b in blocks]
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for bidx in block_indices:
            cover_count[block_to_ts[bidx]] += 1
            
        sol_set = set(block_indices)
        improved = False
        
        # Try random pairs
        n = len(block_indices)
        for _ in range(20): # Limit attempts
            if time.time() > deadline - 3.0:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            idx_i = block_indices[i]
            idx_j = block_indices[j]
            
            ts_i = block_to_ts[idx_i]
            ts_j = block_to_ts[idx_j]
            
            # Unique ts covered by i or j
            # A ts is unique to the pair if cover_count[ts] == 1 and ts is in i or j
            # Actually, if we remove i and j, a ts is lost if cover_count[ts] == 1 and ts in i, 
            # OR cover_count[ts] == 2 and ts in i and j (so it becomes 0).
            # Wait, if ts is in i and j, cover_count is at least 2. If we remove both, it drops by 2.
            # If cover_count was 2, it becomes 0. So it's "unique" to the pair in the sense that it's lost.
            # If cover_count was 3, it becomes 1. Not lost.
            
            # So, lost_ts = { ts in i U j | cover_count[ts] - (1 if ts in i else 0) - (1 if ts in j else 0) == 0 }
            # Simpler: 
            # 1. Temporarily remove i and j
            cover_count[ts_i] -= 1
            cover_count[ts_j] -= 1
            
            lost_ts = np.where((cover_count == 0) & ((block_to_ts[idx_i] == np.arange(num_t_subsets)) | (block_to_ts[idx_j] == np.arange(num_t_subsets))))[0]
            # This vectorized op is slow. Let's do it manually.
            # Restore cover count
            cover_count[ts_i] += 1
            cover_count[ts_j] += 1
            
            # Manual check
            lost_ts_list = []
            for ts_idx in ts_i:
                if cover_count[ts_idx] == 1:
                    lost_ts_list.append(ts_idx)
            for ts_idx in ts_j:
                # If ts_idx was also in ts_i, cover_count was at least 2.
                # If cover_count is 2, and it's in both, removing both makes it 0.
                # If cover_count is 1, it can't be in both.
                if cover_count[ts_idx] == 2 and ts_idx in ts_i:
                    # It was 2, now 0
                    if ts_idx not in lost_ts_list:
                        lost_ts_list.append(ts_idx)
                elif cover_count[ts_idx] == 1:
                    if ts_idx not in lost_ts_list:
                        lost_ts_list.append(ts_idx)
                        
            if not lost_ts_list:
                # Both blocks are redundant
                cover_count[ts_i] -= 1
                cover_count[ts_j] -= 1
                sol_set.discard(idx_i)
                sol_set.discard(idx_j)
                block_indices.pop(j)
                block_indices.pop(i)
                improved = True
                break
                
            # Find a block c that covers all lost_ts_list
            if lost_ts_list:
                candidates = ts_to_blocks_set[lost_ts_list[0]].copy()
                for ts_idx in lost_ts_list[1:]:
                    candidates.intersection_update(ts_to_blocks_set[ts_idx])
                    if not candidates:
                        break
                        
                if candidates:
                    valid = [c for c in candidates if c not in sol_set]
                    if valid:
                        new_idx = random.choice(valid)
                        # Perform exchange
                        cover_count[ts_i] -= 1
                        cover_count[ts_j] -= 1
                        new_ts = block_to_ts[new_idx]
                        cover_count[new_ts] += 1
                        
                        sol_set.discard(idx_i)
                        sol_set.discard(idx_j)
                        sol_set.add(new_idx)
                        
                        block_indices.pop(j)
                        block_indices[i] = new_idx
                        improved = True
                        break
                        
        return [list(all_k_blocks[idx]) for idx in block_indices], improved

    # 5. Simulated Annealing
    def sa_polish(blocks, max_iters=5000):
        if not blocks:
            return blocks
            
        block_indices = [get_block_idx(b) for b in blocks]
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for bidx in block_indices:
            cover_count[block_to_ts[bidx]] += 1
            
        sol_set = set(block_indices)
        best_len = len(block_indices)
        best_blocks = list(block_indices)
        current_len = len(block_indices)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            # Calculate objective: len * 1000 + uncovered_count
            uncovered = np.sum(cover_count == 0)
            obj = current_len * 1000 + uncovered
            
            # Move:
            # 1. Remove a block (if it doesn't break coverage)
            # 2. Add a block (if it helps)
            # 3. Swap a block
            
            if random.random() < 0.3 and sol_set:
                # Try remove
                candidates_to_remove = [b for b in sol_set if not np.any(cover_count[block_to_ts[b]] == 1)]
                if candidates_to_remove:
                    old_idx = random.choice(candidates_to_remove)
                    ts_arr = block_to_ts[old_idx]
                    # Check if removal is safe
                    # If any ts in ts_arr has count 1, removal breaks coverage
                    if not np.any(cover_count[ts_arr] == 1):
                        cover_count[ts_arr] -= 1
                        sol_set.discard(old_idx)
                        block_indices.remove(old_idx)
                        current_len -= 1
                        if current_len < best_len and uncovered == 0:
                            best_len = current_len
                            best_blocks = list(block_indices)
                else:
                    # Try swap
                    old_idx = random.choice(list(sol_set))
                    ts_arr = block_to_ts[old_idx]
                    unique_ts = ts_arr[cover_count[ts_arr] == 1]
                    if unique_ts:
                        candidates = ts_to_blocks_set[unique_ts[0]].copy()
                        for ts_idx in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks_set[ts_idx])
                        valid = [c for c in candidates if c not in sol_set]
                        if valid:
                            new_idx = random.choice(valid)
                            # Delta objective
                            # Remove old, add new
                            # New uncovered = old uncovered - (new covers unique) + (old unique not covered by new)
                            # Since new covers all unique_ts of old, uncovered decreases by 0 (they stay covered)
                            # But new might cover other previously uncovered ts.
                            new_ts = block_to_ts[new_idx]
                            delta_uncovered = -np.sum((cover_count[new_ts] == 0) & (new_ts != ts_arr))
                            # This is an approximation. Let's just calculate new obj.
                            # Apply move
                            cover_count[ts_arr] -= 1
                            cover_count[new_ts] += 1
                            sol_set.discard(old_idx)
                            sol_set.add(new_idx)
                            block_indices[block_indices.index(old_idx)] = new_idx
                            
                            new_uncovered = np.sum(cover_count == 0)
                            new_obj = current_len * 1000 + new_uncovered
                            if new_obj < obj or random.random() < np.exp(-(new_obj - obj) / T):
                                if new_len < best_len and new_uncovered == 0:
                                    best_len = new_len
                                    best_blocks = list(block_indices)
                            else:
                                # Revert
                                cover_count[new_ts] -= 1
                                cover_count[ts_arr] += 1
                                sol_set.discard(new_idx)
                                sol_set.add(old_idx)
                                block_indices[block_indices.index(new_idx)] = old_idx
            else:
                # Try add
                # Find uncovered ts
                uncovered_ts = np.where(cover_count == 0)[0]
                if len(uncovered_ts) > 0:
                    ts_idx = uncovered_ts[random.randint(0, len(uncovered_ts)-1)]
                    candidates = ts_to_blocks[ts_idx]
                    valid = [c for c in candidates if c not in sol_set]
                    if valid:
                        new_idx = random.choice(valid)
                        new_ts = block_to_ts[new_idx]
                        cover_count[new_ts] += 1
                        sol_set.add(new_idx)
                        block_indices.append(new_idx)
                        current_len += 1
                        
                        new_uncovered = np.sum(cover_count == 0)
                        new_obj = current_len * 1000 + new_uncovered
                        if new_obj < obj or random.random() < np.exp(-(new_obj - obj) / T):
                            if new_len < best_len and new_uncovered == 0:
                                best_len = new_len
                                best_blocks = list(block_indices)
                        else:
                            # Revert
                            cover_count[new_ts] -= 1
                            sol_set.discard(new_idx)
                            block_indices.pop()
                            current_len -= 1
                            
            T = max(T_min, T * cooling)
            
        return [list(all_k_blocks[idx]) for idx in best_blocks]

    # Main
    # Phase 1: Greedy
    sol = greedy_construct()
    sol = remove_redundancy(sol)
    best_len = len(sol)
    
    # Phase 2: Local Search
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        sol, imp1 = local_search_1swap(sol, max_rounds=10)
        if imp1:
            sol = remove_redundancy(sol)
            if len(sol) < best_len:
                best_len = len(sol)
        
        sol, imp2 = local_search_2exchange(sol)
        if imp2:
            sol = remove_redundancy(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                
        if not imp1 and not imp2:
            break
            
    # Phase 3: SA Polish
    if time.time() < deadline - 5.0:
        sol = sa_polish(sol, max_iters=2000)
        sol = remove_redundancy(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            
    return sol
