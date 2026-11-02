import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

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

    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    idx_to_block_tuple = {i: b for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        rng = random.Random(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and rng.random() < explore_prob:
                threshold = max_gain - 3
                explore_cands = np.where((gain >= threshold) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[rng.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            if len(newly_covered) > 0:
                all_blocks_to_decrement = []
                for ts_idx in newly_covered:
                    all_blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                if all_blocks_to_decrement:
                    counts = np.bincount(all_blocks_to_decrement, minlength=num_k_blocks)
                    gain -= counts
                    gain[is_selected] = -1000

        return sol

    def run_packing_greedy(seed):
        # Build a maximal t-packing first
        rng = random.Random(seed)
        is_packed = np.full(num_t_subsets, False, dtype=bool)
        packing = []
        # Randomly order blocks to break symmetry
        block_order = list(range(num_k_blocks))
        rng.shuffle(block_order)
        
        for b_idx in block_order:
            if time.time() > deadline - 10.0:
                break
            bts = block_to_ts_np[b_idx]
            # Check if any t-subset is already packed
            if not np.any(is_packed[bts]):
                packing.append(b_idx)
                is_packed[bts] = True
                if len(packing) > 300: # Heuristic limit for packing size
                    break

        # Now complete with greedy
        # Initialize state based on packing
        gain = np.full(num_k_blocks, 0, dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        
        sol_idx = list(packing)
        
        for b_idx in packing:
            is_selected[b_idx] = True
            bts = block_to_ts_np[b_idx]
            is_uncovered[bts] = False
            # Update gain for other blocks: they lose the benefit of covering these t-subsets
            # Actually, greedy gain is number of NEW t-subsets covered.
            # Since these are already covered, any block covering them gets 0 gain for those.
            # We can just recompute gain from scratch or update carefully.
            # Simpler: Just run standard greedy on the remaining uncovered sets.
            # But we need to account for the fact that packing blocks are already selected.
            
        # Recompute gain properly
        # gain[b] = count of t-subsets in b that are currently UNCOVERED
        for b_idx in range(num_k_blocks):
            if is_selected[b_idx]:
                gain[b_idx] = -1
            else:
                bts = block_to_ts_np[b_idx]
                gain[b_idx] = np.sum(is_uncovered[bts])
                
        sol = [list(all_k_blocks[i]) for i in packing]
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
            is_uncovered[newly_covered] = False
            
            all_blocks_to_decrement = []
            for ts_idx in newly_covered:
                all_blocks_to_decrement.extend(ts_to_blocks[ts_idx])
            if all_blocks_to_decrement:
                counts = np.bincount(all_blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000
                
        return sol

    def local_search(blocks, max_iters=3000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        block_ts_indices = []
        block_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
                block_indices.append(idx)
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)
                block_indices.append(-1)

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set(block_indices)
        improved_count = 0

        # Phase A: Removal
        for _pass in range(5):
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    if time.time() > deadline - 3.0:
                        return blocks, improved_count
                    ts_arr = block_ts_indices[i]
                    if np.all(cover_count[ts_arr] >= 2):
                        np.subtract.at(cover_count, ts_arr, 1)
                        blocks.pop(i)
                        block_ts_indices.pop(i)
                        block_indices.pop(i)
                        sol_block_set.discard(block_indices[i] if i < len(block_indices) else -1)
                        improved_count += 1
                        changed = True
                        break

        # Phase B: 1-Swap
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            i = random.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]
            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                block_indices.pop(i)
                improved_count += 1
                continue

            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue

            new_block_idx = random.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                block_indices[i] = new_block_idx
                sol_block_set.discard(block_indices[i])
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        # Phase C: 2-for-1 Merge
        # Try to replace two blocks with one
        merge_iters = 0
        max_merge_iters = 500
        while merge_iters < max_merge_iters:
            merge_iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) < 2:
                break
            
            i = random.randint(0, len(blocks) - 1)
            j = random.randint(0, len(blocks) - 1)
            if i == j:
                continue
                
            ts_arr_i = block_ts_indices[i]
            ts_arr_j = block_ts_indices[j]
            
            # Unique to i or j (and not others)
            # We want t-subsets s such that cover_count[s] == 2 and (s in i or s in j)
            # Actually, if cover_count[s] > 2, it's safe to remove i or j w.r.t s.
            # We need s such that it is covered by i or j, and NOT by any other block.
            # So cover_count[s] must be 1 or 2.
            # If cover_count[s] == 1, it must be in i or j.
            # If cover_count[s] == 2, it must be in exactly one of i, j (or both, but then count would be >= 2 from them, plus others).
            
            # Simpler: Compute the set of t-subsets that are uniquely covered by i or j.
            # U = { s | s in (i U j) and cover_count[s] <= 2 and (s in i or s in j) }
            # Actually, if s is in i and j, and no other, cover_count is 2.
            # If s is in i and not j, and no other, cover_count is 1.
            
            mask_i = cover_count[ts_arr_i] <= 2
            mask_j = cover_count[ts_arr_j] <= 2
            
            # We need to be careful: if s is in i and j, it's in both.
            # Let's just gather candidates for a block k that covers all "critical" t-subsets.
            critical_ts = []
            for s in set(list(ts_arr_i) + list(ts_arr_j)):
                if cover_count[s] <= 2:
                    # Check if s is covered by any block other than i and j
                    # We don't have a direct "covered by" map for blocks, only ts_to_blocks.
                    # But we know cover_count[s].
                    # If cover_count[s] == 1, it's covered by exactly one of i, j.
                    # If cover_count[s] == 2, it's covered by exactly two blocks.
                    # If s is in i and j, and no other, count is 2.
                    # If s is in i, not j, not other, count is 1.
                    # So all s in (i U j) with count <= 2 are critical.
                    critical_ts.append(s)
                    
            if not critical_ts:
                continue
                
            # Find a block k that covers all critical_ts
            # Candidates must cover critical_ts[0]
            cands = set(ts_to_blocks[int(critical_ts[0])])
            for s in critical_ts[1:]:
                cands.intersection_update(ts_to_blocks[int(s)])
                if not cands:
                    break
                    
            if not cands:
                continue
                
            # We cannot use i or j themselves as the new block (that would be no-op or invalid)
            # We want a block k such that k != i and k != j.
            # Also, we want to reduce the count, so we remove i and j and add k.
            # This is valid if k covers all critical_ts.
            # But we must ensure that removing i and j doesn't uncover any non-critical ts.
            # Non-critical ts are those with cover_count > 2.
            # Since they are covered by >= 3 blocks, removing 2 leaves >= 1. So they are safe.
            
            valid_cands = [c for c in cands if c != block_indices[i] and c != block_indices[j]]
            if not valid_cands:
                continue
                
            new_block_idx = random.choice(valid_cands)
            
            # Perform the swap
            np.subtract.at(cover_count, ts_arr_i, 1)
            np.subtract.at(cover_count, ts_arr_j, 1)
            np.add.at(cover_count, block_to_ts_np[new_block_idx], 1)
            
            if np.min(cover_count) >= 1:
                # Remove i and j, add new block
                # Order matters for indices
                if i > j:
                    i, j = j, i
                # Remove j first (higher index or just one of them)
                # Let's pop j, then i
                blocks.pop(j)
                block_ts_indices.pop(j)
                block_indices.pop(j)
                if i < len(blocks): # i index might have shifted if j < i? No, we swapped to i<j.
                    # If i < j, removing j doesn't affect i's index.
                    pass
                else:
                    # This shouldn't happen if i < j
                    pass
                    
                # Wait, if i < j:
                # Remove j: indices 0..j-1, j+1..
                # Remove i: indices 0..i-1, i+1.. (which is now i.. in the new list)
                # So:
                blocks.pop(j)
                block_ts_indices.pop(j)
                block_indices.pop(j)
                
                blocks.pop(i)
                block_ts_indices.pop(i)
                block_indices.pop(i)
                
                # Add new block
                blocks.append(list(all_k_blocks[new_block_idx]))
                block_ts_indices.append(block_to_ts_np[new_block_idx])
                block_indices.append(new_block_idx)
                
                sol_block_set.discard(block_indices[-1]) # Just in case
                sol_block_set.add(new_block_idx)
                improved_count += 2 # Removed 2, added 1, net -1 block. But "improvement" count is arbitrary.
            else:
                # Revert
                np.add.at(cover_count, ts_arr_i, 1)
                np.add.at(cover_count, ts_arr_j, 1)
                np.subtract.at(cover_count, block_to_ts_np[new_block_idx], 1)

        # Final cleanup
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    block_indices.pop(i)
                    improved_count += 1
                    changed = True
                    break

        return blocks, improved_count

    # Initial solutions
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    best_solution, _ = local_search(best_solution, max_iters=2000)
    best_len = len(best_solution)

    # Packing Greedy
    sol_p = run_packing_greedy(1)
    if sol_p is not None:
        sol_p, _ = local_search(sol_p, max_iters=2000)
        if len(sol_p) < best_len:
            best_len = len(sol_p)
            best_solution = sol_p

    # Randomized Greedy
    for i in range(1, 10):
        if time.time() > deadline - 30.0:
            break
        sol = run_greedy(i, explore_prob=0.1)
        if sol is not None:
            sol, _ = local_search(sol, max_iters=1000)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    # Iterated Local Search
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        # Perturb
        if len(best_solution) > 1:
            i = random.randint(0, len(best_solution) - 1)
            # Remove block i and re-add a random one? Or just remove and let local search fix it?
            # Removing a block breaks coverage. We should swap it.
            # Let's just run local search on a copy with a random swap
            perturbed = [list(b) for b in best_solution]
            if len(perturbed) > 0:
                idx = random.randint(0, len(perturbed) - 1)
                # Find a random block to swap with
                # Just pick a random block from all_k_blocks
                new_b = random.choice(all_k_blocks)
                perturbed[idx] = list(new_b)
                
            perturbed, _ = local_search(perturbed, max_iters=500)
            if len(perturbed) < best_len:
                best_len = len(perturbed)
                best_solution = perturbed

    # Final Polish
    if time.time() < deadline - 5.0:
        best_solution, _ = local_search(best_solution, max_iters=3000)
        best_len = len(best_solution)

    return best_solution
