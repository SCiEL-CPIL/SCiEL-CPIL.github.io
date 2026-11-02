import time
import random
import itertools
import numpy as np
import sys
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
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For each block, list of t-subset indices it covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(np.array(ts_list, dtype=np.int32))

    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def optimize_redundancy(blocks):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_indices.append(idx)
                cover_count[block_to_ts[idx]] += 1

        if not block_indices:
            return []

        n = len(block_indices)
        unique_counts = np.zeros(n, dtype=np.int32)
        for i, bidx in enumerate(block_indices):
            ts_arr = block_to_ts[bidx]
            unique_counts[i] = np.sum(cover_count[ts_arr] == 1)

        changed = True
        while changed:
            changed = False
            zero_mask = unique_counts == 0
            if not np.any(zero_mask):
                break
            zero_indices = np.where(zero_mask)[0]
            for pos in zero_indices:
                bidx = block_indices[pos]
                ts_arr = block_to_ts[bidx]
                cover_count[ts_arr] -= 1
                block_indices.pop(pos)
                unique_counts = np.delete(unique_counts, pos)
                changed = True
                # Recompute unique counts for remaining blocks
                for i in range(len(block_indices)):
                    ts_arr2 = block_to_ts[block_indices[i]]
                    unique_counts[i] = np.sum(cover_count[ts_arr2] == 1)
                if time.time() > deadline - 2.0:
                    break

        return [list(all_k_blocks[idx]) for idx in block_indices]

    def greedy_deterministic():
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 20.0:
                break
            max_idx = int(np.argmax(gain))
            if gain[max_idx] <= 0:
                break
            sol.append(list(all_k_blocks[max_idx]))
            is_selected[max_idx] = True
            gain[max_idx] = -1
            newly_covered = block_to_ts[max_idx][is_uncovered[block_to_ts[max_idx]]]
            if len(newly_covered) > 0:
                is_uncovered[newly_covered] = False
                for ts_idx in newly_covered:
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
        return sol

    def randomized_greedy(seed, temperature=0.0):
        rng = random.Random(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 15.0:
                return None
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            if temperature > 0 and rng.random() < 0.2:
                explore_gain = max(0, max_gain - 3)
                explore_candidates = np.where((gain >= explore_gain) & (~is_selected))[0]
                if len(explore_candidates) > 0:
                    b_idx = int(explore_candidates[rng.randint(0, len(explore_candidates) - 1)])
                else:
                    b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            newly_covered = block_to_ts[b_idx][is_uncovered[block_to_ts[b_idx]]]
            if len(newly_covered) > 0:
                is_uncovered[newly_covered] = False
                for ts_idx in newly_covered:
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        return sol

    def local_search_swap(blocks, max_rounds=15):
        """1-swap: replace a block with another that covers all its unique t-subsets."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                cover_count[block_to_ts[b_idx]] += 1

        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            round_improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    break
                old_b_idx = get_block_idx(blocks[i])
                if old_b_idx < 0:
                    continue
                unique_ts = [ts_idx for ts_idx in block_to_ts[old_b_idx] if cover_count[ts_idx] == 1]
                if not unique_ts:
                    continue
                candidates = ts_to_blocks_set[unique_ts[0]].copy()
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks_set[ts_idx])
                    if not candidates:
                        break
                if not candidates:
                    continue
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                # Pick the one that maximizes additional coverage
                best_new = None
                best_gain = -1
                for c in valid_candidates[:15]:
                    ts_arr = block_to_ts[c]
                    gain_val = np.sum(cover_count[ts_arr] == 0)
                    if gain_val > best_gain:
                        best_gain = gain_val
                        best_new = c
                if best_new is None:
                    best_new = random.choice(valid_candidates)
                new_block_idx = best_new
                new_block = list(all_k_blocks[new_block_idx])
                for ts_idx in block_to_ts[old_b_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] += 1
                sol_block_indices.discard(old_b_idx)
                sol_block_indices.add(new_block_idx)
                blocks[i] = new_block
                round_improved = True
                improved = True
            if not round_improved:
                break
        return blocks, improved

    def local_search_2exchange(blocks):
        """2-exchange: replace two blocks with one."""
        if len(blocks) < 2:
            return blocks, False
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                cover_count[block_to_ts[b_idx]] += 1
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
        improved = False
        n = len(blocks)
        for i in range(n):
            if time.time() > deadline - 3.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 3.0:
                    break
                old_i_idx = get_block_idx(blocks[i])
                old_j_idx = get_block_idx(blocks[j])
                if old_i_idx < 0 or old_j_idx < 0:
                    continue
                unique_ts_i = [ts_idx for ts_idx in block_to_ts[old_i_idx] if cover_count[ts_idx] == 1]
                unique_ts_j = [ts_idx for ts_idx in block_to_ts[old_j_idx] if cover_count[ts_idx] == 1]
                if not unique_ts_i and not unique_ts_j:
                    continue
                all_unique = unique_ts_i + unique_ts_j
                if not all_unique:
                    continue
                candidates = ts_to_blocks_set[all_unique[0]].copy()
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks_set[ts_idx])
                    if not candidates:
                        break
                if candidates:
                    valid = [c for c in candidates if c not in sol_block_indices]
                    if valid:
                        new_block_idx = random.choice(valid)
                        new_block = list(all_k_blocks[new_block_idx])
                        for ts_idx in block_to_ts[old_i_idx]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[old_j_idx]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[new_block_idx]:
                            cover_count[ts_idx] += 1
                        sol_block_indices.discard(old_i_idx)
                        sol_block_indices.discard(old_j_idx)
                        sol_block_indices.add(new_block_idx)
                        blocks.pop(j)
                        blocks[i] = new_block
                        improved = True
                        break
            if improved:
                break
        return blocks, improved

    def sa_polish(blocks, max_iters=4000):
        """Simulated annealing polish with focus on removing redundant blocks."""
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                cover_count[block_to_ts[idx]] += 1
        
        best_len = len(blocks)
        best_blocks = list(blocks)
        current_len = len(blocks)
        T = 30.0
        T_min = 0.1
        cooling = 0.999
        rng = random.Random(42)

        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            uncovered = np.where(cover_count == 0)[0]
            
            if len(uncovered) > 0 and rng.random() < 0.4:
                # Add a block to cover an uncovered t-subset
                ts_idx = int(uncovered[rng.randint(0, len(uncovered) - 1)])
                candidate_blocks = ts_to_blocks[ts_idx]
                if not candidate_blocks:
                    continue
                valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                new_idx = valid_candidates[rng.randint(0, len(valid_candidates) - 1)]
                new_covered = block_to_ts[new_idx]
                delta_uncovered = -np.sum(cover_count[new_covered] == 0)
                delta_obj = delta_uncovered * 100 + 1
                if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / T):
                    for ts_idx2 in new_covered:
                        cover_count[ts_idx2] += 1
                    sol_block_indices.add(new_idx)
                    current_len += 1
            else:
                # Try to remove or swap a block
                if not sol_block_indices:
                    continue
                block_criticality = {}
                for b_idx in sol_block_indices:
                    crit = 0
                    for ts_idx2 in block_to_ts[b_idx]:
                        if cover_count[ts_idx2] == 1:
                            crit += 1
                    block_criticality[b_idx] = crit
                
                sorted_blocks = sorted(block_criticality.items(), key=lambda x: x[1])
                if sorted_blocks[0][1] == 0:
                    # Remove redundant block
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts[old_idx]
                    for ts_idx2 in old_covered:
                        cover_count[ts_idx2] -= 1
                    sol_block_indices.discard(old_idx)
                    current_len -= 1
                elif len(sorted_blocks) > 1:
                    # Try to swap
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts[old_idx]
                    unique_ts = [ts_idx2 for ts_idx2 in old_covered if cover_count[ts_idx2] == 1]
                    if unique_ts:
                        candidates = ts_to_blocks_set[unique_ts[0]].copy()
                        for ts_idx2 in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks_set[ts_idx2])
                            if not candidates:
                                break
                        valid = [c for c in candidates if c not in sol_block_indices]
                        if valid:
                            new_idx = rng.choice(valid)
                            new_covered = block_to_ts[new_idx]
                            for ts_idx2 in old_covered:
                                cover_count[ts_idx2] -= 1
                            for ts_idx2 in new_covered:
                                cover_count[ts_idx2] += 1
                            sol_block_indices.discard(old_idx)
                            sol_block_indices.add(new_idx)
            
            # Check if better
            blocks_list = [list(all_k_blocks[idx]) for idx in sol_block_indices]
            if len(uncovered) == 0 and current_len < best_len:
                best_len = current_len
                best_blocks = list(blocks_list)
            
            T = max(T_min, T * cooling)
        
        return best_blocks

    # Main algorithm
    # Phase 1: Deterministic greedy
    best_solution = greedy_deterministic()
    best_len = len(best_solution)
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    # Phase 2: Randomized restarts
    num_runs = 40
    for i in range(num_runs):
        if time.time() > deadline - 20.0:
            break
        temp = 0.0 if i < 20 else 0.5
        sol = randomized_greedy(i, temperature=temp)
        if sol is None:
            continue
        sol = optimize_redundancy(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 3: Local search
    for _ in range(15):
        if time.time() > deadline - 8.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_rounds=8)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    # Phase 4: SA polish
    if time.time() < deadline - 5.0:
        polished = sa_polish(best_solution, max_iters=2500)
        polished = optimize_redundancy(polished)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(best_solution)

    return best_solution
