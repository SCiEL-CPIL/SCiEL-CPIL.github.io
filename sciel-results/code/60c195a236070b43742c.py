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

    # Convert to numpy arrays for fast operations
    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    # Sets for fast intersection
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
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

            if explore_prob > 0 and random.random() < explore_prob:
                # Explore: pick from blocks with gain >= max_gain - 3
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            # Identify newly covered t-subsets
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            # Vectorized gain update using bincount
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])

            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run multiple randomized greedy with varying exploration
    num_runs = 120
    for i in range(1, num_runs):
        if time.time() > deadline - 20.0:
            break

        # Vary exploration probability
        explore = 0.0 if i < 40 else (0.1 if i < 80 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
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

    def local_search_swap(blocks):
        if not blocks:
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
        for i in range(len(blocks)):
            if time.time() > deadline - 3.0:
                return blocks, improved

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

            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue

            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])

            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)

            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)

            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1

            improved = True

        return blocks, improved

    def local_search_2exchange(blocks):
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
        n = len(blocks)

        for i in range(n):
            if time.time() > deadline - 3.0:
                return blocks, improved
            for j in range(i + 1, n):
                if time.time() > deadline - 3.0:
                    return blocks, improved

                old_i_idx = get_block_idx(blocks[i])
                old_j_idx = get_block_idx(blocks[j])
                if old_i_idx < 0 or old_j_idx < 0:
                    continue

                unique_ts_i = [ts_idx for ts_idx in block_ts_indices[i] if cover_count[ts_idx] == 1]
                unique_ts_j = [ts_idx for ts_idx in block_ts_indices[j] if cover_count[ts_idx] == 1]

                if not unique_ts_i and not unique_ts_j:
                    continue

                all_unique = unique_ts_i + unique_ts_j
                if not all_unique:
                    continue

                # Try single replacement
                candidates = set(ts_to_blocks[all_unique[0]])
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break

                if candidates:
                    valid = [c for c in candidates if c not in sol_block_indices]
                    if valid:
                        new_block_idx = random.choice(valid)
                        new_block = list(all_k_blocks[new_block_idx])

                        for ts_idx in block_ts_indices[i]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_ts_indices[j]:
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

                # Try two separate replacements
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

                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_ts_indices[j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_i_idx]:
                        cover_count[ts_idx] += 1
                    for ts_idx in block_to_ts[new_j_idx]:
                        cover_count[ts_idx] += 1

                    sol_block_indices.discard(old_i_idx)
                    sol_block_indices.discard(old_j_idx)
                    sol_block_indices.add(new_i_idx)
                    sol_block_indices.add(new_j_idx)

                    blocks[i] = list(all_k_blocks[new_i_idx])
                    blocks[j] = list(all_k_blocks[new_j_idx])
                    improved = True
                    break
            if improved:
                break

        return blocks, improved

    def sa_polish(blocks, max_iters=2000):
        if not blocks:
            return blocks

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1

        best_len = len(blocks)
        best_blocks = list(blocks)
        current_len = len(blocks)

        T = 100.0
        T_min = 1.0
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break

            uncovered = np.where(cover_count == 0)[0]
            current_obj = len(uncovered) * 1000 + current_len

            if len(uncovered) > 0 and random.random() < 0.6:
                ts_idx = int(uncovered[random.randint(0, len(uncovered) - 1)])
                candidate_blocks = ts_to_blocks[ts_idx]
                if not candidate_blocks:
                    continue

                if random.random() < 0.5 and sol_block_indices:
                    existing_list = list(sol_block_indices)
                    old_idx = existing_list[random.randint(0, len(existing_list) - 1)]

                    valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                    if not valid_candidates:
                        continue

                    new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]

                    old_covered = block_to_ts_np[old_idx]
                    new_covered = block_to_ts_np[new_idx]

                    delta_uncovered = 0
                    for ts_idx2 in old_covered:
                        if cover_count[ts_idx2] == 1:
                            delta_uncovered += 1
                    for ts_idx2 in new_covered:
                        if cover_count[ts_idx2] == 0:
                            delta_uncovered -= 1

                    delta_obj = delta_uncovered * 1000

                    if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                        for ts_idx2 in old_covered:
                            cover_count[ts_idx2] -= 1
                        for ts_idx2 in new_covered:
                            cover_count[ts_idx2] += 1
                        sol_block_indices.discard(old_idx)
                        sol_block_indices.add(new_idx)
                else:
                    valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                    if not valid_candidates:
                        continue

                    new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    new_covered = block_to_ts_np[new_idx]

                    delta_uncovered = 0
                    for ts_idx2 in new_covered:
                        if cover_count[ts_idx2] == 0:
                            delta_uncovered -= 1

                    delta_obj = delta_uncovered * 1000 + 1

                    if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                        for ts_idx2 in new_covered:
                            cover_count[ts_idx2] += 1
                        sol_block_indices.add(new_idx)
                        current_len += 1
            else:
                if not sol_block_indices:
                    continue

                block_criticality = {}
                for b_idx in sol_block_indices:
                    crit = 0
                    for ts_idx2 in block_to_ts_np[b_idx]:
                        if cover_count[ts_idx2] == 1:
                            crit += 1
                    block_criticality[b_idx] = crit

                sorted_blocks = sorted(block_criticality.items(), key=lambda x: x[1])
                if sorted_blocks[0][1] == 0:
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts_np[old_idx]

                    for ts_idx2 in old_covered:
                        cover_count[ts_idx2] -= 1
                    sol_block_indices.discard(old_idx)
                    current_len -= 1
                elif len(sorted_blocks) > 1:
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts_np[old_idx]
                    unique_ts = [ts_idx2 for ts_idx2 in old_covered if cover_count[ts_idx2] == 1]

                    if unique_ts:
                        candidates = set(ts_to_blocks[unique_ts[0]])
                        for ts_idx2 in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks[ts_idx2])
                            if not candidates:
                                break

                        valid = [c for c in candidates if c not in sol_block_indices]
                        if valid:
                            new_idx = random.choice(valid)
                            new_covered = block_to_ts_np[new_idx]

                            for ts_idx2 in old_covered:
                                cover_count[ts_idx2] -= 1
                            for ts_idx2 in new_covered:
                                cover_count[ts_idx2] += 1
                            sol_block_indices.discard(old_idx)
                            sol_block_indices.add(new_idx)

            blocks = [list(all_k_blocks[idx]) for idx in sol_block_indices]
            blocks = optimize_redundancy(blocks)
            current_len = len(blocks)

            if len(uncovered) == 0 and current_len < best_len:
                best_len = current_len
                best_blocks = list(blocks)

            T = max(T_min, T * cooling)

        return best_blocks

    # Run local search iterations with cached state
    def build_cover_state(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                block_ts_indices.append(block_to_ts[idx])
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
                for ts_idx in ts_list:
                    cover_count[ts_idx] += 1
        return cover_count, block_ts_indices, sol_block_indices

    def optimize_redundancy_fast(blocks, cover_count, block_ts_indices, sol_block_indices):
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 3.0:
                break
            for i in range(len(blocks)):
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    old_idx = get_block_idx(blocks[i])
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    if old_idx >= 0:
                        sol_block_indices.discard(old_idx)
                    changed = True
                    break
        return cover_count, block_ts_indices, sol_block_indices

    # Build initial state
    cover_count, block_ts_indices, sol_block_indices = build_cover_state(best_solution)
    cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
        best_solution, cover_count, block_ts_indices, sol_block_indices)
    if len(best_solution) < best_len:
        best_len = len(best_solution)

    # Local search with shared state
    for _ in range(60):
        if time.time() > deadline - 8.0:
            break

        # Single swap
        improved = False
        n = len(best_solution)
        for i in range(n):
            if time.time() > deadline - 4.0:
                break
            unique_ts = [ts_idx for ts_idx in block_ts_indices[i] if cover_count[ts_idx] == 1]
            if not unique_ts:
                continue
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            if not candidates:
                continue
            valid = [c for c in candidates if c not in sol_block_indices]
            if not valid:
                continue
            new_idx = random.choice(valid)
            old_idx = get_block_idx(best_solution[i])

            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_idx]:
                cover_count[ts_idx] += 1
            if old_idx >= 0:
                sol_block_indices.discard(old_idx)
            sol_block_indices.add(new_idx)
            best_solution[i] = list(all_k_blocks[new_idx])
            block_ts_indices[i] = block_to_ts[new_idx]
            improved = True
            break

        if not improved:
            # Two-exchange: remove one, add two
            n = len(best_solution)
            for i in range(n):
                if time.time() > deadline - 4.0:
                    break
                unique_ts_i = [ts_idx for ts_idx in block_ts_indices[i] if cover_count[ts_idx] == 1]
                if not unique_ts_i:
                    continue
                # Try to replace with a single block that covers all unique
                candidates = set(ts_to_blocks[unique_ts_i[0]])
                for ts_idx in unique_ts_i[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                valid = [c for c in candidates if c not in sol_block_indices]
                if valid:
                    new_idx = random.choice(valid)
                    old_idx = get_block_idx(best_solution[i])
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_idx]:
                        cover_count[ts_idx] += 1
                    if old_idx >= 0:
                        sol_block_indices.discard(old_idx)
                    sol_block_indices.add(new_idx)
                    best_solution[i] = list(all_k_blocks[new_idx])
                    block_ts_indices[i] = block_to_ts[new_idx]
                    improved = True
                    break

        if improved:
            cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
                best_solution, cover_count, block_ts_indices, sol_block_indices)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # Try removing a redundant block
            removed = False
            for i in range(len(best_solution) - 1, -1, -1):
                if time.time() > deadline - 4.0:
                    break
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    old_idx = get_block_idx(best_solution[i])
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    best_solution.pop(i)
                    block_ts_indices.pop(i)
                    if old_idx >= 0:
                        sol_block_indices.discard(old_idx)
                    removed = True
                    break
            if not removed:
                break

    # Final SA polish with state tracking
    if time.time() < deadline - 8.0:
        # Rebuild state from current best
        cover_count, block_ts_indices, sol_block_indices = build_cover_state(best_solution)
        cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
            best_solution, cover_count, block_ts_indices, sol_block_indices)

        best_len_sa = len(best_solution)
        best_blocks_sa = list(best_solution)
        current_len = len(best_solution)

        T = 50.0
        T_min = 0.5
        cooling = 0.998
        max_sa_iters = 3000

        for it in range(max_sa_iters):
            if time.time() > deadline - 3.0:
                break

            uncovered = np.where(cover_count == 0)[0]

            if len(uncovered) > 0 and random.random() < 0.5:
                # Add a block to cover an uncovered t-subset
                ts_idx = int(uncovered[random.randint(0, len(uncovered) - 1)])
                candidate_blocks = ts_to_blocks[ts_idx]
                if not candidate_blocks:
                    continue
                valid = [c for c in candidate_blocks if c not in sol_block_indices]
                if not valid:
                    continue
                new_idx = valid[random.randint(0, len(valid) - 1)]
                new_covered = block_to_ts_np[new_idx]

                delta_uncovered = 0
                for ts_idx2 in new_covered:
                    if cover_count[ts_idx2] == 0:
                        delta_uncovered -= 1

                delta_obj = delta_uncovered * 1000 + 1
                if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                    for ts_idx2 in new_covered:
                        cover_count[ts_idx2] += 1
                    sol_block_indices.add(new_idx)
                    best_solution.append(list(all_k_blocks[new_idx]))
                    block_ts_indices.append(block_to_ts[new_idx])
                    current_len += 1
            else:
                # Remove or swap a low-criticality block
                if not sol_block_indices:
                    continue

                # Find block with fewest unique t-subsets
                min_crit = float('inf')
                min_b_idx = -1
                for b_idx in sol_block_indices:
                    crit = 0
                    for ts_idx2 in block_to_ts_np[b_idx]:
                        if cover_count[ts_idx2] == 1:
                            crit += 1
                    if crit < min_crit:
                        min_crit = crit
                        min_b_idx = b_idx

                if min_b_idx < 0:
                    continue

                b_idx = min_b_idx
                old_covered = block_to_ts_np[b_idx]

                if min_crit == 0:
                    # Safe removal
                    for ts_idx2 in old_covered:
                        cover_count[ts_idx2] -= 1
                    sol_block_indices.discard(b_idx)
                    # Find and remove from lists
                    for i in range(len(best_solution)):
                        if get_block_idx(best_solution[i]) == b_idx:
                            best_solution.pop(i)
                            block_ts_indices.pop(i)
                            break
                    current_len -= 1
                else:
                    # Try swap
                    unique_ts = [ts_idx2 for ts_idx2 in old_covered if cover_count[ts_idx2] == 1]
                    if unique_ts:
                        candidates = set(ts_to_blocks[unique_ts[0]])
                        for ts_idx2 in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks[ts_idx2])
                            if not candidates:
                                break
                        valid = [c for c in candidates if c not in sol_block_indices]
                        if valid:
                            new_idx = random.choice(valid)
                            new_covered = block_to_ts_np[new_idx]

                            delta_uncovered = 0
                            for ts_idx2 in old_covered:
                                if cover_count[ts_idx2] == 1:
                                    delta_uncovered += 1
                            for ts_idx2 in new_covered:
                                if cover_count[ts_idx2] == 0:
                                    delta_uncovered -= 1

                            delta_obj = delta_uncovered * 1000
                            if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                                for ts_idx2 in old_covered:
                                    cover_count[ts_idx2] -= 1
                                for ts_idx2 in new_covered:
                                    cover_count[ts_idx2] += 1
                                sol_block_indices.discard(b_idx)
                                sol_block_indices.add(new_idx)
                                for i in range(len(best_solution)):
                                    if get_block_idx(best_solution[i]) == b_idx:
                                        best_solution[i] = list(all_k_blocks[new_idx])
                                        block_ts_indices[i] = block_to_ts[new_idx]
                                        break

            # Periodic redundancy cleanup
            if it % 50 == 0:
                cover_count, block_ts_indices, sol_block_indices = optimize_redundancy_fast(
                    best_solution, cover_count, block_ts_indices, sol_block_indices)
                current_len = len(best_solution)

            if len(uncovered) == 0 and current_len < best_len_sa:
                best_len_sa = current_len
                best_blocks_sa = list(best_solution)

            T = max(T_min, T * cooling)

        if best_len_sa < best_len:
            best_solution = best_blocks_sa
            best_len = best_len_sa

    return best_solution
