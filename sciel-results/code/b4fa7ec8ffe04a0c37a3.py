import time
import random
import itertools
import numpy as np

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
    num_t = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build block-to-ts and ts-to-blocks
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t)]
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

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        rng = random.Random(seed)
        gain = np.full(num_k, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.full(num_k, False, dtype=bool)
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
                threshold = max(0, max_gain - 2)
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
                    counts = np.bincount(all_blocks_to_decrement, minlength=num_k)
                    gain -= counts
                    gain[is_selected] = -1000

        return sol

    def local_search(blocks, max_iters=5000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

        cover_count = np.zeros(num_t, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)

        improved_count = 0

        # Phase A: Remove redundant blocks
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
                    if i < len(blocks):
                        idx = get_block_idx(blocks[i])
                    else:
                        idx = -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    changed = True
                    break

        # Phase B: Local search with 1-exchange and 2-for-1
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
                improved_count += 1
                continue

            # Try 1-exchange
            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                # Try 2-for-1
                if len(blocks) >= 2:
                    j = random.randint(0, len(blocks) - 1)
                    if j == i:
                        j = (j + 1) % len(blocks)

                    ts_arr_j = block_ts_indices[j]
                    unique_mask_j = cover_count[ts_arr_j] == 1
                    unique_ts_j = ts_arr_j[unique_mask_j]

                    if len(unique_ts_j) == 0:
                        np.subtract.at(cover_count, ts_arr_j, 1)
                        blocks.pop(j)
                        block_ts_indices.pop(j)
                        if j < i:
                            i -= 1
                        improved_count += 1
                        continue

                    all_unique = np.unique(np.concatenate([unique_ts, unique_ts_j]))
                    if len(all_unique) > 0:
                        cand_set = set(ts_to_blocks[int(all_unique[0])])
                        for ts_idx in all_unique[1:]:
                            cand_set.intersection_update(ts_to_blocks[int(ts_idx)])
                            if not cand_set:
                                break

                        if cand_set:
                            valid_cands = [c for c in cand_set if c not in sol_block_set]
                            if valid_cands:
                                new_idx1 = random.choice(valid_cands)
                                remaining = [c for c in valid_cands if c != new_idx1]
                                if remaining:
                                    new_idx2 = random.choice(remaining)
                                else:
                                    new_idx2 = new_idx1

                                if new_idx1 != new_idx2:
                                    ts_arr_n1 = block_to_ts_np[new_idx1]
                                    ts_arr_n2 = block_to_ts_np[new_idx2]

                                    np.subtract.at(cover_count, ts_arr_i, 1)
                                    np.subtract.at(cover_count, ts_arr_j, 1)
                                    np.add.at(cover_count, ts_arr_n1, 1)
                                    np.add.at(cover_count, ts_arr_n2, 1)

                                    if np.min(cover_count) >= 1:
                                        blocks.pop(j)
                                        block_ts_indices.pop(j)
                                        if j < i:
                                            i -= 1
                                        blocks.pop(i)
                                        block_ts_indices.pop(i)

                                        blocks.append(list(all_k_blocks[new_idx1]))
                                        block_ts_indices.append(ts_arr_n1)
                                        blocks.append(list(all_k_blocks[new_idx2]))
                                        block_ts_indices.append(ts_arr_n2)

                                        sol_block_set.discard(get_block_idx(blocks[-2]))
                                        sol_block_set.discard(get_block_idx(blocks[-1]))
                                        sol_block_set.add(new_idx1)
                                        sol_block_set.add(new_idx2)
                                        improved_count += 1
                                    else:
                                        np.subtract.at(cover_count, ts_arr_n1, 1)
                                        np.subtract.at(cover_count, ts_arr_n2, 1)
                                        np.add.at(cover_count, ts_arr_i, 1)
                                        np.add.at(cover_count, ts_arr_j, 1)
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
                old_idx = get_block_idx(blocks[i])
                if old_idx >= 0:
                    sol_block_set.discard(old_idx)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        return blocks, improved_count

    # Initial greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    best_solution, _ = local_search(best_solution, max_iters=3000)
    best_len = len(best_solution)

    # Run multiple greedy + local search
    for i in range(1, 80):
        if time.time() > deadline - 50.0:
            break
        explore = 0.0 if i < 20 else (0.1 if i < 40 else 0.15)
        sol = run_greedy(i, explore_prob=explore)
        if sol is None:
            continue
        sol, _ = local_search(sol, max_iters=2000)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Iterated local search with perturbation
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break

        # Perturbation: remove 1-3 random blocks, then add blocks to restore coverage
        perturbed = list(best_solution)
        num_remove = min(random.randint(1, 3), len(perturbed) - 1)
        indices_to_remove = set(random.sample(range(len(perturbed)), num_remove))

        # Compute coverage after removal
        remaining = [perturbed[i] for i in range(len(perturbed)) if i not in indices_to_remove]
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in remaining:
            idx = get_block_idx(b)
            if idx >= 0:
                np.add.at(cover_count, block_to_ts_np[idx], 1)
            else:
                for ts in itertools.combinations(b, t):
                    ts_idx = ts_index_map[ts]
                    cover_count[ts_idx] += 1

        # Find uncovered ts and add blocks to cover them
        uncovered = np.where(cover_count == 0)[0]
        for ts_idx in uncovered:
            if time.time() > deadline - 5.0:
                break
            cand_blocks = ts_to_blocks[int(ts_idx)]
            # Pick block that covers most uncovered
            best_b = -1
            best_gain = -1
            for b_idx in random.sample(cand_blocks, min(20, len(cand_blocks))):
                bts = block_to_ts_np[b_idx]
                gain = np.sum(cover_count[bts] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b >= 0:
                remaining.append(list(all_k_blocks[best_b]))
                np.add.at(cover_count, block_to_ts_np[best_b], 1)

        # Run local search on perturbed solution
        perturbed, _ = local_search(remaining, max_iters=1500)
        if len(perturbed) < best_len:
            best_len = len(perturbed)
            best_solution = perturbed

    return best_solution
