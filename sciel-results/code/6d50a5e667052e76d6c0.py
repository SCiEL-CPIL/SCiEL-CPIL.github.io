import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin for 150s limit

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

    # --- Shared polish state: one cover_count maintained incrementally ---
    def build_polish_state(blocks):
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
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)
        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)
        return cover_count, block_ts_indices, sol_block_set

    # Smooth surrogate for coverage uniformity: negative Shannon entropy of
    # the (normalized) coverage distribution. Lower = more uniform = less
    # structural redundancy. We maximize this (i.e. minimize its magnitude).
    def coverage_entropy(cover_count):
        total = int(np.sum(cover_count))
        if total <= 0:
            return 0.0
        p = cover_count.astype(np.float64) / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def polish(blocks, rng, max_iters=4000):
        """Unified local search: redundancy strip, 1-for-1 (entropy-scored),
        2-for-1, and 2-for-2 merges. Maintains a single cover_count."""
        if len(blocks) == 0:
            return blocks, 0
        cover_count, block_ts_indices, sol_block_set = build_polish_state(blocks)
        improved = 0
        iters = 0

        # Phase A: strip redundancies to a fixed point
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    idx = get_block_idx(blocks[i]) if i < len(blocks) else -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved += 1
                    changed = True
                    break

        # Phase B: mixed moves
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            move = rng.random()
            if move < 0.55:
                # ---- 1-for-1 swap, scored by coverage uniformity + overlap ----
                i = rng.randint(0, len(blocks) - 1)
                ts_arr_i = block_ts_indices[i]
                unique_mask = cover_count[ts_arr_i] == 1
                unique_ts = ts_arr_i[unique_mask]
                if len(unique_ts) == 0:
                    np.subtract.at(cover_count, ts_arr_i, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    idx = get_block_idx(blocks[i]) if i < len(blocks) else -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved += 1
                    continue
                candidates = set(ts_to_blocks[int(unique_ts[0])])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                    if not candidates:
                        break
                cur_idx = get_block_idx(blocks[i])
                candidates.discard(cur_idx)
                if not candidates:
                    continue
                in_i = np.zeros(num_t_subsets, dtype=bool)
                in_i[ts_arr_i] = True
                cand_list = list(candidates)
                if len(cand_list) > 24:
                    rng.shuffle(cand_list)
                    cand_list = cand_list[:24]
                best_cand = None
                best_score = -np.inf
                for c in cand_list:
                    new_ts = block_to_ts_np[c]
                    in_c = np.zeros(num_t_subsets, dtype=bool)
                    in_c[new_ts] = True
                    only_i = in_i & ~in_c
                    if np.any(only_i) and np.min(cover_count[only_i]) < 2:
                        continue
                    # Score: prefer candidates that make coverage more uniform
                    # (higher entropy) and that overlap existing coverage.
                    new_cover = cover_count.copy()
                    np.subtract.at(new_cover, ts_arr_i, 1)
                    np.add.at(new_cover, new_ts, 1)
                    ent = coverage_entropy(new_cover)
                    overlap = int(np.sum(cover_count[new_ts] >= 2))
                    score = ent + 0.01 * overlap
                    if score > best_score:
                        best_score = score
                        best_cand = c
                if best_cand is not None:
                    np.subtract.at(cover_count, ts_arr_i, 1)
                    new_ts_arr = block_to_ts_np[best_cand]
                    np.add.at(cover_count, new_ts_arr, 1)
                    blocks[i] = list(all_k_blocks[best_cand])
                    block_ts_indices[i] = new_ts_arr
                    sol_block_set.discard(cur_idx)
                    sol_block_set.add(best_cand)
                    improved += 1

            elif move < 0.80:
                # ---- 2-for-1: remove two blocks, add one covering both uniques ----
                if len(blocks) < 2:
                    continue
                i = rng.randint(0, len(blocks) - 1)
                j = rng.randint(0, len(blocks) - 1)
                if i == j:
                    continue
                ts_i = block_ts_indices[i]
                ts_j = block_ts_indices[j]
                uniq_i = ts_i[cover_count[ts_i] == 1]
                uniq_j = ts_j[cover_count[ts_j] == 1]
                all_unique = np.concatenate([uniq_i, uniq_j]) if (len(uniq_i) or len(uniq_j)) else np.array([], dtype=np.int32)
                if len(all_unique) == 0:
                    np.subtract.at(cover_count, ts_i, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    idx = get_block_idx(blocks[i]) if i < len(blocks) else -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved += 1
                    continue
                candidates = set(ts_to_blocks[int(all_unique[0])])
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                    if not candidates:
                        break
                idx_i = get_block_idx(blocks[i])
                idx_j = get_block_idx(blocks[j])
                candidates.discard(idx_i)
                candidates.discard(idx_j)
                if not candidates:
                    continue
                new_block_idx = rng.choice(list(candidates))
                new_ts_arr = block_to_ts_np[new_block_idx]
                np.subtract.at(cover_count, ts_i, 1)
                np.subtract.at(cover_count, ts_j, 1)
                np.add.at(cover_count, new_ts_arr, 1)
                if np.min(cover_count) >= 1:
                    hi = max(i, j)
                    lo = min(i, j)
                    blocks.pop(hi)
                    block_ts_indices.pop(hi)
                    blocks.pop(lo)
                    block_ts_indices.pop(lo)
                    blocks.append(list(all_k_blocks[new_block_idx]))
                    block_ts_indices.append(new_ts_arr)
                    sol_block_set.discard(idx_i)
                    sol_block_set.discard(idx_j)
                    sol_block_set.add(new_block_idx)
                    improved += 1
                else:
                    np.subtract.at(cover_count, new_ts_arr, 1)
                    np.add.at(cover_count, ts_j, 1)
                    np.add.at(cover_count, ts_i, 1)

            else:
                # ---- 2-for-2: replace two blocks with two new ones ----
                if len(blocks) < 2:
                    continue
                i = rng.randint(0, len(blocks) - 1)
                j = rng.randint(0, len(blocks) - 1)
                if i == j:
                    continue
                ts_i = block_ts_indices[i]
                ts_j = block_ts_indices[j]
                uniq_i = ts_i[cover_count[ts_i] == 1]
                uniq_j = ts_j[cover_count[ts_j] == 1]
                all_unique = np.concatenate([uniq_i, uniq_j]) if (len(uniq_i) or len(uniq_j)) else np.array([], dtype=np.int32)
                if len(all_unique) == 0:
                    np.subtract.at(cover_count, ts_i, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    idx = get_block_idx(blocks[i]) if i < len(blocks) else -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved += 1
                    continue
                candidates = set(ts_to_blocks[int(all_unique[0])])
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                    if not candidates:
                        break
                idx_i = get_block_idx(blocks[i])
                idx_j = get_block_idx(blocks[j])
                candidates.discard(idx_i)
                candidates.discard(idx_j)
                if len(candidates) < 2:
                    continue
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                cand_list = cand_list[:6]
                found = False
                for a in range(len(cand_list)):
                    for b_cand in range(a + 1, len(cand_list)):
                        new_ts_a = block_to_ts_np[cand_list[a]]
                        new_ts_b = block_to_ts_np[cand_list[b_cand]]
                        temp_cover = cover_count.copy()
                        np.subtract.at(temp_cover, ts_i, 1)
                        np.subtract.at(temp_cover, ts_j, 1)
                        np.add.at(temp_cover, new_ts_a, 1)
                        np.add.at(temp_cover, new_ts_b, 1)
                        if np.min(temp_cover) >= 1:
                            hi = max(i, j)
                            lo = min(i, j)
                            blocks.pop(hi)
                            block_ts_indices.pop(hi)
                            blocks.pop(lo)
                            block_ts_indices.pop(lo)
                            blocks.append(list(all_k_blocks[cand_list[a]]))
                            block_ts_indices.append(new_ts_a)
                            blocks.append(list(all_k_blocks[cand_list[b_cand]]))
                            block_ts_indices.append(new_ts_b)
                            sol_block_set.discard(idx_i)
                            sol_block_set.discard(idx_j)
                            sol_block_set.add(cand_list[a])
                            sol_block_set.add(cand_list[b_cand])
                            improved += 1
                            found = True
                            break
                    if found:
                        break

        return blocks, improved

    # --- Initial solution: deterministic greedy, then randomized restarts ---
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    num_runs = 120
    for i in range(1, num_runs):
        if time.time() > deadline - 30.0:
            break
        explore = 0.0 if i < 40 else (0.1 if i < 80 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # --- Iterative polish with the unified local search ---
    rng = random.Random(12345)
    for _ in range(400):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = polish(best_solution, rng, max_iters=2500)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        if improved == 0 and time.time() > deadline - 10.0:
            break

    # --- Final aggressive redundancy strip ---
    for _ in range(200):
        if time.time() > deadline - 2.0:
            break
        if len(best_solution) == 0:
            break
        cover_count, block_ts_indices, sol_block_set = build_polish_state(best_solution)
        changed = False
        for i in range(len(block_ts_indices)):
            if time.time() > deadline - 2.0:
                break
            ts_arr = block_ts_indices[i]
            if np.all(cover_count[ts_arr] >= 2):
                np.subtract.at(cover_count, ts_arr, 1)
                best_solution.pop(i)
                block_ts_indices.pop(i)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
                changed = True
                break
        if not changed:
            break

    return best_solution
