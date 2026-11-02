import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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
    num_b = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(np.array(ts_list, dtype=np.int32))

    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_b, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.full(num_b, False, dtype=bool)
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
                ec = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                b_idx = int(ec[random.randint(0, len(ec) - 1)]) if len(ec) > 0 else int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            bts = block_to_ts[b_idx]
            newly = bts[is_uncovered[bts]]
            if len(newly) > 0:
                is_uncovered[newly] = False
                to_dec = []
                for ts_idx in newly:
                    to_dec.extend(ts_to_blocks[ts_idx])
                if to_dec:
                    counts = np.bincount(to_dec, minlength=num_b)
                    gain -= counts
                    gain[is_selected] = -1000
        return sol

    def lp_rounding(seed, max_rounds=80):
        """Iterative LP rounding: solve dual subproblems to get prices,
        select blocks by reduced cost (price-weighted coverage gain - 1)."""
        random.seed(seed)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.full(num_b, False, dtype=bool)
        cover_count = np.zeros(num_t, dtype=np.int32)
        sol = []

        for rnd in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            if not is_uncovered.any():
                break

            # Solve dual subproblem: maximize sum(y) s.t. sum(y_j for j in block) <= 1
            # Use multiplicative weights / Frank-Wolfe style iteration
            y = np.full(np.sum(is_uncovered), 1.0 / C(k, t), dtype=np.float64)
            uncovered_indices = np.where(is_uncovered)[0]

            # Frank-Wolfe iterations on the dual
            for fw_iter in range(15):
                if time.time() > deadline - 10.0:
                    break
                # Compute primal infeasibility: for each block, sum of y for uncovered ts in it
                # Find the block with maximum sum(y_j)
                # Efficient: for each uncovered ts, add y to all blocks containing it
                block_sum = np.zeros(num_b, dtype=np.float64)
                for idx in range(len(uncovered_indices)):
                    ts_idx = uncovered_indices[idx]
                    y_val = y[idx]
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            block_sum[b_idx] += y_val

                # Find max block_sum (violated constraint most)
                active = ~is_selected
                if not active.any():
                    break
                max_block = int(np.argmax(block_sum * active))
                max_val = block_sum[max_block]
                if max_val <= 1.0 + 1e-9:
                    break  # Dual feasible

                # Frank-Wolfe: move y toward the constraint-violating block
                # The gradient of the dual objective w.r.t. y is 1 for uncovered
                # The constraint violated is for max_block
                # Update: y <- (1-alpha)*y + alpha*(indicator of max_block's ts)
                alpha = 2.0 / (fw_iter + 2.0)
                block_ts = block_to_ts[max_block]
                # Map block_ts to positions in uncovered_indices
                # For each ts in block_ts that is uncovered, set y to 1 (the vertex)
                for ts_idx in block_ts:
                    pos = np.where(uncovered_indices == ts_idx)[0]
                    if len(pos) > 0:
                        y[pos[0]] = (1 - alpha) * y[pos[0]] + alpha * 1.0

                # Renormalize: ensure sum(y) doesn't grow unbounded
                # Actually for the dual, y >= 0 and sum(y_j for j in block) <= 1
                # The objective is to maximize sum(y), so we want y as large as possible
                # Frank-Wolfe on the dual: maximize c^T y s.t. Ay <= 1, y >= 0
                # c = 1 (all ones), A is the block-contains-ts incidence
                # The FW step: find vertex of polytope maximizing gradient direction
                # Gradient of objective = 1 (uniform), so we want to maximize sum(y)
                # Subject to constraints. The FW update moves toward the feasible vertex
                # that best improves the objective given the current constraint violation.

            # Compute reduced cost for each block:
            # RC(b) = sum(y_j for j in uncovered ts in block b) - 1
            # Select block with highest positive RC
            block_rc = np.zeros(num_b, dtype=np.float64)
            for idx in range(len(uncovered_indices)):
                ts_idx = uncovered_indices[idx]
                y_val = y[idx]
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        block_rc[b_idx] += y_val
            block_rc -= 1.0  # subtract cost of 1

            # Select best block
            active = ~is_selected
            if not active.any():
                break
            max_rc = block_rc[active].max()
            if max_rc < 1e-6:
                # Fall back to greedy
                break
            candidates = np.where((block_rc == max_rc) & active)[0]
            b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            bts = block_to_ts[b_idx]
            newly = bts[is_uncovered[bts]]
            if len(newly) > 0:
                is_uncovered[newly] = False
                cover_count[newly] += 1
                # Also increment cover_count for ts already covered
                for ts_idx in bts:
                    cover_count[ts_idx] += 1

        # Greedy completion for any remaining uncovered
        if is_uncovered.any():
            gain = np.zeros(num_b, dtype=np.int32)
            for b_idx in range(num_b):
                if not is_selected[b_idx]:
                    bts = block_to_ts[b_idx]
                    gain[b_idx] = np.sum(is_uncovered[bts])
            while is_uncovered.any():
                if time.time() > deadline - 5.0:
                    break
                max_g = int(np.max(gain))
                if max_g <= 0:
                    break
                cands = np.where((gain == max_g) & (~is_selected))[0]
                if len(cands) == 0:
                    break
                b_idx = int(cands[random.randint(0, len(cands) - 1)])
                sol.append(list(all_k_blocks[b_idx]))
                is_selected[b_idx] = True
                bts = block_to_ts[b_idx]
                newly = bts[is_uncovered[bts]]
                if len(newly) > 0:
                    is_uncovered[newly] = False
                    to_dec = []
                    for ts_idx in newly:
                        to_dec.extend(ts_to_blocks[ts_idx])
                    if to_dec:
                        counts = np.bincount(to_dec, minlength=num_b)
                        gain -= counts
                        gain[is_selected] = -1000

        return sol

    def fast_local_search(blocks, max_iters=3000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        cover_count = np.zeros(num_t, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

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
                    improved_count += 1
                    changed = True
                    break

        # Phase B: 1-exchange
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
            new_ts_arr = block_to_ts[new_block_idx]

            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                sol_block_set.discard(get_block_idx(blocks[i]))
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        return blocks, improved_count

    def fast_local_search_2exchange(blocks, max_iters=2000):
        n_blocks = len(blocks)
        if n_blocks < 3:
            return blocks, 0

        cover_count = np.zeros(num_t, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)

        improved_count = 0
        iters = 0

        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 5.0:
                break
            if len(blocks) < 3:
                break

            i = random.randint(0, len(blocks) - 1)
            j = random.randint(0, len(blocks) - 1)
            if i == j:
                continue

            ts_i = block_ts_indices[i]
            ts_j = block_ts_indices[j]

            unique_i = ts_i[cover_count[ts_i] == 1]
            unique_j = ts_j[cover_count[ts_j] == 1]

            all_unique = np.concatenate([unique_i, unique_j]) if (len(unique_i) > 0 or len(unique_j) > 0) else np.array([], dtype=np.int32)

            if len(all_unique) == 0:
                np.subtract.at(cover_count, ts_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                if i < len(blocks):
                    j = j - 1 if j > i else j
                improved_count += 1
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

            new_block_idx = random.choice(list(candidates))
            new_ts_arr = block_to_ts[new_block_idx]

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
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_j, 1)
                np.add.at(cover_count, ts_i, 1)

        return blocks, improved_count

    # Phase 1: Generate initial solutions
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Try LP rounding with multiple seeds
    for seed in range(1, 15):
        if time.time() > deadline - 60.0:
            break
        sol = lp_rounding(seed, max_rounds=60)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # More greedy runs
    for i in range(15, 50):
        if time.time() > deadline - 40.0:
            break
        explore = 0.0 if i < 30 else (0.1 if i < 40 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Local search
    for _ in range(300):
        if time.time() > deadline - 8.0:
            break
        best_solution, improved = fast_local_search(best_solution, max_iters=2000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

        if time.time() > deadline - 6.0:
            break
        best_solution, improved2 = fast_local_search_2exchange(best_solution, max_iters=1000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

        total_improved = improved + improved2
        if total_improved == 0 and time.time() > deadline - 15.0:
            break

    # Final polish: redundancy removal
    for _ in range(30):
        if time.time() > deadline - 2.0:
            break
        changed = False
        for i in range(len(best_solution)):
            if time.time() > deadline - 2.0:
                break
            b = best_solution[i]
            idx = get_block_idx(b)
            if idx < 0:
                continue
            ts_arr = block_to_ts[idx]
            temp_cover = np.zeros(num_t, dtype=np.int32)
            for j, b2 in enumerate(best_solution):
                if j == i:
                    continue
                idx2 = get_block_idx(b2)
                if idx2 >= 0:
                    np.add.at(temp_cover, block_to_ts[idx2], 1)
            if np.all(temp_cover[ts_arr] >= 1):
                best_solution.pop(i)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
                changed = True
                break
        if not changed:
            break

    return best_solution
