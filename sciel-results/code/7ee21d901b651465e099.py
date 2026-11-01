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
    num_t_subsets = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Convert to numpy arrays for speed
    block_ts_np = [np.array(x, dtype=np.int32) for x in block_to_ts]

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Greedy with randomization parameter
    def greedy_with_randomization(seed, alpha=1.0):
        random.seed(seed)
        rng = np.random.RandomState(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol_indices = []

        while True:
            if time.time() > deadline - 10.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            # Soft-max selection: pick from top candidates with probability proportional to gain^alpha
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if alpha < 1.0 and len(candidates) > 1:
                # Weighted random selection among top candidates
                weights = np.ones(len(candidates), dtype=np.float64)
                # Add small noise to break ties
                weights += rng.rand(len(candidates)) * 0.1
                probs = weights / weights.sum()
                b_idx = candidates[rng.choice(len(candidates), p=probs)]
            else:
                b_idx = candidates[rng.randint(0, len(candidates))]

            sol_indices.append(int(b_idx))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol_indices

    # Run multiple diverse greedy instances
    best_sol_indices = None
    best_len = float('inf')

    # Deterministic greedy first
    det_sol = greedy_with_randomization(0, alpha=1.0)
    if det_sol is not None:
        best_sol_indices = det_sol
        best_len = len(det_sol)

    # Randomized greedy with different alphas and seeds
    for seed in range(1, 50):
        if time.time() > deadline - 15.0:
            break
        alpha = 1.0 + (seed % 5) * 0.2  # Vary alpha
        sol = greedy_with_randomization(seed, alpha=alpha)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_sol_indices = sol

    # Convert to blocks
    def indices_to_blocks(indices):
        return [list(all_k_blocks[i]) for i in indices]

    best_solution = indices_to_blocks(best_sol_indices)

    # Efficient redundancy removal
    def optimize_redundancy(blocks):
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
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
        return blocks

    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    # Local search: single-block swap
    def local_search_swap(blocks, max_passes=50):
        if not blocks:
            return blocks, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        for _ in range(max_passes):
            if time.time() > deadline - 3.0:
                break

            found_swap = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    break

                unique_ts = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)

                if not unique_ts:
                    continue

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

                for ts in itertools.combinations(old_block, t):
                    cover_count[ts_index_map[ts]] -= 1
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1

                improved = True
                found_swap = True
                break

            if not found_swap:
                break

        return blocks, improved

    # 2-exchange: replace two blocks with one or two
    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        n = len(blocks)

        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    break

                unique_ts_i = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts_i.append(idx)

                unique_ts_j = []
                for ts in itertools.combinations(blocks[j], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts_j.append(idx)

                all_unique = unique_ts_i + unique_ts_j
                if not all_unique:
                    continue

                # Try single block covering all
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
                        old_i = blocks[i]
                        old_j = blocks[j]

                        for ts in itertools.combinations(old_i, t):
                            cover_count[ts_index_map[ts]] -= 1
                        for ts in itertools.combinations(old_j, t):
                            cover_count[ts_index_map[ts]] -= 1
                        for ts in itertools.combinations(new_block, t):
                            cover_count[ts_index_map[ts]] += 1

                        sol_block_indices.discard(get_block_idx(old_i))
                        sol_block_indices.discard(get_block_idx(old_j))
                        sol_block_indices.add(new_block_idx)

                        blocks.pop(j)
                        blocks[i] = new_block
                        improved = True
                        break

                # Try two separate blocks
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
                    new_i = list(all_k_blocks[new_i_idx])
                    new_j = list(all_k_blocks[new_j_idx])
                    old_i = blocks[i]
                    old_j = blocks[j]

                    for ts in itertools.combinations(old_i, t):
                        cover_count[ts_index_map[ts]] -= 1
                    for ts in itertools.combinations(old_j, t):
                        cover_count[ts_index_map[ts]] -= 1
                    for ts in itertools.combinations(new_i, t):
                        cover_count[ts_index_map[ts]] += 1
                    for ts in itertools.combinations(new_j, t):
                        cover_count[ts_index_map[ts]] += 1

                    sol_block_indices.discard(get_block_idx(old_i))
                    sol_block_indices.discard(get_block_idx(old_j))
                    sol_block_indices.add(new_i_idx)
                    sol_block_indices.add(new_j_idx)

                    blocks[i] = new_i
                    blocks[j] = new_j
                    improved = True
                    break
            if improved:
                break

        return blocks, improved

    # Main local search loop
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_passes=30)
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

    # Final polish: try more aggressive swaps
    if time.time() < deadline - 3.0:
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            best_solution, improved = local_search_swap(best_solution, max_passes=20)
            if improved:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    return best_solution
