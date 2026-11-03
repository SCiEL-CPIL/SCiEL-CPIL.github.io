import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    # Build block_cols: for each block, the array of t-subset indices it covers
    block_cols = []
    for block in k_subs:
        cols = []
        for t_sub in itertools.combinations(block, t):
            cols.append(t_sub_to_idx[t_sub])
        block_cols.append(np.array(cols, dtype=np.int32))

    # Precompute column counts for each block (for quick gain estimation)
    block_sizes = np.array([len(c) for c in block_cols], dtype=np.int32)

    def compute_gains(u, selected_set):
        """Compute gain for each block: number of uncovered t-subsets it covers."""
        gains = np.zeros(n_k, dtype=np.int32)
        for i in range(n_k):
            if i in selected_set:
                continue
            gains[i] = np.sum(u[block_cols[i]])
        return gains

    def greedy_cover(seed_blocks=None, rng=None, max_iter=None, tie_break=False):
        if rng is None:
            rng = random
        if max_iter is None:
            max_iter = 20000

        u = np.ones(n_t, dtype=np.int8)
        selected = []
        selected_set = set()

        if seed_blocks:
            for idx in seed_blocks:
                u[block_cols[idx]] = 0
                selected.append(idx)
                selected_set.add(idx)

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if u.sum() == 0:
                break

            gains = compute_gains(u, selected_set)
            max_gain = gains.max()
            if max_gain == 0:
                break

            if tie_break:
                candidates = np.where(gains == max_gain)[0]
                best_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                best_idx = int(np.argmax(gains))

            selected.append(best_idx)
            selected_set.add(best_idx)
            u[block_cols[best_idx]] = 0

        return selected

    def remove_redundant(selected_blocks):
        if not selected_blocks:
            return selected_blocks

        n_sel = len(selected_blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in selected_blocks:
            cover_count[block_cols[idx]] += 1

        active = [True] * n_sel
        changed = True

        while changed:
            if time.time() > deadline:
                break
            changed = False

            redundant = []
            for i in range(n_sel):
                if not active[i]:
                    continue
                cols = block_cols[selected_blocks[i]]
                if np.any(cover_count[cols] == 1):
                    continue
                redundant.append(i)

            if redundant:
                changed = True
                for i in redundant:
                    cols = block_cols[selected_blocks[i]]
                    cover_count[cols] -= 1
                    active[i] = False

        return [selected_blocks[i] for i in range(n_sel) if active[i]]

    def local_search(selected_blocks, max_rounds=15, rng=None):
        if rng is None:
            rng = random
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)

        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)
            order = list(range(n_best))
            rng.shuffle(order)

            for i in order:
                if time.time() > deadline - 5:
                    break

                removed_idx = best[i]
                removed_cols = block_cols[removed_idx]

                temp_cover = np.zeros(n_t, dtype=np.int8)
                for j, idx in enumerate(best):
                    if j == i:
                        continue
                    temp_cover[block_cols[idx]] += 1

                uncovered_mask = temp_cover[removed_cols] == 0
                uncovered_count = int(np.sum(uncovered_mask))

                if uncovered_count == 0:
                    best.pop(i)
                    improved = True
                    break

                uncovered_indices = removed_cols[uncovered_mask]
                best_candidate = -1
                best_score = 0

                sample_size = min(3000, n_k)
                candidates = rng.sample(range(n_k), sample_size) if n_k > sample_size else list(range(n_k))

                for cand in candidates:
                    if cand == removed_idx or cand in best:
                        continue
                    cand_set = set(block_cols[cand])
                    score = sum(1 for c in uncovered_indices if c in cand_set)
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                        if score == uncovered_count:
                            break

                if best_candidate >= 0 and best_score == uncovered_count:
                    best[i] = best_candidate
                    improved = True
                    break

            if not improved:
                break

        return best

    def swap_improve(selected_blocks, max_rounds=8, rng=None):
        if rng is None:
            rng = random
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)

        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)

            if n_best >= 2:
                max_pairs = min(300, n_best * (n_best - 1) // 2)
                pairs = [(i, j) for i in range(n_best) for j in range(i + 1, n_best)]
                if len(pairs) > max_pairs:
                    pairs = rng.sample(pairs, max_pairs)

                for i, j in pairs:
                    if time.time() > deadline - 5:
                        break

                    idx_i = best[i]
                    idx_j = best[j]

                    temp_cover = np.zeros(n_t, dtype=np.int8)
                    for m, idx in enumerate(best):
                        if m == i or m == j:
                            continue
                        temp_cover[block_cols[idx]] += 1

                    uncovered_set = set()
                    for c in block_cols[idx_i]:
                        if temp_cover[c] == 0:
                            uncovered_set.add(c)
                    for c in block_cols[idx_j]:
                        if temp_cover[c] == 0:
                            uncovered_set.add(c)

                    if len(uncovered_set) == 0:
                        if j > i:
                            best.pop(j)
                            best.pop(i)
                        else:
                            best.pop(i)
                            best.pop(j)
                        improved = True
                        break

                    sample_size = min(2000, n_k)
                    candidates = rng.sample(range(n_k), sample_size) if n_k > sample_size else list(range(n_k))

                    best_candidate = -1
                    for cand in candidates:
                        if cand == idx_i or cand == idx_j or cand in best:
                            continue
                        if uncovered_set.issubset(set(block_cols[cand])):
                            best_candidate = cand
                            break

                    if best_candidate >= 0:
                        new_best = [best[m] for m in range(n_best) if m != i and m != j]
                        new_best.append(best_candidate)
                        if len(new_best) < best_size:
                            best = new_best
                            best_size = len(best)
                            improved = True
                            break

            if not improved:
                break

        return best

    best_result = None
    best_size = float('inf')

    # 1. Standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)

    # 2. Randomized greedy with tie-breaking
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline - 50:
            break
        rng = random.Random(restart * 7919 + 104729)
        if restart % 2 == 0:
            result = greedy_cover(tie_break=True, rng=rng)
        else:
            seed_size = min(5, n_k)
            seed_blocks = rng.sample(range(n_k), seed_size)
            result = greedy_cover(seed_blocks=seed_blocks, rng=rng, tie_break=True)
        result = remove_redundant(result)
        if len(result) < best_size:
            best_result = result
            best_size = len(result)

    # 3. Local search
    if best_result and time.time() < deadline - 25:
        rng = random.Random(42)
        improved = local_search(best_result, max_rounds=15, rng=rng)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    # 4. Swap improvement
    if best_result and time.time() < deadline - 15:
        rng = random.Random(99)
        improved = swap_improve(best_result, max_rounds=8, rng=rng)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    # 5. More local search
    if best_result and time.time() < deadline - 8:
        rng = random.Random(77)
        improved = local_search(best_result, max_rounds=10, rng=rng)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    if best_result:
        best_result = remove_redundant(best_result)

    if best_result is None:
        best_result = []

    return [list(k_subs[i]) for i in best_result]
