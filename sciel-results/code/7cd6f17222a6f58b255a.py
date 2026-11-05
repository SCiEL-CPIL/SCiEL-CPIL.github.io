import numpy as np
import itertools
import time
import random
import math

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)

    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]

    # Precompute subset_to_blocks as sets for fast intersection in swap
    subset_to_blocks_sets = [set(blocks) for blocks in subset_to_blocks]

    # Precompute block size (number of t-subsets per block)
    block_sizes = np.array([len(s) for s in block_to_subsets], dtype=np.int32)

    def run_greedy_full(random_seed=None):
        rng = np.random.default_rng(random_seed if random_seed is not None else None)

        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy()

        selected_blocks = []

        while True:
            if not np.any(uncovered):
                break

            noise = rng.uniform(0, 0.4, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]

            if len(newly_covered) == 0:
                continue

            uncovered[newly_covered] = False
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1

        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        rng = np.random.default_rng(random_seed if random_seed is not None else None)

        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []

        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            blocks_covering = subset_block_arrays[s_idx]
            gains[blocks_covering] += 1

        selected_blocks = []
        current_uncovered = uncovered_mask.copy()

        while True:
            if not np.any(current_uncovered):
                break

            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]

            if len(newly_covered) == 0:
                continue

            current_uncovered[newly_covered] = False
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1

        return selected_blocks

    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search: deletion + smooth-objective swap (soft-max surrogate)."""
        rng = random.Random(random_seed if random_seed is not None else None)

        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks, coverage_counts.copy()

        cover_count = coverage_counts.copy()
        current_blocks_set = set(current_blocks)

        # Dynamic clause weights: weight for each t-subset based on tightness
        clause_weights = np.ones(num_subsets, dtype=np.float64)

        improved = True
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 25
        lambda_pen = 0.05
        tau = 1.0

        def softmax_of(vals, temperature):
            if len(vals) == 0:
                return 0.0
            arr = np.asarray(vals, dtype=np.float64)
            m = np.max(arr)
            if m == 0:
                return 0.0
            return m + temperature * math.log(np.sum(np.exp((arr - m) / temperature)))

        while improved and (time.time() - start_time) < time_limit:
            improved = False
            no_improve_count += 1

            if no_improve_count > max_no_improve:
                clause_weights = np.ones(num_subsets, dtype=np.float64)
                no_improve_count = 0

            # Deletion pass
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                is_redundant = True
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                        if cover_count[s_idx] == 1:
                            clause_weights[s_idx] *= 1.2
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    no_improve_count = 0
                    break

            # Swap pass with smooth surrogate objective
            if (time.time() - start_time) > time_limit:
                break

            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            swaps_done = 0
            max_swaps = 100

            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue

                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]

                unique_subs = [s_idx for s_idx in old_subs if cover_count[s_idx] == 1]

                if not unique_subs:
                    # Not critical: smooth random diversifying swap
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block or new_block in current_blocks_set:
                        continue
                    new_subs = block_subset_arrays[new_block]
                    gain_vals = [clause_weights[s] for s in new_subs]
                    loss_vals = [clause_weights[s] for s in old_subs]
                    score = softmax_of(gain_vals, tau) - lambda_pen * softmax_of(loss_vals, tau)
                    if score > 0 or rng.random() < 0.25:
                        for s_idx in old_subs:
                            cover_count[s_idx] -= 1
                            if cover_count[s_idx] == 1:
                                clause_weights[s_idx] *= 1.2
                        for s_idx in new_subs:
                            cover_count[s_idx] += 1
                            if cover_count[s_idx] == 2:
                                clause_weights[s_idx] *= 0.8
                        current_blocks_set.discard(old_block)
                        current_blocks_set.add(new_block)
                        current_blocks[i] = new_block
                        swaps_done += 1
                        improved = True
                        no_improve_count = 0
                else:
                    # Critical: candidates must cover all unique subsets
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = subset_to_blocks_sets[s_idx].copy()
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    if not candidates:
                        continue
                    candidates.discard(old_block)
                    if not candidates:
                        continue

                    # Smooth objective over candidate set (log-sum-exp soft-max)
                    best_score = -float('inf')
                    best_cand = None
                    # Subsample if too many candidates to bound cost
                    cand_list = list(candidates)
                    if len(cand_list) > 400:
                        cand_list = rng.sample(cand_list, 400)
                    for cand in cand_list:
                        if cand in current_blocks_set:
                            continue
                        cand_subs = block_subset_arrays[cand]
                        gain_vals = [clause_weights[s] for s in cand_subs]
                        loss_vals = [clause_weights[s] for s in old_subs]
                        score = softmax_of(gain_vals, tau) - lambda_pen * softmax_of(loss_vals, tau)
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    if best_cand is None:
                        continue

                    new_subs = block_subset_arrays[best_cand]
                    for s_idx in old_subs:
                        cover_count[s_idx] -= 1
                        if cover_count[s_idx] == 1:
                            clause_weights[s_idx] *= 1.2
                    for s_idx in new_subs:
                        cover_count[s_idx] += 1
                        if cover_count[s_idx] == 2:
                            clause_weights[s_idx] *= 0.8
                    current_blocks_set.discard(old_block)
                    current_blocks_set.add(best_cand)
                    current_blocks[i] = best_cand
                    swaps_done += 1
                    improved = True
                    no_improve_count = 0

        return current_blocks, cover_count

    def compute_coverage(blocks_indices):
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            b_subs = block_subset_arrays[b_idx]
            cover_count[b_subs] += 1
        return cover_count

    def final_polish(blocks_indices, coverage_counts, time_budget):
        """High-precision polish: deletion to fixpoint + exact critical-swap pass."""
        blocks = list(blocks_indices)
        cov = coverage_counts.copy()
        bset = set(blocks)
        end = time.time() + time_budget

        # Deletion to fixpoint
        changed = True
        while changed and time.time() < end:
            changed = False
            for i in range(len(blocks)):
                b_idx = blocks[i]
                b_subs = block_subset_arrays[b_idx]
                redundant = True
                for s_idx in b_subs:
                    if cov[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for s_idx in b_subs:
                        cov[s_idx] -= 1
                    bset.discard(b_idx)
                    blocks.pop(i)
                    changed = True
                    break

        # Exact critical-swap pass (only swaps that keep all subsets covered)
        improved = True
        while improved and time.time() < end:
            improved = False
            for i in range(len(blocks)):
                if time.time() >= end:
                    break
                old_block = blocks[i]
                old_subs = block_subset_arrays[old_block]
                unique_subs = [s_idx for s_idx in old_subs if cov[s_idx] == 1]
                if not unique_subs:
                    continue
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = subset_to_blocks_sets[s_idx].copy()
                    else:
                        candidates.intersection_update(subset_to_blocks_sets[s_idx])
                        if not candidates:
                            break
                if not candidates:
                    continue
                candidates.discard(old_block)
                candidates -= bset
                if not candidates:
                    continue
                new_block = next(iter(candidates))
                new_subs = block_subset_arrays[new_block]
                for s_idx in old_subs:
                    cov[s_idx] -= 1
                for s_idx in new_subs:
                    cov[s_idx] += 1
                bset.discard(old_block)
                bset.add(new_block)
                blocks[i] = new_block
                improved = True

        return blocks, cov

    best_blocks = None
    best_len = float('inf')
    best_coverage = None

    start_time = time.time()
    deadline = start_time + 142.0  # explicit guard, safety margin vs 150s kill

    def tleft():
        return deadline - time.time()

    trial = 0
    while tleft() > 20.0:
        # Strategy 1: Full Greedy + Local Search
        greedy_res = run_greedy_full(trial)
        coverage = compute_coverage(greedy_res)

        ls_time = min(4.0, max(1.5, tleft() * 0.18))
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=ls_time, random_seed=trial)

        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            best_coverage = min_cov

        # Strategy 2: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and tleft() > 35.0:
            perturbed = list(best_blocks)
            num_remove = min(max(3, len(perturbed) // (8 + trial % 8)), len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)

                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    b_subs = block_subset_arrays[b_idx]
                    cover_count_temp[b_subs] += 1

                uncovered_mask = (cover_count_temp == 0).astype(bool)
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)

                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                new_coverage = compute_coverage(new_set)

                ls_time2 = min(4.0, max(1.5, tleft() * 0.18))
                min_res2, min_cov2 = minimize_with_swaps(new_set, new_coverage, time_limit=ls_time2, random_seed=trial + 2000)

                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
                    best_coverage = min_cov2

        # Strategy 3: Diversification on best solution
        if best_blocks is not None and len(best_blocks) > 20 and tleft() > 30.0:
            ls_time3 = min(5.0, max(2.0, tleft() * 0.22))
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=ls_time3, random_seed=trial + 5000)

            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov

        trial += 1

    # Final high-precision polish on best solution
    if best_blocks is not None and tleft() > 6.0:
        polish_budget = min(10.0, tleft() - 4.0)
        final_blocks, final_cov = final_polish(best_blocks, best_coverage, polish_budget)
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            best_coverage = final_cov

    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)

    return [list(all_blocks[i]) for i in best_blocks]
