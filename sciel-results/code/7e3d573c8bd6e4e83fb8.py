import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Precompute covered_indices for all blocks
    all_covered = [np.where(M[i] == 1)[0] for i in range(num_kblocks)]

    # Build reverse index: for each tsub, which blocks cover it
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_kblocks):
        for idx in all_covered[i]:
            tsub_blocks[idx].append(i)
    tsub_blocks = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in tsub_blocks]

    rng = random.Random(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    def greedy_fill(uncovered_mask, current_counts, current_mask, current_indices):
        """Greedily add blocks to cover all uncovered tsubs."""
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 8.0:
                break
            gains = M @ uncovered_mask
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            current_mask[best_idx] = True
            current_indices.append(best_idx)
            covered_indices = all_covered[best_idx]
            current_counts[covered_indices] += 1
            uncovered_mask = (current_counts == 0).astype(np.uint8)

    def vectorized_remove_redundant(current_mask, current_indices, current_counts):
        """Vectorized redundancy removal: remove all blocks that are redundant."""
        if len(current_indices) == 0:
            return
        changed = True
        while changed:
            changed = False
            if len(current_indices) == 0:
                break
            sel = np.array(current_indices, dtype=np.int64)
            M_sel = M[sel]  # (n_sel, num_tsubs)
            # coverage counts within the selected set
            cov_counts = M_sel.sum(axis=0)  # (num_tsubs,)
            # A block i is redundant if all tsubs it covers have cov_counts > 1
            # i.e., M_sel[i] * (cov_counts == 1) has no 1s
            # redundant_mask[i] = not any(M_sel[i] & (cov_counts == 1))
            unique_mask = (cov_counts == 1).astype(np.uint8)
            # For each selected block, count how many tsubs it uniquely covers
            unique_cover = M_sel @ unique_mask  # (n_sel,)
            redundant = unique_cover == 0
            if not np.any(redundant):
                break
            # Remove all redundant blocks at once
            for i in np.where(redundant)[0]:
                b_idx = sel[i]
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                covered_indices = all_covered[b_idx]
                current_counts[covered_indices] -= 1
            changed = True

    def try_swaps(current_mask, current_indices, current_counts):
        """Try single-block swaps to improve coverage."""
        improved = False
        if len(current_indices) == 0:
            return improved
        idx_list = current_indices[:]
        rng.shuffle(idx_list)

        for b_idx in idx_list:
            if time.monotonic() > deadline - 2.0:
                break
            if not current_mask[b_idx]:
                continue
            covered_indices = all_covered[b_idx]
            critical_mask = current_counts == 1
            critical_indices = covered_indices[critical_mask[covered_indices]]

            if len(critical_indices) == 0:
                # Block is redundant
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                current_counts[covered_indices] -= 1
                improved = True
                break

            if len(critical_indices) == 1:
                crit = critical_indices[0]
                candidates = tsub_blocks[crit]
                candidates = candidates[~current_mask[candidates]]
                if len(candidates) == 0:
                    continue
                best_cand = None
                best_score = -1
                for cand in candidates[:50]:
                    cand_row = M[cand]
                    score = np.sum(cand_row * (current_counts == 1))
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                if best_cand is not None:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    current_mask[best_cand] = True
                    current_indices.append(best_cand)
                    cand_covered = all_covered[best_cand]
                    current_counts[cand_covered] += 1
                    improved = True
                    break
            elif len(critical_indices) <= 3:
                common = None
                for crit in critical_indices:
                    blocks_c = tsub_blocks[crit]
                    if common is None:
                        common = set(blocks_c.tolist())
                    else:
                        common &= set(blocks_c.tolist())
                if common:
                    candidates = np.array(list(common), dtype=np.int64)
                    candidates = candidates[~current_mask[candidates]]
                    if len(candidates) > 0:
                        best_cand = None
                        best_score = -1
                        for cand in candidates[:30]:
                            cand_row = M[cand]
                            score = np.sum(cand_row * (current_counts == 1))
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        if best_cand is not None:
                            current_mask[b_idx] = False
                            current_indices.remove(b_idx)
                            current_counts[covered_indices] -= 1
                            current_mask[best_cand] = True
                            current_indices.append(best_cand)
                            cand_covered = all_covered[best_cand]
                            current_counts[cand_covered] += 1
                            improved = True
                            break

        return improved

    def perturb(current_mask, current_indices, current_counts):
        """Perturb: remove some blocks and greedily refill."""
        if len(current_indices) == 0:
            return
        unique_mask = (current_counts == 1).astype(np.uint8)
        gains = M @ unique_mask

        n_remove = min(5 + rng.randint(0, 5), max(1, len(current_indices) // 3))

        sorted_indices = np.argsort(gains)
        low_gain_candidates = sorted_indices[:max(1, n_remove * 2)]
        valid_low_gain = [i for i in low_gain_candidates if current_mask[i]]

        to_remove = set()
        if valid_low_gain:
            pick_low = rng.sample(valid_low_gain, min(len(valid_low_gain), n_remove // 2 + 1))
            to_remove.update(pick_low)

        remaining = [i for i in current_indices if i not in to_remove]
        if remaining:
            pick_rand = rng.sample(remaining, min(len(remaining), n_remove - len(to_remove)))
            to_remove.update(pick_rand)

        if len(to_remove) < n_remove:
            remaining = [i for i in current_indices if i not in to_remove]
            if remaining:
                pick_extra = rng.sample(remaining, n_remove - len(to_remove))
                to_remove.update(pick_extra)

        for b_idx in list(to_remove):
            if b_idx in current_indices:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                covered_indices = all_covered[b_idx]
                current_counts[covered_indices] -= 1

        uncovered_mask = (current_counts == 0).astype(np.uint8)
        greedy_fill(uncovered_mask, current_counts, current_mask, current_indices)

    # ========== Phase 1: Deterministic Greedy ==========
    uncovered = np.ones(num_tsubs, dtype=np.uint8)
    counts = np.zeros(num_tsubs, dtype=np.int32)
    selected_mask = np.zeros(num_kblocks, dtype=bool)
    selected_indices = []

    while np.any(uncovered):
        if time.monotonic() > deadline - 60.0:
            break
        gains = M @ uncovered
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        covered_indices = all_covered[best_idx]
        uncovered[covered_indices] = 0
        counts[covered_indices] += 1

    if not selected_indices:
        return []

    # Vectorized redundancy removal on greedy solution
    vectorized_remove_redundant(selected_mask, selected_indices, counts)

    best_len = len(selected_indices)
    best_solution = selected_indices[:]
    best_counts = counts.copy()
    best_mask = selected_mask.copy()

    # ========== Phase 2: Local Search on Greedy ==========
    current_indices = best_solution[:]
    current_counts = best_counts.copy()
    current_mask = best_mask.copy()

    T = 5.0
    cooling = 0.95
    min_T = 0.1
    stagnation = 0
    max_stagnation = 20

    while time.monotonic() < deadline - 40.0:
        improved = False

        vectorized_remove_redundant(current_mask, current_indices, current_counts)
        if len(current_indices) < best_len:
            best_len = len(current_indices)
            best_solution = current_indices[:]
            best_counts = current_counts.copy()
            best_mask = current_mask.copy()
            improved = True

        if not improved and try_swaps(current_mask, current_indices, current_counts):
            improved = True

        if improved:
            stagnation = 0
            T = max(T * cooling, min_T)
            if len(current_indices) < best_len:
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
        else:
            stagnation += 1
            if stagnation >= max_stagnation or (T > min_T and rng.random() < 0.2):
                save_indices = current_indices[:]
                save_counts = current_counts.copy()
                save_mask = current_mask.copy()
                save_len = len(current_indices)

                perturb(current_mask, current_indices, current_counts)
                vectorized_remove_redundant(current_mask, current_indices, current_counts)

                new_len = len(current_indices)
                delta = new_len - save_len

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = True
                else:
                    p_accept = np.exp(-delta / T)
                    if rng.random() < p_accept:
                        accept = True

                if accept:
                    T = max(T * cooling, min_T)
                    stagnation = 0
                    if new_len < best_len:
                        best_len = new_len
                        best_solution = current_indices[:]
                        best_counts = current_counts.copy()
                        best_mask = current_mask.copy()
                else:
                    current_indices = save_indices
                    current_counts = save_counts
                    current_mask = save_mask
                    T = min(T * 1.1, 10.0)
                    stagnation = 0

            T = max(T * cooling, min_T)

    # ========== Phase 3: Probabilistic Method with Alteration ==========
    # Try multiple random starting points using probabilistic selection
    for prob_seed in range(5):
        if time.monotonic() > deadline - 10.0:
            break
        rng_p = random.Random(1000 + prob_seed * 777)

        # Choose probability p for random block selection
        # Expected number of blocks: p * num_kblocks
        # We want roughly 150-200 blocks as a starting point
        p = 0.004 + 0.001 * prob_seed  # ~200 to 250 blocks
        random_mask = rng_p.random(num_kblocks) < p
        random_indices = [i for i in range(num_kblocks) if random_mask[i]]

        # Compute coverage
        random_counts = np.zeros(num_tsubs, dtype=np.int32)
        random_mask_arr = np.zeros(num_kblocks, dtype=bool)
        for b_idx in random_indices:
            random_mask_arr[b_idx] = True
            covered_indices = all_covered[b_idx]
            random_counts[covered_indices] += 1

        # Greedily fill any gaps
        uncovered_mask = (random_counts == 0).astype(np.uint8)
        greedy_fill(uncovered_mask, random_counts, random_mask_arr, random_indices)

        # Alteration: aggressively remove redundant blocks
        vectorized_remove_redundant(random_mask_arr, random_indices, random_counts)

        # Local search polish
        cur_idx = random_indices[:]
        cur_cnt = random_counts.copy()
        cur_mask = random_mask_arr.copy()

        for _ in range(30):
            if time.monotonic() > deadline - 5.0:
                break
            improved_p = False
            vectorized_remove_redundant(cur_mask, cur_idx, cur_cnt)
            if try_swaps(cur_mask, cur_idx, cur_cnt):
                improved_p = True
            if not improved_p:
                break

        if len(cur_idx) < best_len:
            best_len = len(cur_idx)
            best_solution = cur_idx[:]
            best_counts = cur_cnt.copy()
            best_mask = cur_mask.copy()

    # ========== Final Cleanup ==========
    final_indices = best_solution[:]
    final_counts = best_counts.copy()
    final_mask = best_mask.copy()

    vectorized_remove_redundant(final_mask, final_indices, final_counts)

    # Verify
    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        covered_indices = all_covered[b_idx]
        verify_counts[covered_indices] += 1

    if not np.all(verify_counts >= 1):
        # Fallback to greedy solution
        final_indices = selected_indices[:]
        final_counts = counts.copy()
        final_mask = selected_mask.copy()
        vectorized_remove_redundant(final_mask, final_indices, final_counts)

    return get_blocks(final_indices)
