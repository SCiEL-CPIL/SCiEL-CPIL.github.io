import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit

    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # Compute masks: for each block, which t-subsets does it cover?
    block_data = []
    for block in all_k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append((block, mask))

    block_masks = [m for _, m in block_data]
    full_mask = (1 << n_t) - 1

    def greedy_construct(use_log=False, rng=None):
        """Greedy set cover with optional log-gain."""
        uncovered = full_mask
        selected = []
        remaining = list(range(n_k))

        while uncovered != 0:
            if time.time() > deadline - 30:
                break
            best_idx = -1
            best_score = -1
            for i in remaining:
                gain = (block_masks[i] & uncovered).bit_count()
                if gain == 0:
                    continue
                if use_log:
                    total_covered = block_masks[i].bit_count()
                    already_covered = total_covered - gain
                    score = gain / (1 + already_covered * 0.1)
                else:
                    score = gain
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            remaining.remove(best_idx)

        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks iteratively."""
        if not selected_indices:
            return selected_indices

        # Build cover counts
        cover_count = [0] * n_t
        for idx in selected_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb

        # Verify full coverage
        for i in range(n_t):
            if cover_count[i] == 0:
                return selected_indices

        current = selected_indices[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cover_count[i] == 1:
                        is_critical = True
                        break
                    m ^= lsb

                if is_critical:
                    new_current.append(idx)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    improved = True

            current = new_current

        return current

    def local_search(selected_indices, max_time=30):
        """Local search: try swaps and deletions."""
        if not selected_indices:
            return selected_indices

        current = prune_blocks(selected_indices)
        current_set = set(current)
        best = current[:]
        best_len = len(best)

        # Try deletion + re-pruning
        for i in range(len(current)):
            if time.time() > deadline - 15:
                break
            trial = [x for j, x in enumerate(current) if j != i]
            cov = [0] * n_t
            for idx in trial:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    cov[k2] += 1
                    m ^= lsb
            if all(c >= 1 for c in cov):
                pruned = prune_blocks(trial)
                if len(pruned) < best_len:
                    best = pruned
                    best_len = len(best)
                    current = pruned
                    current_set = set(current)

        # Try replacement: remove one, add one
        if time.time() < deadline - 20:
            for i in range(len(current)):
                if time.time() > deadline - 15:
                    break
                removed_idx = current[i]
                trial_without = [x for j, x in enumerate(current) if j != i]
                cov = [0] * n_t
                for idx in trial_without:
                    m = block_masks[idx]
                    while m:
                        lsb = m & -m
                        k2 = lsb.bit_length() - 1
                        cov[k2] += 1
                        m ^= lsb

                uncovered_mask = 0
                for k2 in range(n_t):
                    if cov[k2] == 0:
                        uncovered_mask |= (1 << k2)

                if uncovered_mask == 0:
                    pruned = prune_blocks(trial_without)
                    if len(pruned) < best_len:
                        best = pruned
                        best_len = len(best)
                        current = pruned
                        current_set = set(current)
                    continue

                best_add = -1
                best_add_gain = -1
                for j in range(n_k):
                    if j in current_set:
                        continue
                    gain = (block_masks[j] & uncovered_mask).bit_count()
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = j
                    if gain == uncovered_mask.bit_count():
                        break

                if best_add >= 0 and best_add_gain > 0:
                    trial = trial_without + [best_add]
                    pruned = prune_blocks(trial)
                    if len(pruned) < best_len:
                        best = pruned
                        best_len = len(best)
                        current = pruned
                        current_set = set(current)

        return best

    def dual_local_search(selected_indices, max_time=40):
        """
        Dual Local Search: state is coverage counts vector.
        Moves: (1) remove a block if no count drops below 1,
               (2) swap a redundant block with a candidate that maximizes min coverage.
        """
        if not selected_indices:
            return selected_indices

        current = prune_blocks(selected_indices)
        current_set = set(current)
        best = current[:]
        best_len = len(best)

        # Build coverage counts
        cover_count = [0] * n_t
        for idx in current:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb

        # Iteration 1: aggressive removal pass
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cover_count[i] == 1:
                        is_critical = True
                        break
                    m ^= lsb

                if is_critical:
                    new_current.append(idx)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    improved = True

            current = new_current
            current_set = set(current)

        if len(current) < best_len:
            best = current[:]
            best_len = len(best)

        # Iteration 2: swap-based improvement
        if time.time() < deadline - 15:
            for i in range(len(current)):
                if time.time() > deadline - 15:
                    break

                removed_idx = current[i]
                removed_mask = block_masks[removed_idx]

                # Compute what would be uncovered if we remove this block
                uncovered_after_removal = 0
                m = removed_mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if cover_count[idx_t] == 1:
                        uncovered_after_removal |= (1 << idx_t)
                    m ^= lsb

                if uncovered_after_removal == 0:
                    # Can remove entirely
                    m = removed_mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        cover_count[idx_t] -= 1
                        m ^= lsb
                    current = [x for j, x in enumerate(current) if j != i]
                    current_set.discard(removed_idx)
                    best = current[:]
                    best_len = len(best)
                    i -= 1
                    continue

                # Try to find a replacement block
                best_candidate = -1
                best_candidate_score = -1

                candidates = []
                for j in range(n_k):
                    if j in current_set:
                        continue
                    if (block_masks[j] & uncovered_after_removal) != uncovered_after_removal:
                        continue
                    extra_coverage = (block_masks[j] & ~uncovered_after_removal).bit_count()
                    score = extra_coverage * 0.01
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = j

                if best_candidate >= 0:
                    m = removed_mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        cover_count[idx_t] -= 1
                        m ^= lsb

                    new_mask = block_masks[best_candidate]
                    m = new_mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        cover_count[idx_t] += 1
                        m ^= lsb

                    current[i] = best_candidate
                    current_set.discard(removed_idx)
                    current_set.add(best_candidate)

        current = prune_blocks(current)
        if len(current) < best_len:
            best = current[:]
            best_len = len(best)

        return best

    # Phase 1: Multiple greedy constructions
    best_solution = None
    best_len = float('inf')

    sol = greedy_construct(use_log=False)
    sol = prune_blocks(sol)
    sol = local_search(sol)
    best_solution = sol
    best_len = len(sol)

    if time.time() < deadline - 40:
        sol = greedy_construct(use_log=True)
        sol = prune_blocks(sol)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)

    if time.time() < deadline - 30:
        sol = dual_local_search(best_solution)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)

    # Phase 2: ILS with better perturbation
    rng = random.Random(42)
    ils_iter = 0
    while time.time() < deadline - 25 and ils_iter < 50:
        ils_iter += 1

        perturbed = best_solution[:]
        if len(perturbed) > 2:
            n_remove = min(rng.randint(1, 3), len(perturbed) - 1)
            to_remove = rng.sample(range(len(perturbed)), n_remove)
            to_remove_set = set(to_remove)
            perturbed = [x for i, x in enumerate(perturbed) if i not in to_remove_set]

        cov = [0] * n_t
        for idx in perturbed:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb

        uncovered = full_mask
        for k2 in range(n_t):
            if cov[k2] == 0:
                uncovered |= (1 << k2)

        while uncovered != 0 and time.time() < deadline - 15:
            best_idx = -1
            best_gain = -1
            for j in range(n_k):
                if j in perturbed:
                    continue
                gain = (block_masks[j] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            if best_idx == -1 or best_gain == 0:
                break
            perturbed.append(best_idx)
            uncovered &= ~block_masks[best_idx]

        perturbed = prune_blocks(perturbed)
        perturbed = local_search(perturbed)

        if time.time() < deadline - 15:
            perturbed2 = dual_local_search(perturbed)
            if len(perturbed2) < len(perturbed):
                perturbed = perturbed2

        if len(perturbed) < best_len:
            best_solution = perturbed
            best_len = len(perturbed)

    # Final validation
    final_cov = [0] * n_t
    for idx in best_solution:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            k2 = lsb.bit_length() - 1
            final_cov[k2] += 1
            m ^= lsb

    for k2 in range(n_t):
        if final_cov[k2] == 0:
            for j in range(n_k):
                if (block_masks[j] >> k2) & 1:
                    if j not in best_solution:
                        best_solution.append(j)
                    break

    blocks = [list(block_data[i][0]) for i in best_solution]

    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)

    return unique_blocks
