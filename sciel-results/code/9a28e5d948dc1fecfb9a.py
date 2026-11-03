import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Precompute which t-subsets each k-subset covers
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]

    # For each t-subset, precompute which blocks cover it (inverse index)
    # This is critical for efficient local search
    t_to_blocks = [set() for _ in range(n_t)]
    for i in range(n_k):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].add(i)

    # Convert to lists for faster iteration
    t_to_blocks = [list(s) for s in t_to_blocks]

    # Frequency of each t-subset (how many blocks cover it)
    t_freq = np.array([len(b) for b in t_to_blocks], dtype=np.float64)
    t_freq[t_freq == 0] = 1.0

    # Weighted greedy: prioritize blocks that cover rare t-subsets
    def weighted_greedy():
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        coverage = np.zeros(n_t, dtype=np.int32)

        while uncovered.any() and time.time() < deadline - 20.0:
            best_gain = -1
            best_idx = -1

            # Compute weighted gain for each block
            # gain = sum of weights of uncovered t-subsets covered by block
            # Weight = 1 / freq (rarer subsets get higher weight)
            for i in range(n_k):
                if not uncovered.any():
                    break
                mask = block_t_indices[i]
                # Only consider uncovered ones
                unc_mask = uncovered[mask]
                if not unc_mask.any():
                    continue
                # Weighted gain
                gains = (1.0 / t_freq[mask[unc_mask]])
                gain = gains.sum()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            coverage[block_t_indices[best_idx]] += 1
            uncovered[block_t_indices[best_idx]] = False

        return selected, coverage

    # Remove redundant blocks
    def remove_redundant(selected, coverage):
        if not selected:
            return selected, coverage

        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for bi in selected:
                if np.min(coverage[block_t_indices[bi]]) > 1:
                    coverage[block_t_indices[bi]] -= 1
                    changed = True
                else:
                    new_selected.append(bi)
            selected = new_selected
        return selected, coverage

    # Local search: try to swap blocks to reduce size
    def local_search(selected, coverage, max_iters=5000):
        if not selected:
            return selected, coverage

        selected = list(selected)
        coverage = coverage.copy()
        current_size = len(selected)
        no_improve_count = 0
        max_no_improve = 200

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if no_improve_count > max_no_improve:
                break

            improved = False

            # Try removing each block and see if we can repair with fewer additions
            for idx in range(len(selected)):
                if time.time() > deadline - 10.0:
                    break

                bi = selected[idx]
                # Tentatively remove
                coverage[block_t_indices[bi]] -= 1
                uncovered_after_remove = (coverage == 0).sum()

                if uncovered_after_remove == 0:
                    # Block was redundant
                    selected.pop(idx)
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Try to find a single block that covers all uncovered
                    # and doesn't introduce too much redundancy
                    uncovered_indices = np.where(coverage == 0)[0]

                    # For each uncovered t-subset, find candidate blocks
                    # We want a block that covers all of them
                    candidate_set = None
                    for t_idx in uncovered_indices:
                        candidates = t_to_blocks[t_idx]
                        # Exclude already selected blocks (except we're removing bi)
                        cand_set = set(candidates) - set(selected)
                        if candidate_set is None:
                            candidate_set = cand_set
                        else:
                            candidate_set &= cand_set

                        if not candidate_set:
                            break

                    if candidate_set:
                        # Find best candidate: one that covers most uncovered
                        best_cand = None
                        best_cover = -1
                        for cand in candidate_set:
                            # How many of the uncovered does it cover?
                            covers = np.sum(coverage[block_t_indices[cand]] == 0)
                            if covers > best_cover:
                                best_cover = covers
                                best_cand = cand

                        if best_cand is not None and best_cover >= len(uncovered_indices):
                            # Replace bi with best_cand
                            selected[idx] = best_cand
                            coverage[block_t_indices[best_cand]] += 1
                            improved = True
                            no_improve_count = 0
                            break

                    # Restore if not improved
                    coverage[block_t_indices[bi]] += 1

            if not improved:
                no_improve_count += 1

        return selected, coverage

    # ILS with perturbation
    def ils_improve(selected, coverage, n_iters=50):
        best_selected = list(selected)
        best_coverage = coverage.copy()
        best_size = len(selected)

        rng = np.random.default_rng(123)

        for it in range(n_iters):
            if time.time() > deadline - 15.0:
                break

            # Perturbation: remove m random blocks
            m = max(1, len(selected) // 10)
            m = min(m, len(selected) - 5) if len(selected) > 5 else 0

            perturbed = list(selected)
            pert_coverage = coverage.copy()

            if m > 0:
                remove_indices = rng.choice(len(perturbed), size=m, replace=False)
                for idx in sorted(remove_indices, reverse=True):
                    bi = perturbed[idx]
                    pert_coverage[block_t_indices[bi]] -= 1
                    perturbed.pop(idx)

            # Greedy repair
            uncovered = (pert_coverage == 0)
            while uncovered.any() and time.time() < deadline - 15.0:
                # Find block with max coverage of uncovered
                best_gain = -1
                best_idx = -1

                # Only consider blocks not already in perturbed
                in_perturbed = set(perturbed)

                for i in range(n_k):
                    if i in in_perturbed:
                        continue
                    mask = block_t_indices[i]
                    unc_mask = uncovered[mask]
                    if not unc_mask.any():
                        continue
                    gain = unc_mask.sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i

                if best_idx == -1:
                    break

                perturbed.append(best_idx)
                pert_coverage[block_t_indices[best_idx]] += 1
                uncovered = (pert_coverage == 0)

            # Local search on perturbed
            perturbed, pert_coverage = local_search(perturbed, pert_coverage, max_iters=1000)

            if len(perturbed) < best_size:
                best_selected = perturbed
                best_coverage = pert_coverage
                best_size = len(perturbed)

        return best_selected, best_coverage

    # Main algorithm
    # 1. Weighted greedy initialization
    initial_selected, initial_coverage = weighted_greedy()

    # 2. Remove redundants
    initial_selected, initial_coverage = remove_redundant(initial_selected, initial_coverage)

    best_selected = initial_selected
    best_coverage = initial_coverage
    best_size = len(initial_selected)

    # 3. Local search
    ls_selected, ls_coverage = local_search(initial_selected, initial_coverage)
    if len(ls_selected) < best_size:
        best_selected = ls_selected
        best_coverage = ls_coverage
        best_size = len(ls_selected)

    # 4. ILS
    if time.time() < deadline - 20.0:
        ils_selected, ils_coverage = ils_improve(best_selected, best_coverage)
        if len(ils_selected) < best_size:
            best_selected = ils_selected
            best_coverage = ils_coverage
            best_size = len(ils_selected)

    # Final redundancy removal
    best_selected, best_coverage = remove_redundant(best_selected, best_coverage)

    # Verify coverage
    if np.all(best_coverage > 0):
        return [list(k_subsets[i]) for i in best_selected]

    # Fallback: if something went wrong, return initial
    if np.all(initial_coverage > 0):
        return [list(k_subsets[i]) for i in initial_selected]

    return []
