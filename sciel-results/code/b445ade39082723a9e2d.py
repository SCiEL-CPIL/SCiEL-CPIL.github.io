import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 140  # safety margin under 150s kill

    # 1. Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # 2. Precompute all k-blocks and their coverage (as numpy int32 arrays)
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # coverage: n_k x n_t int32 matrix (0/1)
    coverage = np.zeros((n_k, n_t), dtype=np.int32)
    # For each t-subset, list of blocks containing it (for swap search)
    blocks_containing = [[] for _ in range(n_t)]

    for i, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            coverage[i, idx] = 1
            blocks_containing[idx].append(i)

    # Per-block coverage count (how many t-subsets each block covers)
    block_cov_count = coverage.sum(axis=1)

    # 3. Greedy construction (vectorized) with multiple variants
    def greedy(weight_mode='max', seed=None):
        if seed is not None:
            random.seed(seed)
        remaining = np.ones(n_t, dtype=np.int32)
        selected = []
        while remaining.sum() > 0:
            if time.time() > deadline - 30:
                break
            if weight_mode == 'max':
                gains = (coverage * remaining).sum(axis=1)
            else:  # 'inv': weight each uncovered subset by 1/(current cover count)
                # current cover count of each t-subset among selected blocks
                cover = np.zeros(n_t, dtype=np.int32)
                for b in selected:
                    cover += coverage[b]
                # weight: subsets with fewer covers are more valuable
                w = 1.0 / (cover + 1)
                gains = (coverage * remaining * w).sum(axis=1)
            # random tie-breaking
            idx = np.argmax(gains + np.random.rand(n_k) * 0.5)
            if gains[idx] == 0:
                break
            selected.append(int(idx))
            remaining -= coverage[idx]
            remaining = np.maximum(remaining, 0)
        return selected

    best_solution = None
    best_size = float('inf')

    # Run several greedy variants for diversity
    variants = [('max', 0), ('max', 1), ('max', 2), ('inv', 0), ('inv', 1)]
    for mode, seed in variants:
        if time.time() > deadline - 40:
            break
        sol = greedy(weight_mode=mode, seed=seed)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]

    if best_solution is None:
        # fallback: pure max greedy
        best_solution = greedy(weight_mode='max', seed=0)
        best_size = len(best_solution)

    # 4. Redundancy removal (iterative)
    def remove_redundant(selected):
        if not selected:
            return selected
        # cover count per t-subset
        cover = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover += coverage[b]
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            new_sel = []
            for b in selected:
                # check if any t-subset is covered only by b
                mask = coverage[b]
                # find t-subsets where mask==1 and cover==1
                crit = np.any((mask == 1) & (cover == 1))
                if crit:
                    new_sel.append(b)
                else:
                    cover -= mask
                    changed = True
            selected = new_sel
        return selected

    current_blocks = remove_redundant(best_solution[:])

    # 5. Single-block swap local search
    def single_swap_search(selected, max_rounds=3):
        if time.time() > deadline - 20:
            return selected
        selected_set = set(selected)
        for _round in range(max_rounds):
            if time.time() > deadline - 15:
                break
            # cover count
            cover = np.zeros(n_t, dtype=np.int32)
            for b in selected:
                cover += coverage[b]
            improved = False
            for i, a in enumerate(selected):
                if time.time() > deadline - 15:
                    break
                # critical subsets of a
                crit_mask = (coverage[a] == 1) & (cover == 1)
                if not crit_mask.any():
                    continue
                crit_indices = np.where(crit_mask)[0]
                # find unselected block b that covers all crit_indices
                # and has max extra coverage
                extra_gain = -1
                best_b = -1
                # candidate blocks: those containing all crit subsets
                # use blocks_containing for the first crit subset as candidates
                if len(crit_indices) > 0:
                    candidates = set(blocks_containing[int(crit_indices[0])])
                    for c in crit_indices[1:]:
                        candidates &= set(blocks_containing[int(c)])
                    candidates -= selected_set
                    for b in candidates:
                        extra = int((coverage[b] & ~coverage[a]).sum())
                        if extra > extra_gain:
                            extra_gain = extra
                            best_b = b
                if best_b != -1 and extra_gain > 0:
                    selected[i] = best_b
                    selected_set.discard(a)
                    selected_set.add(best_b)
                    improved = True
                    break
            if not improved:
                break
        return selected

    current_blocks = single_swap_search(current_blocks)
    current_blocks = remove_redundant(current_blocks)

    # 6. Pairwise swap: remove two blocks, add one that covers union of their criticals
    def pair_swap_search(selected, max_rounds=2):
        if time.time() > deadline - 20:
            return selected
        n = len(selected)
        for _round in range(max_rounds):
            if time.time() > deadline - 15:
                break
            cover = np.zeros(n_t, dtype=np.int32)
            for b in selected:
                cover += coverage[b]
            improved = False
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, n):
                    a = selected[i]
                    c = selected[j]
                    # union of their critical subsets
                    crit_a = (coverage[a] == 1) & (cover == 1)
                    crit_c = (coverage[c] == 1) & (cover == 1)
                    crit_union = crit_a | crit_c
                    if not crit_union.any():
                        continue
                    crit_indices = np.where(crit_union)[0]
                    # find unselected block covering all crit_indices
                    candidates = set(blocks_containing[int(crit_indices[0])])
                    for ci in crit_indices[1:]:
                        candidates &= set(blocks_containing[int(ci)])
                    candidates -= set(selected)
                    best_b = -1
                    best_extra = -1
                    for b in candidates:
                        extra = int((coverage[b] & ~(coverage[a] | coverage[c])).sum())
                        if extra > best_extra:
                            best_extra = extra
                            best_b = b
                    if best_b != -1 and best_extra > 0:
                        # replace a and c with b
                        selected = [x for x in selected if x != a and x != c]
                        selected.append(best_b)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return selected

    if time.time() < deadline - 20:
        current_blocks = pair_swap_search(current_blocks)
        current_blocks = remove_redundant(current_blocks)

    # 7. Final redundancy removal
    current_blocks = remove_redundant(current_blocks)

    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    return blocks
