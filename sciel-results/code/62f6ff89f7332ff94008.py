"""Kissing number in dimension 11: greedy + swap local search for large independent sets."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit

    D = 11

    # Generate candidate vectors
    candidates_list = []

    # Weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-3 vectors (norm^2=3): C(11,3)*8 = 1320
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected (i != j)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Precompute full degree for each node
    full_deg = conflict.sum(axis=1)

    def greedy_min_degree(remaining_init, time_limit):
        """Min-degree greedy: pick vertex with fewest conflicts among remaining."""
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    def swap_local_search(selected_list, time_limit, max_rounds=500):
        """Swap-based local search: try to grow by swapping one selected vector for two."""
        selected = list(selected_list)
        sel_set = set(selected)
        # compat_with_all[j] = 1 if j is compatible with all currently selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            # Try simple addition first
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue

            # Try swap: remove one selected, add two new ones
            found_swap = False
            if len(selected) > 30:
                trial_indices = np.random.choice(len(selected), size=min(15, len(selected)), replace=False)
            else:
                trial_indices = range(len(selected))

            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                # Find candidates compatible with all selected except s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[s] = 0
                mask_except_s[list(sel_set)] = 0

                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue

                # Sort by full degree (least constraining first)
                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]

                found_pair = False
                for i in range(min(25, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    compat_with_c1 = compat[c1]
                    for j in range(i + 1, min(50, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if compat_with_c1[c2]:
                            # Do the swap
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)
                            # Recompute compat_with_all
                            compat_with_all = np.ones(n, dtype=np.uint8)
                            for s3 in selected:
                                compat_with_all = compat_with_all & compat[s3]
                            for s3 in selected:
                                compat_with_all[s3] = 0
                            found_swap = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return selected

    def remove_add_local_search(selected_list, time_limit, max_rounds=300):
        """Try removing 2 and adding 3, or removing 1 and adding 2 with different pairing."""
        selected = list(selected_list)
        sel_set = set(selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break

            # Try adding one
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue

            # Try 2-for-3 swap: remove 2 selected, add 3 new
            found_swap = False
            if len(selected) > 40:
                trial_pairs = [(random.randint(0, len(selected)-1), random.randint(0, len(selected)-1)) for _ in range(10)]
            else:
                trial_pairs = [(i, j) for i in range(len(selected)) for j in range(i+1, len(selected))]
            
            for si, sj in trial_pairs:
                if time.time() > time_limit:
                    break
                if si == sj:
                    continue
                s1 = selected[si]
                s2 = selected[sj]
                
                # Find candidates compatible with all selected except s1 and s2
                mask_except = np.ones(n, dtype=np.uint8)
                for s3 in selected:
                    if s3 != s1 and s3 != s2:
                        mask_except = mask_except & compat[:, s3]
                mask_except[s1] = 0
                mask_except[s2] = 0
                mask_except[list(sel_set)] = 0

                cands = np.where(mask_except > 0)[0]
                if len(cands) < 3:
                    continue

                sorted_cands = cands[np.argsort(full_deg[cands])]
                found_triple = False
                for i in range(min(15, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    for j in range(i+1, min(30, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if not compat[c1, c2]:
                            continue
                        for k in range(j+1, min(45, len(sorted_cands))):
                            c3 = sorted_cands[k]
                            if compat[c1, c3] and compat[c2, c3]:
                                # Do the swap
                                selected.remove(s1)
                                selected.remove(s2)
                                sel_set.discard(s1)
                                sel_set.discard(s2)
                                for c in [c1, c2, c3]:
                                    selected.append(c)
                                    sel_set.add(c)
                                compat_with_all = np.ones(n, dtype=np.uint8)
                                for s4 in selected:
                                    compat_with_all = compat_with_all & compat[s4]
                                for s4 in selected:
                                    compat_with_all[s4] = 0
                                found_swap = True
                                found_triple = True
                                break
                        if found_triple:
                            break
                    if found_triple:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return selected

    # Main search
    best_selected = []
    np.random.seed(42)
    random.seed(42)

    # Strategy 1: Pure min-degree greedy
    sel = greedy_min_degree(np.ones(n, dtype=np.uint8), deadline - 60.0)
    if len(sel) > len(best_selected):
        best_selected = sel

    # Strategy 2: Multiple randomized greedy restarts
    for trial in range(80):
        if time.time() > deadline - 30.0:
            break
        frac = np.random.uniform(0.5, 1.0)
        subset_size = int(n * frac)
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        remaining_init = np.zeros(n, dtype=np.uint8)
        remaining_init[subset_indices] = 1
        sel = greedy_min_degree(remaining_init, deadline - 30.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # Strategy 3: Apply swap local search to the best
    if best_selected and time.time() < deadline - 15.0:
        improved = swap_local_search(best_selected, deadline - 15.0, max_rounds=400)
        if len(improved) > len(best_selected):
            best_selected = improved

    # Strategy 4: Apply 2-for-3 local search
    if best_selected and time.time() < deadline - 5.0:
        improved = remove_add_local_search(best_selected, deadline - 5.0, max_rounds=200)
        if len(improved) > len(best_selected):
            best_selected = improved

    # Final greedy fill
    if best_selected and time.time() < deadline - 2.0:
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 2.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
