"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Generate all 4-subsets of range(11)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)

    # Build conflict matrix: edge if intersection size == 3
    # Two 4-subsets intersect in 3 elements iff they share 3 and differ in 1
    Conflict = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i + 1, n_sup):
            s_j = set(supports[j])
            if len(s_i & s_j) == 3:
                Conflict[i, j] = 1
                Conflict[j, i] = 1

    # Precompute degree
    deg_initial = Conflict.sum(axis=1).astype(np.int64)

    def run_greedy(seed):
        if time.time() > deadline:
            return []
        rng = np.random.default_rng(seed)
        active = np.ones(n_sup, dtype=bool)
        deg = deg_initial.copy()
        selected = []

        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            selected.append(idx)

            # Remove idx and its neighbors
            removed_mask = np.zeros(n_sup, dtype=bool)
            removed_mask[idx] = True
            removed_mask[Conflict[idx].astype(bool) & active] = True
            removed_indices = np.where(removed_mask)[0]

            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]

            active[removed_mask] = False

        return selected

    def local_search(selected_indices, max_iters=80):
        if time.time() > deadline:
            return list(selected_indices)

        selected_set = set(selected_indices)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters * 12345 + 7)
            rng_ls.shuffle(indices_list)

            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)

                # Find j not in selected_set that conflicts ONLY with i
                sel_mask = np.zeros(n_sup, dtype=bool)
                sel_mask[list(selected_set)] = True

                non_sel = np.where(~sel_mask)[0]
                if len(non_sel) == 0:
                    continue

                # conflicts of each non-selected j with selected set
                conf_with_sel = Conflict[non_sel, :].sum(axis=1)
                single_conf = non_sel[conf_with_sel == 1]

                valid_j = []
                for j in single_conf:
                    conflicting_sel = np.where(sel_mask & Conflict[j].astype(bool))[0]
                    if len(conflicting_sel) == 1 and conflicting_sel[0] == i:
                        valid_j.append(j)

                # Try to find two valid_j that are compatible with each other
                if len(valid_j) >= 2:
                    found = False
                    for a_idx in range(len(valid_j)):
                        if found:
                            break
                        for b_idx in range(a_idx + 1, len(valid_j)):
                            ja, jb = valid_j[a_idx], valid_j[b_idx]
                            if Conflict[ja, jb] == 0:
                                selected_set.remove(i)
                                selected_set.add(ja)
                                selected_set.add(jb)
                                improved = True
                                found = True
                                break

                if improved:
                    break  # Restart the loop

        return list(selected_set)

    def local_search_2for1(selected_indices, max_iters=40):
        """Try to replace two selected supports with three non-conflicting ones."""
        if time.time() > deadline:
            return list(selected_indices)

        selected_set = set(selected_indices)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters * 54321 + 13)
            rng_ls.shuffle(indices_list)

            # Try pairs of selected elements
            for a_pos in range(len(indices_list)):
                if time.time() > deadline:
                    return list(selected_set)
                i = indices_list[a_pos]
                for b_pos in range(a_pos + 1, len(indices_list)):
                    j = indices_list[b_pos]
                    if Conflict[i, j] == 1:
                        continue  # i and j already conflict, can't both be selected

                    sel_mask = np.zeros(n_sup, dtype=bool)
                    sel_mask[list(selected_set)] = True

                    non_sel = np.where(~sel_mask)[0]
                    if len(non_sel) == 0:
                        continue

                    # For each non-selected k, count conflicts with selected set
                    conf_with_sel = Conflict[non_sel, :].sum(axis=1)
                    # We want k whose conflicts with selected set are a subset of {i, j}
                    # and at least one of i or j
                    candidate_k = non_sel[(conf_with_sel >= 1) & (conf_with_sel <= 2)]

                    valid_k = []
                    for k in candidate_k:
                        conflicting_sel = np.where(sel_mask & Conflict[k].astype(bool))[0]
                        if len(conflicting_sel) == 0:
                            continue
                        if all(c in (i, j) for c in conflicting_sel):
                            valid_k.append(k)

                    if len(valid_k) >= 3:
                        # Try to find three mutually compatible k's
                        found = False
                        for x in range(len(valid_k)):
                            if found:
                                break
                            k1 = valid_k[x]
                            for y in range(x + 1, len(valid_k)):
                                if found:
                                    break
                                k2 = valid_k[y]
                                if Conflict[k1, k2] == 1:
                                    continue
                                for z in range(y + 1, len(valid_k)):
                                    k3 = valid_k[z]
                                    if Conflict[k1, k3] == 1 or Conflict[k2, k3] == 1:
                                        continue
                                    # Also need k1,k2,k3 to not conflict with each other
                                    # and to be compatible with rest of selected set
                                    # (they already are by construction)
                                    selected_set.discard(i)
                                    selected_set.discard(j)
                                    selected_set.add(k1)
                                    selected_set.add(k2)
                                    selected_set.add(k3)
                                    improved = True
                                    found = True
                                    break

                    if improved:
                        break

                if improved:
                    break

        return list(selected_set)

    best_set = []
    best_size = 0

    # Multiple random restarts with greedy
    n_restarts = 300
    for seed in range(n_restarts):
        if time.time() > deadline - 15:
            break
        result = run_greedy(seed)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result

    # Local search on best
    if time.time() < deadline - 10:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set

    # 2-for-1 local search
    if time.time() < deadline - 10:
        improved_set = local_search_2for1(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set

    # Additional restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for seed in range(n_restarts, n_restarts + 300):
            if time.time() > deadline - 10:
                break
            result = run_greedy(seed)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
                if time.time() < deadline - 10:
                    improved_set = local_search(best_set)
                    if len(improved_set) > best_size:
                        best_size = len(improved_set)
                        best_set = improved_set

    # Construct vectors from best independent set of supports
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)

    if not vectors:
        # Fallback: return seed solution
        seed_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    seed_vectors.append(v)
        return np.array(seed_vectors, dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
