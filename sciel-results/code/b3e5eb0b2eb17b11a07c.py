import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150 s kill

    # ---- Build candidate pool: all integer vectors with squared norm 4 ----
    candidates = []

    # Type 1: one entry is ±2, rest 0  -> norm^2 = 4
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: four entries are ±1, rest 0  -> norm^2 = 4
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # ---- Conflict matrix: conflict iff dot > 0 ----
    # (equal norm^2=4, need ||u-v||^2 >= 4  <=>  u.v <= 0)
    M = candidates @ candidates.T
    Conflict = (M > 0).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)

    def run_greedy(seed):
        if time.time() > deadline:
            return []
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
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
            idx = int(rng.choice(candidates_for_pick))

            selected.append(idx)

            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True

            removed_indices = np.where(removed_mask)[0]
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]

            active[removed_mask] = False

        return selected

    # ---- Local search: add compatible vectors, then 1-for-2 swaps ----
    def local_search(selected_indices):
        if time.time() > deadline:
            return list(selected_indices)
        selected_set = set(selected_indices)

        # Phase 1: add any unselected vector compatible with all selected
        improved = True
        while improved and time.time() < deadline:
            improved = False
            sel_arr = np.array(list(selected_set), dtype=np.int64)
            if len(sel_arr) > 0:
                # conflicts of each unselected with the selected set
                conflict_counts = Conflict[sel_arr, :].sum(axis=0)
            else:
                conflict_counts = np.zeros(N, dtype=np.int64)
            compatible_unselected = np.where((conflict_counts == 0) & (~np.isin(np.arange(N), np.array(list(selected_set), dtype=np.int64))))[0]
            for j in compatible_unselected:
                selected_set.add(int(j))
                improved = True
            if improved:
                continue

        # Phase 2: 1-for-2 swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng()
            rng_ls.shuffle(indices_list)
            for i in indices_list:
                if time.time() > deadline:
                    break
                if i not in selected_set:
                    continue
                conflicts_with_i = Conflict[i].astype(bool)
                neighbors_of_i = np.where(conflicts_with_i)[0]
                valid_js = [j for j in neighbors_of_i if j not in selected_set]

                compatible_with_rest = []
                for j in valid_js:
                    conflict_count = 0
                    for k in selected_set:
                        if k == i:
                            continue
                        if Conflict[j, k]:
                            conflict_count += 1
                            break
                    if conflict_count == 0:
                        compatible_with_rest.append(j)

                found_pair = False
                for a in range(len(compatible_with_rest)):
                    if time.time() > deadline:
                        break
                    j1 = compatible_with_rest[a]
                    for b in range(a + 1, len(compatible_with_rest)):
                        j2 = compatible_with_rest[b]
                        if not Conflict[j1, j2]:
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    break

        return list(selected_set)

    # ---- Run multiple greedy seeds, keep best, local-search it ----
    best_size = 0
    best_set = []

    for seed in range(40):
        if time.time() > deadline:
            break
        sel = run_greedy(seed)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel

    # Local search on the best greedy set
    if time.time() < deadline - 5:
        best_set = local_search(best_set)

    if len(best_set) == 0:
        # Fallback: return a trivially valid small set (weight-2 vectors)
        vecs = []
        for pos in itertools.combinations(range(D), 2):
            p1, p2 = pos
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[p1] = s1
                    v[p2] = s2
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    result = candidates[np.array(best_set, dtype=np.int64)]
    return result.astype(np.int64)
