import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate all weight-4 vectors
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, s):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute conflict matrix
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Greedy with min-degree
    def greedy_construct(rng):
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        count = 0

        while True:
            if time.time() > deadline - 20.0:
                break
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break

            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1

            min_deg = deg.min()
            if min_deg > n:
                break

            cands = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(cands) == 0:
                break

            pick = rng.choice(cands)
            selected[pick] = 1
            remaining[pick] = 0
            remaining[conflict[pick] == 1] = 0
            count += 1

        return selected, count

    # Simulated annealing local search
    def local_search_sa(selected, rng, time_budget):
        sel = selected.copy()
        current_count = int(np.sum(sel))
        best_count = current_count
        best_sel = sel.copy()

        t_end = time.time() + time_budget
        temperature = 10.0
        cooling = 0.9995

        while time.time() < t_end:
            sel_indices = np.where(sel == 1)[0]
            k_sel = len(sel_indices)
            if k_sel == 0:
                break

            conflict_with_S = conflict[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)

            # Try 1-for-2: remove v, add u, w
            v_idx = rng.integers(0, k_sel)
            v = sel_indices[v_idx]

            # Nodes compatible with S\{v}
            cand_mask = ((total_inc == 0) | ((total_inc == 1) & (conflict_with_S[:, v_idx] == 1))) & (sel == 0)
            cand_indices = np.where(cand_mask)[0]

            if len(cand_indices) >= 2:
                # Find compatible pair
                sub_conf = conflict[np.ix_(cand_indices, cand_indices)]
                zero_pos = np.argwhere(sub_conf == 0)
                valid = [(r, c) for r, c in zero_pos if r < c]

                if valid:
                    r, c = valid[rng.integers(0, len(valid))]
                    u, w = cand_indices[r], cand_indices[c]

                    new_sel = sel.copy()
                    new_sel[v] = 0
                    new_sel[u] = 1
                    new_sel[w] = 1
                    new_count = int(np.sum(new_sel))

                    delta = new_count - current_count
                    if delta > 0 or rng.random() < np.exp(delta / temperature):
                        sel = new_sel
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_sel = sel.copy()
            else:
                # 1-for-1 swap
                cand_mask = ((total_inc == 0) | ((total_inc == 1) & (conflict_with_S[:, v_idx] == 1))) & (sel == 0)
                cand_indices = np.where(cand_mask)[0]
                if len(cand_indices) > 0:
                    u = cand_indices[rng.integers(0, len(cand_indices))]
                    new_sel = sel.copy()
                    new_sel[v] = 0
                    new_sel[u] = 1
                    new_count = int(np.sum(new_sel))
                    delta = new_count - current_count
                    if delta >= 0 or rng.random() < np.exp(delta / temperature):
                        sel = new_sel
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_sel = sel.copy()

            temperature *= cooling

        return best_sel, best_count

    # Run multiple greedy + SA cycles
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)

    rng = np.random.default_rng(42)

    # Phase 1: Multiple greedy starts
    n_greedy = 30
    for i in range(n_greedy):
        if time.time() > deadline - 30.0:
            break
        sel, cnt = greedy_construct(rng)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()

    # Phase 2: SA refinement on best and top candidates
    if best_count > 0:
        remaining_time = deadline - time.time() - 5.0
        sa_budget = max(remaining_time * 0.4, 5.0)
        best_selected, best_count = local_search_sa(best_selected, rng, sa_budget)

    # Phase 3: More SA restarts if time permits
    if time.time() < deadline - 10.0:
        for i in range(5):
            if time.time() > deadline - 10.0:
                break
            sel, cnt = greedy_construct(rng)
            if cnt >= best_count - 5:  # Only refine good starts
                sa_budget = min(deadline - time.time() - 5.0, 10.0)
                if sa_budget > 2.0:
                    sel2, cnt2 = local_search_sa(sel, rng, sa_budget)
                    if cnt2 > best_count:
                        best_count = cnt2
                        best_selected = sel2

    # Add Type A vectors (±2e_i)
    type_a_vectors = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type_a_vectors.append(v)
    type_a_arr = np.array(type_a_vectors, dtype=np.int64)

    # Combine
    if best_count > 0:
        type_b_selected = candidates[best_selected == 1].astype(np.int64)
        all_vectors = np.vstack([type_a_arr, type_b_selected])
    else:
        all_vectors = type_a_arr

    # Verify
    norms_sq = np.sum(all_vectors * all_vectors, axis=1)
    R_sq = np.max(norms_sq)
    k = len(all_vectors)
    dots = all_vectors @ all_vectors.T
    num = norms_sq[:, None] + norms_sq[None, :] - R_sq
    thresh = num // 2
    i_idx, j_idx = np.triu_indices(k, k=1)
    valid = np.all(dots[i_idx, j_idx] <= thresh[i_idx, j_idx])

    if not valid:
        return type_a_arr

    return all_vectors
