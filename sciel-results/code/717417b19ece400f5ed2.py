import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # ---- Generate candidate vectors (all have norm^2 = 4) ----
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)
    if N4 == 0:
        return vectors_v1

    # ---- Conflict graph: two W4 vectors conflict if their dot > 2 ----
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # ---------------- Greedy initializers ----------------
    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        return in_set

    def greedy_max_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        return in_set

    # ---------------- Shared O(1) not-in-set structure ----------------
    def make_nis(in_set):
        not_in_set_list = [int(i) for i in np.where(~in_set)[0]]
        pos_in_nis = np.full(N4, -1, dtype=np.int32)
        for idx, node in enumerate(not_in_set_list):
            pos_in_nis[node] = idx

        def remove_from_nis(v):
            p = pos_in_nis[v]
            if p < 0:
                return
            last = not_in_set_list[-1]
            not_in_set_list[p] = last
            pos_in_nis[last] = p
            not_in_set_list.pop()
            pos_in_nis[v] = -1

        def add_to_nis(v):
            if pos_in_nis[v] >= 0:
                return
            pos_in_nis[v] = len(not_in_set_list)
            not_in_set_list.append(v)

        return not_in_set_list, pos_in_nis, remove_from_nis, add_to_nis

    def build_counts(in_set):
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        return count_conflicts

    # ---------------- Local search: add + SA + 2-for-1 swaps ----------------
    def local_search(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        in_set = in_set_init.copy()
        in_mask = in_set.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        count_conflicts = build_counts(in_set)
        nis_list, pos_in_nis, remove_from_nis, add_to_nis = make_nis(in_set)

        T = 3.0
        stuck_counter = 0
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            if len(nis_list) == 0:
                break

            nis_arr = np.asarray(nis_list, dtype=np.int32)
            counts = count_conflicts[nis_arr]
            min_count = int(np.min(counts))

            if min_count == 0:
                zero_cands = nis_arr[counts == 0]
                v = int(zero_cands[rng.integers(len(zero_cands))])
                in_set[v] = True
                in_mask[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                remove_from_nis(v)
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                T *= cooling
                if T < min_T:
                    T = min_T
                continue

            # 2-for-1 exchange: remove one in-set node, add its now-free neighbors
            in_indices = np.where(in_mask)[0]
            if len(in_indices) > 0 and rng.random() < 0.5:
                i = int(in_indices[rng.integers(len(in_indices))])
                in_set[i] = False
                in_mask[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1
                add_to_nis(i)

                nis_arr = np.asarray(nis_list, dtype=np.int32)
                counts = count_conflicts[nis_arr]
                addable = nis_arr[counts == 0]
                gain = int(len(addable))
                delta = gain - 1
                if delta >= 0:
                    for v in addable:
                        v = int(v)
                        in_set[v] = True
                        in_mask[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        remove_from_nis(v)
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    # revert
                    in_set[i] = True
                    in_mask[i] = True
                    current_size += 1
                    count_conflicts[neighbors[i]] += 1
                    remove_from_nis(i)
                    stuck_counter += 1
                T *= cooling
                if T < min_T:
                    T = min_T
                continue

            # SA single move: add node v, evict its in-set neighbors
            min_cands = nis_arr[counts == min_count]
            v = int(min_cands[rng.integers(len(min_cands))])
            k = min_count
            delta = 1 - k

            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)

            if accept:
                nbrs_v = neighbors[v]
                to_remove = nbrs_v[in_mask[nbrs_v]]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    in_mask[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    for u in to_remove:
                        add_to_nis(int(u))
                in_set[v] = True
                in_mask[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                remove_from_nis(v)
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # ---------------- Exact polish: free adds + 1-for-1 swaps ----------------
    def exact_polish(in_set_init, max_time=5.0):
        in_set = in_set_init.copy()
        in_mask = in_set.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        count_conflicts = build_counts(in_set)
        nis_list, pos_in_nis, remove_from_nis, add_to_nis = make_nis(in_set)
        polish_deadline = time.time() + max_time

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # free additions
            if len(nis_list) > 0:
                nis_arr = np.asarray(nis_list, dtype=np.int32)
                counts = count_conflicts[nis_arr]
                zero_cands = nis_arr[counts == 0]
                for v in zero_cands:
                    v = int(v)
                    in_set[v] = True
                    in_mask[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                    remove_from_nis(v)
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            # 1-for-1 swaps: remove i, add a single non-neighbor
            in_set_indices = np.where(in_mask)[0].copy()
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                in_set[i] = False
                in_mask[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1
                add_to_nis(int(i))

                if len(nis_list) > 0:
                    nis_arr = np.asarray(nis_list, dtype=np.int32)
                    counts = count_conflicts[nis_arr]
                    addable = nis_arr[counts == 0]
                    addable = addable[addable != i]
                    if len(addable) > 0:
                        v = int(addable[0])
                        in_set[v] = True
                        in_mask[v] = True
                        count_conflicts[neighbors[v]] += 1
                        remove_from_nis(v)
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                    else:
                        in_set[i] = True
                        in_mask[i] = True
                        current_size += 1
                        count_conflicts[neighbors[i]] += 1
                        remove_from_nis(int(i))
                else:
                    in_set[i] = True
                    in_mask[i] = True
                    current_size += 1
                    count_conflicts[neighbors[i]] += 1
                    remove_from_nis(int(i))

        return local_best_in_set, local_best_size

    # ---------------- Main loop ----------------
    n_restarts = 0
    max_restarts = 80

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        if n_restarts % 3 == 0:
            in_set = greedy_max_degree()
        else:
            in_set = greedy_min_degree()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.4, 12.0)
        if sa_time < 2:
            sa_time = 2.0
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # second SA polish on the SA result
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size >= current_size:
            polish_time = min(remaining_time * 0.35, 8.0)
            if polish_time > 1:
                polish_result, polish_size = local_search(
                    sa_result, max_time=polish_time, min_T=0.02, cooling=0.9998)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

        # exact polish periodically
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 2 and n_restarts % 5 == 0 and best_size > 0:
            ep_time = min(remaining_time * 0.3, 4.0)
            if ep_time > 0.5:
                ep_result, ep_size = exact_polish(best_in_set, max_time=ep_time)
                if ep_size > best_size:
                    best_size = ep_size
                    best_in_set = ep_result.copy()

    # Final exact polish on the best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        final_polish_time = min(remaining_time * 0.85, 6.0)
        if final_polish_time > 0.5:
            final_result, final_size = exact_polish(best_in_set, max_time=final_polish_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_result

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    # ---- Exact validity verification (drop any violating vector) ----
    if len(final_vectors) > 1:
        max_norm_sq = int(np.max(np.sum(final_vectors.astype(np.int64) ** 2, axis=1)))
        dists_sq = np.sum(final_vectors.astype(np.int64) ** 2, axis=1)
        dists_sq = dists_sq[:, None] + dists_sq[None, :] - 2 * (final_vectors.astype(np.int64) @ final_vectors.astype(np.int64).T)
        dists_sq = np.maximum(dists_sq, 0)
        np.fill_diagonal(dists_sq, np.inf)
        valid = np.all(dists_sq >= max_norm_sq, axis=1)
        if not np.all(valid):
            final_vectors = final_vectors[valid]

    if len(final_vectors) == 0:
        return vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
