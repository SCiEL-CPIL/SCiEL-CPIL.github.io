import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # ---- Candidate vectors: all norm^2 = 4 vectors in R^11 ----
    # W4: weight 4, entries in {-1,0,1}
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    # W1: weight 1, entries in {-2,0,2}
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

    # Conflict matrix: two vectors conflict if dot > 2 (dist^2 = 8 - 2*dot < 4)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    # Initialize best solution as empty
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree(rng, in_set_init=None):
        in_set = in_set_init.copy() if in_set_init is not None else np.zeros(N4, dtype=bool)
        alive = ~in_set
        degrees = neighbor_counts.copy()
        # degrees of alive nodes = number of alive neighbors
        alive_idx = np.where(alive)[0]
        if len(alive_idx) > 0:
            degrees[alive_idx] = conflicts[alive_idx][:, alive_idx].sum(axis=1)
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

    def greedy_max_degree(rng, in_set_init=None):
        in_set = in_set_init.copy() if in_set_init is not None else np.zeros(N4, dtype=bool)
        alive = ~in_set
        alive_idx = np.where(alive)[0]
        degrees = np.zeros(N4, dtype=np.int32)
        if len(alive_idx) > 0:
            degrees[alive_idx] = conflicts[alive_idx][:, alive_idx].sum(axis=1)
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

    def sa_local_search(in_set_init, rng, max_time=10.0, min_T=0.03, cooling=0.9997):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        T = 4.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        not_in_set = list(np.where(~in_set)[0])
        pos_in_not_in_set = {}
        for idx, v in enumerate(not_in_set):
            pos_in_not_in_set[v] = idx

        def remove_from_not_in_set(v):
            if v in pos_in_not_in_set:
                idx = pos_in_not_in_set[v]
                last = not_in_set[-1]
                not_in_set[idx] = last
                pos_in_not_in_set[last] = idx
                not_in_set.pop()
                del pos_in_not_in_set[v]

        def add_to_not_in_set(v):
            if v not in pos_in_not_in_set:
                pos_in_not_in_set[v] = len(not_in_set)
                not_in_set.append(v)

        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break
            if len(not_in_set) > 400:
                sample_indices = rng.choice(len(not_in_set), size=400, replace=False)
                sample = np.array([not_in_set[i] for i in sample_indices])
            else:
                sample = np.array(not_in_set)

            counts = count_conflicts[sample]
            min_count = int(np.min(counts))

            if min_count == 0:
                zero_mask = counts == 0
                zero_cands = sample[zero_mask]
                v = int(zero_cands[rng.integers(len(zero_cands))])
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            one_mask = counts == 1
            if np.any(one_mask) and rng.random() < 0.6:
                one_cands = sample[one_mask]
                v = int(one_cands[rng.integers(len(one_cands))])
                k = 1
                delta = 0
                accept = rng.random() < min(1.0, 0.4 + T * 0.3)
            else:
                min_mask = counts == min_count
                min_cands = sample[min_mask]
                v = int(min_cands[rng.integers(len(min_cands))])
                k = int(count_conflicts[v])
                delta = 1 - k
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    if delta < -25 * T:
                        accept = False
                    else:
                        accept = rng.random() < np.exp(delta / T)

            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                        add_to_not_in_set(int(u))
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 1000:
                T = max(T, 3.0)
                stuck_counter = 0
            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_polish(in_set_init, rng, max_time=6.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                for v in zero_cands:
                    in_set[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            in_set_indices = np.where(in_set)[0].copy()
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) > 0:
                    counts = count_conflicts[not_in_set]
                    addable = not_in_set[counts == 0]
                    if len(addable) > 1:
                        for v in addable:
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                    elif len(addable) == 1:
                        v = addable[0]
                        if v != i:
                            in_set[v] = True
                            count_conflicts[neighbors[v]] += 1
                            improved = True
                        else:
                            in_set[i] = True
                            current_size += 1
                            count_conflicts[neighbors[i]] += 1
                    else:
                        in_set[i] = True
                        current_size += 1
                        count_conflicts[neighbors[i]] += 1
                else:
                    in_set[i] = True
                    current_size += 1
                    count_conflicts[neighbors[i]] += 1

            if time.time() < polish_deadline and current_size >= 2:
                in_set_indices = np.where(in_set)[0].copy()
                n_in = len(in_set_indices)
                max_pairs = min(n_in * (n_in - 1) // 2, 800)
                pair_count = 0
                for pi in range(n_in):
                    if time.time() >= polish_deadline or pair_count >= max_pairs:
                        break
                    i = in_set_indices[pi]
                    for pj in range(pi + 1, n_in):
                        if time.time() >= polish_deadline or pair_count >= max_pairs:
                            break
                        pair_count += 1
                        j = in_set_indices[pj]
                        in_set[i] = False
                        in_set[j] = False
                        current_size -= 2
                        count_conflicts[neighbors[i]] -= 1
                        count_conflicts[neighbors[j]] -= 1
                        not_in_set = np.where(~in_set)[0]
                        if len(not_in_set) > 0:
                            counts = count_conflicts[not_in_set]
                            addable = not_in_set[counts == 0]
                            if len(addable) > 2:
                                for v in addable:
                                    in_set[v] = True
                                    current_size += 1
                                    count_conflicts[neighbors[v]] += 1
                                improved = True
                                if current_size > local_best_size:
                                    local_best_size = current_size
                                    local_best_in_set = in_set.copy()
                            else:
                                in_set[i] = True
                                in_set[j] = True
                                current_size += 2
                                count_conflicts[neighbors[i]] += 1
                                count_conflicts[neighbors[j]] += 1
                        else:
                            in_set[i] = True
                            in_set[j] = True
                            current_size += 2
                            count_conflicts[neighbors[i]] += 1
                            count_conflicts[neighbors[j]] += 1

        return local_best_in_set, local_best_size

    # ---- Main search loop ----
    n_restarts = 0
    max_restarts = 250
    base_seed = 42

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        rng = np.random.default_rng(base_seed + n_restarts * 7919)

        # Seed a fraction of restarts from the algebraic kissing set for diversity
        if n_restarts % 5 == 0:
            in_set = best_in_set.copy()
        elif n_restarts % 4 == 0:
            in_set = greedy_max_degree(rng)
        elif n_restarts % 4 == 1:
            in_set = greedy_min_degree(rng)
        else:
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                chosen = alive_indices[rng.integers(len(alive_indices))]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False

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
        sa_result, sa_size = sa_local_search(in_set, rng, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size >= current_size:
            polish_time = min(remaining_time * 0.3, 6.0)
            if polish_time > 1:
                polish_result, polish_size = exact_polish(sa_result, rng, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final polish on best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        rng_final = np.random.default_rng(12345)
        final_polish_time = min(remaining_time * 0.8, 8.0)
        if final_polish_time > 0.5:
            final_result, final_size = exact_polish(best_in_set, rng_final, max_time=final_polish_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_result
            remaining_time2 = deadline - 1 - time.time()
            if remaining_time2 > 1:
                second_time = min(remaining_time2 * 0.6, 4.0)
                if second_time > 0.5:
                    second_result, second_size = exact_polish(best_in_set, rng_final, max_time=second_time)
                    if second_size > best_size:
                        best_size = second_size
                        best_in_set = second_result

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors

# EVOLVE-BLOCK-END
