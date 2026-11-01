import numpy as np
import time
import itertools

D = 11
KISS_UPPER = 165          # kissing number upper bound for dimension 11
SUPPORT_TARGET = 66       # 66 supports * 16 sign variants = 1056 vectors (below upper bound)
V1_COUNT = 2 * D          # 22 weight-1 vectors (entries +/-2, norm^2 = 4)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before the 150 s kill

    # ---- 1. Build the candidate vector pool (all norm^2 = 4) ----
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)

    # weight-4 vectors: 16 sign variants per support
    v4 = np.zeros((n_supports * 16, D), dtype=np.int64)
    for i, sup in enumerate(supports):
        for k, signs in enumerate(itertools.product([1, -1], repeat=4)):
            row = i * 16 + k
            for pos, s in zip(sup, signs):
                v4[row, pos] = s

    # weight-1 vectors: +/-2 on a single coordinate
    v1 = np.zeros((V1_COUNT, D), dtype=np.int64)
    for pos in range(D):
        v1[2 * pos, pos] = 2
        v1[2 * pos + 1, pos] = -2

    V4 = n_supports * 16
    V = V4 + V1_COUNT

    # ---- 2. Conflict matrices (conflict iff |dot| > 2, i.e. distance < norm) ----
    dots4 = v4 @ v4.T
    conf4 = np.abs(dots4) > 2
    np.fill_diagonal(conf4, False)

    dots1 = v1 @ v1.T
    conf1 = np.abs(dots1) > 2
    np.fill_diagonal(conf1, False)

    dots14 = v1 @ v4.T
    conf14 = np.abs(dots14) > 2

    conf = np.zeros((V, V), dtype=bool)
    conf[:V4, :V4] = conf4
    conf[V4:, V4:] = conf1
    conf[:V4, V4:] = conf14.T
    conf[V4:, :V4] = conf14

    # support-level conflict (two supports conflict iff >= 3 shared coordinates)
    confS = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            if len(si & set(supports[j])) >= 3:
                confS[i, j] = True
                confS[j, i] = True

    neighbors = [np.where(conf[i])[0] for i in range(V)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    nbrS = [np.where(confS[i])[0] for i in range(n_supports)]
    nbrS_counts = np.array([len(n) for n in nbrS], dtype=np.int32)

    rng = np.random.default_rng(42)

    # ---- 3. Vector-level min-conflict simulated annealing (refinement) ----
    def vector_sa(in_set_init, max_time, seed):
        r = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        count_conflicts = np.zeros(V, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        stuck_counter = 0
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = count_conflicts[not_in_set]
            min_count = int(np.min(counts))
            min_cands = not_in_set[counts == min_count]
            v = min_cands[r.integers(len(min_cands))]

            k = min_count
            delta = 1 - k
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = r.random() < 0.5
            else:
                accept = r.random() < np.exp(delta / T)

            if accept:
                to_remove = np.where(conf[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1

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

    # ---- 4. 2-for-1 swap move (escape local optima) ----
    def swap_move(in_set, r, max_time):
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()

        count_conflicts = np.zeros(V, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        T = 2.0
        cooling = 0.999
        min_T = 0.05
        swap_deadline = time.time() + max_time

        while time.time() < swap_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = count_conflicts[not_in_set]
            min_count = int(np.min(counts))
            min_cands = not_in_set[counts == min_count]
            v = min_cands[r.integers(len(min_cands))]
            k = min_count
            delta = 1 - k
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = r.random() < 0.5
            else:
                accept = r.random() < np.exp(delta / T)

            if accept:
                to_remove = np.where(conf[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
            T *= cooling
            if T < min_T:
                T = min_T
        return local_best_in_set, local_best_size

    # ---- 5. Support-level greedy min-degree (fast, robust initializer) ----
    def support_greedy():
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = nbrS_counts.copy()
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = int(np.min(degrees[alive_indices]))
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], nbrS[chosen])))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= confS[rem][:, to_remove].sum(axis=1)
        return in_set

    # ---- 6. Support-level SA (produces a strong seed) ----
    def support_sa(in_set_init, max_time, seed):
        r = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[nbrS[i]] += 1
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        stuck_counter = 0
        sa_deadline = time.time() + max_time
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = count_conflicts[not_in_set]
            min_count = int(np.min(counts))
            min_cands = not_in_set[counts == min_count]
            s = min_cands[r.integers(len(min_cands))]
            k = min_count
            delta = 1 - k
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = r.random() < 0.5
            else:
                accept = r.random() < np.exp(delta / T)
            if accept:
                to_remove = np.where(confS[s] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[nbrS[u]] -= 1
                in_set[s] = True
                current_size += delta
                count_conflicts[nbrS[s]] += 1
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

    def support_to_vector(support_in_set):
        vec = np.zeros(V, dtype=bool)
        for i in np.where(support_in_set)[0]:
            vec[i * 16:(i + 1) * 16] = True
        return vec

    # ---- 7. Main loop: support-level seed -> vector-level refinement ----
    best_size = 0
    best_in_set = np.zeros(V, dtype=bool)

    n_restarts = 0
    max_restarts = 1000
    seed_counter = 42

    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        seed_counter += 1

        # support-level seed
        sup = support_greedy()
        sup_size = int(sup.sum())
        if sup_size > SUPPORT_TARGET:
            # trim to the upper-bound target (guidance)
            idx = np.where(sup)[0]
            keep = idx[:SUPPORT_TARGET]
            sup = np.zeros(n_supports, dtype=bool)
            sup[keep] = True
            sup_size = int(sup.sum())

        vec = support_to_vector(sup)
        cur_size = int(vec.sum()) + V1_COUNT
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = vec.copy()

        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break

        # vector-level SA refinement
        sa_time = min(remaining * 0.6, 12.0)
        if sa_time < 1.0:
            sa_time = 1.0
        sa_res, sa_size = vector_sa(vec, sa_time, seed_counter)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_res.copy()

        # 2-for-1 swap polish on the improved state
        remaining = deadline - 5 - time.time()
        if remaining > 4:
            swap_time = min(remaining * 0.4, 8.0)
            if swap_time > 1.0:
                sw_res, sw_size = swap_move(sa_res, np.random.default_rng(seed_counter + 1), swap_time)
                if sw_size > best_size:
                    best_size = sw_size
                    best_in_set = sw_res.copy()

        # final high-precision polish pass on the global best
        remaining = deadline - 5 - time.time()
        if remaining > 6 and n_restarts % 5 == 0:
            polish_time = min(remaining * 0.5, 10.0)
            if polish_time > 3:
                po_res, po_size = vector_sa(best_in_set, polish_time, seed_counter + 2)
                if po_size > best_size:
                    best_size = po_size
                    best_in_set = po_res.copy()

    # ---- 8. Assemble final integer vectors ----
    final_vectors = np.vstack([v4[best_in_set[:V4]], v1])
    return final_vectors
