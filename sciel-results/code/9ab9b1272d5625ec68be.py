import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def _generate_e11_roots():
    """Generate the 792 roots of the E11 root lattice (norm^2 = 2).

    E11 = D11 union (D11 + (1/2,1/2,...,1/2)), where D11 is the even
    sub-lattice of Z^11 (sum of coordinates even). Roots are the vectors
    of norm^2 = 2.
    """
    roots = []
    # --- D11 roots: weight 2, entries in {-1,0,1} ---
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    # --- Coset roots: half-integer vectors (2x+1)/2 with sum(x)=0 ---
    for x in itertools.product((-1, 1), repeat=D):
        if sum(x) != 0:
            continue
        v = np.array([2 * xi + 1 for xi in x], dtype=np.int64) // 2
        roots.append(v)

    arr = np.array(roots, dtype=np.int64)
    # Deduplicate (keep first occurrence)
    _, uniq_idx = np.unique(arr, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    arr = arr[uniq_idx]
    # Verify all have norm^2 = 2
    norms = arr @ arr.T
    np.fill_diagonal(norms, 0)
    if not np.all(np.sum(arr * arr, axis=1) == 2):
        return None
    return arr


def construct():
    start_time = time.time()
    deadline = start_time + 115

    # ---- Primary: algebraic E11 root construction ----
    try:
        roots = _generate_e11_roots()
        if roots is not None and len(roots) > 0:
            # Verify kissing constraint: min pairwise distance^2 >= max norm^2
            norms2 = np.sum(roots * roots, axis=1)
            max_norm2 = int(np.max(norms2))
            dots = roots @ roots.T
            np.fill_diagonal(dots, 0)
            min_dist2 = int(2 * max_norm2 - np.max(dots))  # d^2 = n_i^2+n_j^2 - 2 dot, min over pairs
            # More careful: min pairwise squared distance
            min_d2 = int(np.min(2 * max_norm2 - dots))
            if min_d2 >= max_norm2:
                return roots
    except Exception:
        pass

    # ---- Fallback: heuristic independent-set search ----
    return _heuristic_search(start_time, deadline)


def _heuristic_search(start_time, deadline):
    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
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

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

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

    def local_search(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)

            if min_count == 0:
                zero_cands = not_in_set[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            move_type = rng.random()

            if move_type < 0.5:
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
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
                    to_remove = np.where(conflicts[v] & in_set)[0]
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

            elif move_type < 0.7:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    delta = -1
                    accept = rng.random() < np.exp(delta / T)
                    if accept:
                        in_set[v] = False
                        current_size += delta
                        count_conflicts[neighbors[v]] -= 1
                    else:
                        stuck_counter += 1

            else:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_remove]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            count_conflicts[neighbors[v_add]] += 1
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

    n_restarts = 0
    max_restarts = 100
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
        sa_time = min(remaining_time * 0.6, 12.0)
        if sa_time < 2:
            sa_time = 2.0
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 8.0)
            if polish_time > 1:
                polish_result, polish_size = local_search(
                    sa_result, max_time=polish_time, min_T=0.02, cooling=0.9998
                )
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors

# EVOLVE-BLOCK-END
