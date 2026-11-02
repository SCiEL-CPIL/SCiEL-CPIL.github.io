import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
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

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
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

    def local_search(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995,
                     use_swaps=True, swap_prob=0.15):
        """Hybrid Greedy + SA with 2-for-1 swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        # Precompute not_in_set list (updated incrementally)
        not_in_set_list = np.where(~in_set)[0]
        iter_count = 0

        while time.time() < sa_deadline:
            if len(not_in_set_list) == 0:
                break

            # Periodic rebuild to prevent drift
            iter_count += 1
            if iter_count % 5000 == 0:
                count_conflicts[:] = 0
                for i in np.where(in_set)[0]:
                    count_conflicts[neighbors[i]] += 1
                not_in_set_list = np.where(~in_set)[0]

            counts = count_conflicts[not_in_set_list]
            min_count = np.min(counts)

            # Try 2-for-1 swap move with some probability
            if use_swaps and min_count >= 1 and rng.random() < swap_prob and len(not_in_set_list) >= 2:
                min_cands = not_in_set_list[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                # Find u not in set, not neighbor of v, with count_conflicts[u] == 0
                zero_mask = count_conflicts[not_in_set_list] == 0
                u_cands_idx = np.where(zero_mask)[0]
                if len(u_cands_idx) > 0:
                    # Filter out neighbors of v
                    v_nbrs_set = set(neighbors[v].tolist())
                    valid_u = []
                    for idx in u_cands_idx:
                        u = not_in_set_list[idx]
                        if u not in v_nbrs_set:
                            valid_u.append(u)
                    if len(valid_u) > 0:
                        u = valid_u[rng.integers(len(valid_u))]
                        # Find in-set neighbors of v and u
                        to_remove_set = set()
                        for w in neighbors[v]:
                            if in_set[w]:
                                to_remove_set.add(w)
                        for w in neighbors[u]:
                            if in_set[w]:
                                to_remove_set.add(w)
                        to_remove = np.array(sorted(to_remove_set), dtype=np.int64)
                        delta = 2 - len(to_remove)
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
                            for w in to_remove:
                                in_set[w] = False
                                count_conflicts[neighbors[w]] -= 1
                            in_set[v] = True
                            in_set[u] = True
                            current_size += delta
                            count_conflicts[neighbors[v]] += 1
                            count_conflicts[neighbors[u]] += 1

                            # Update not_in_set_list
                            mask = np.ones(len(not_in_set_list), dtype=bool)
                            for node in [v, u]:
                                mask[not_in_set_list == node] = False
                            not_in_set_list = not_in_set_list[mask]

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
                        continue

            # Standard single-add SA step
            if min_count == 0:
                # Add a 0-conflict node
                zero_cands = not_in_set_list[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]

                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1

                # Remove v from not_in_set_list
                not_in_set_list = not_in_set_list[not_in_set_list != v]

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # SA step: pick node with minimum conflict count
            min_cands = not_in_set_list[counts == min_count]
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
                # Remove neighbors of v that are in set (use neighbor list, not full matrix)
                to_remove_list = [w for w in neighbors[v] if in_set[w]]
                to_remove = np.array(to_remove_list, dtype=np.int64) if to_remove_list else np.array([], dtype=np.int64)

                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    # Remove to_remove from not_in_set_list
                    mask = np.ones(len(not_in_set_list), dtype=bool)
                    for node in to_remove:
                        mask[not_in_set_list == node] = False
                    not_in_set_list = not_in_set_list[mask]

                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1

                # Remove v from not_in_set_list
                mask = np.ones(len(not_in_set_list), dtype=bool)
                mask[not_in_set_list == v] = False
                not_in_set_list = not_in_set_list[mask]

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

    def final_polish(in_set_init, max_time=15.0):
        """High-precision polish: very low temperature, many iterations."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 0.1
        cooling = 0.9999
        min_T = 0.01

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        not_in_set_list = np.where(~in_set)[0]
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time
        iter_count = 0

        while time.time() < polish_deadline:
            if len(not_in_set_list) == 0:
                break

            # Periodic rebuild to prevent drift
            iter_count += 1
            if iter_count % 5000 == 0:
                count_conflicts[:] = 0
                for i in np.where(in_set)[0]:
                    count_conflicts[neighbors[i]] += 1
                not_in_set_list = np.where(~in_set)[0]

            counts = count_conflicts[not_in_set_list]
            min_count = np.min(counts)

            if min_count == 0:
                zero_cands = not_in_set_list[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                mask = np.ones(len(not_in_set_list), dtype=bool)
                mask[not_in_set_list == v] = False
                not_in_set_list = not_in_set_list[mask]
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                continue

            min_cands = not_in_set_list[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            k = min_count
            delta = 1 - k

            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.3
            else:
                if delta < -10 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)

            if accept:
                to_remove_list = [w for w in neighbors[v] if in_set[w]]
                to_remove = np.array(to_remove_list, dtype=np.int64) if to_remove_list else np.array([], dtype=np.int64)
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    mask = np.ones(len(not_in_set_list), dtype=bool)
                    for node in to_remove:
                        mask[not_in_set_list == node] = False
                    not_in_set_list = not_in_set_list[mask]
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                mask = np.ones(len(not_in_set_list), dtype=bool)
                mask[not_in_set_list == v] = False
                not_in_set_list = not_in_set_list[mask]
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 1000
    seed_counter = 0

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        seed_counter += 1
        rng = np.random.default_rng(42 + seed_counter)

        # 1. Greedy Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.5, 10.0)
        if sa_time < 1.5:
            sa_time = 1.5

        sa_result, sa_size = local_search(in_set, max_time=sa_time, use_swaps=True, swap_prob=0.15)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Polish (Second pass on best with tighter parameters)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 6.0)
            if polish_time > 1:
                polish_result, polish_size = local_search(
                    sa_result, max_time=polish_time, min_T=0.02, cooling=0.9998,
                    use_swaps=True, swap_prob=0.1
                )
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # 4. Final high-precision polish pass
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 3 and best_size > 0:
        polish_time = min(remaining_time, 15.0)
        if polish_time > 2:
            final_result, final_size = final_polish(best_in_set, max_time=polish_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_result.copy()

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
