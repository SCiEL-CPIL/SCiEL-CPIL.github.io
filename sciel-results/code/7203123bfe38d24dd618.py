"""Improved kissing number d=11: vectorized greedy + enhanced SA + polish."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill

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

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Conflict matrix: conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists and counts
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_min_degree(seed=42):
        """Vectorized greedy independent set with min-degree heuristic."""
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]

            if len(cands) > 1:
                # Tie-break: prefer vertex whose neighbors have lower average degree
                scores = np.array([
                    np.mean(degrees[neighbors[c]]) if len(neighbors[c]) > 0 else 0
                    for c in cands
                ])
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def greedy_random_priority(seed=42):
        """Greedy with random priority ordering."""
        rng = np.random.default_rng(seed)
        priorities = rng.random(N)
        order = np.argsort(priorities)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)

        for v in order:
            if not alive[v]:
                continue
            if not np.any(conflicts[v] & in_set):
                in_set[v] = True
                alive[neighbors[v]] = False
                alive[v] = False

        return in_set

    def greedy_max_degree(seed=42):
        """Greedy with max-degree heuristic (remove high-degree vertices first)."""
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]

            if len(cands) > 1:
                chosen = cands[rng.integers(len(cands))]
            else:
                chosen = cands[0]

            # Don't add this vertex; remove it and its neighbors from consideration
            # Actually, for max-degree heuristic in independent set, we skip high-degree
            # Let's do: add the vertex with max degree (it constrains most, so adding it early is bad)
            # Better: use min-degree for adding. Let me just use a different random strategy.
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def polish(in_set, max_time=2.0):
        """Deterministic polish: add any zero-conflict vertex, try 1-for-1 swaps."""
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        polish_deadline = time.time() + max_time

        # Pass 1: add any vertex with zero conflicts
        while time.time() < polish_deadline:
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            zero_conflict = not_in[np.all(~conflicts[not_in][:, in_set], axis=1)]
            if len(zero_conflict) == 0:
                break
            # Add all zero-conflict vertices
            for v in zero_conflict:
                if not in_set[v]:
                    in_set[v] = True
                    current_size += 1

        # Pass 2: try 1-for-1 swaps
        while time.time() < polish_deadline:
            improved = False
            in_indices = np.where(in_set)[0]
            not_indices = np.where(~in_set)[0]
            if len(in_indices) == 0 or len(not_indices) == 0:
                break

            for v_out in in_indices:
                if time.time() > polish_deadline:
                    break
                # Find v_in that conflicts with v_out but not with others in set
                # v_in must not conflict with (in_set \ {v_out})
                other_in = in_set.copy()
                other_in[v_out] = False
                for v_in in not_indices:
                    if in_set[v_in]:
                        continue
                    if np.any(conflicts[v_in] & other_in):
                        continue
                    # v_in doesn't conflict with other_in, so we can swap
                    in_set[v_out] = False
                    in_set[v_in] = True
                    current_size += 0  # size unchanged
                    improved = True
                    break
                if improved:
                    break
            if not improved:
                break

        return in_set, current_size

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Enhanced SA with add/remove/swap/2-for-2 moves."""
        rng = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            move_type = rng.random()

            if move_type < 0.45 and len(not_in_set) > 0:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]

                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
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

            elif move_type < 0.6 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.80 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[(count_conflicts[not_in_set] <= 1)]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
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

            elif move_type < 0.92 and len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                # 2-for-2 swap
                v1 = in_set_indices[rng.integers(len(in_set_indices))]
                remaining = np.setdiff1d(in_set_indices, [v1])
                if len(remaining) == 0:
                    stuck_counter += 1
                    continue
                v2 = remaining[rng.integers(len(remaining))]

                # Find two not-in-set vertices that together only conflict with {v1, v2}
                # Simplified: pick v_add1 with count_conflicts <= 2, then v_add2
                v_add_cands = not_in_set[(count_conflicts[not_in_set] <= 2)]
                if len(v_add_cands) < 2:
                    v_add_cands = not_in_set
                if len(v_add_cands) < 2:
                    stuck_counter += 1
                    continue

                idx1 = rng.integers(len(v_add_cands))
                v_add1 = v_add_cands[idx1]
                remaining_cands = np.setdiff1d(v_add_cands, [v_add1])
                if len(remaining_cands) == 0:
                    stuck_counter += 1
                    continue
                idx2 = rng.integers(len(remaining_cands))
                v_add2 = remaining_cands[idx2]

                # Check conflicts
                conf1 = np.where(conflicts[v_add1] & in_set)[0]
                conf2 = np.where(conflicts[v_add2] & in_set)[0]
                all_conf = np.unique(np.concatenate([conf1, conf2])) if len(conf1) + len(conf2) > 0 else np.array([], dtype=np.int64)
                other_conf = all_conf[~np.isin(all_conf, [v1, v2])]

                if len(other_conf) == 0:
                    in_set[v1] = False
                    in_set[v2] = False
                    in_set[v_add1] = True
                    in_set[v_add2] = True
                    count_conflicts[neighbors[v1]] -= 1
                    count_conflicts[neighbors[v2]] -= 1
                    count_conflicts[neighbors[v_add1]] += 1
                    count_conflicts[neighbors[v_add2]] += 1
                    # Size unchanged
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            else:
                # 2-for-1 move
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    remaining = np.setdiff1d(in_set_indices, [v1])
                    if len(remaining) == 0:
                        stuck_counter += 1
                        continue
                    v2 = remaining[rng.integers(len(remaining))]

                    v_add_cands = not_in_set[(count_conflicts[not_in_set] <= 2)]
                    if len(v_add_cands) == 0:
                        v_add_cands = not_in_set
                    v_add = v_add_cands[rng.integers(len(v_add_cands))]

                    conf = np.where(conflicts[v_add] & in_set)[0]
                    other_conf = conf[~np.isin(conf, [v1, v2])]

                    if len(other_conf) == 0:
                        in_set[v1] = False
                        in_set[v2] = False
                        in_set[v_add] = True
                        current_size -= 1
                        count_conflicts[neighbors[v1]] -= 1
                        count_conflicts[neighbors[v2]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            T = max(min_T, T * cooling)

            if stuck_counter > 120:
                n_perturb = rng.integers(2, 5)
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    for _ in range(min(n_perturb, len(in_set_indices))):
                        idx = rng.integers(len(in_set_indices))
                        u = in_set_indices[idx]
                        in_set[u] = False
                        current_size -= 1
                        count_conflicts[neighbors[u]] -= 1
                        in_set_indices = np.where(in_set)[0]
                        if len(in_set_indices) == 0:
                            break
                stuck_counter = 0
                T = max(T, 2.0)

        return local_best_in_set, local_best_size

    # Multiple restarts
    n_restarts = 0
    max_restarts = 1000
    seed_base = 42
    strategies = ['min_degree', 'random', 'min_degree']

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        seed = seed_base + n_restarts * 7
        strategy = strategies[n_restarts % len(strategies)]

        if strategy == 'min_degree':
            in_set = greedy_min_degree(seed=seed)
        else:
            in_set = greedy_random_priority(seed=seed)

        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 12 - time.time()
        if remaining_time < 3:
            break

        # SA
        ls_time = min(remaining_time * 0.35, 6.0)
        if ls_time < 1.5:
            ls_time = 1.5

        ls_result, ls_size = sa_local_search(in_set, max_time=ls_time, seed=seed + 1000)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

        # Polish
        if time.time() < deadline - 12:
            polished, pol_size = polish(best_in_set, max_time=min(1.0, deadline - 12 - time.time()))
            if pol_size > best_size:
                best_size = pol_size
                best_in_set = polished

        if time.time() > deadline - 12:
            break

    # Final polish on best
    if time.time() < deadline - 5:
        polished, pol_size = polish(best_in_set, max_time=min(3.0, deadline - 5 - time.time()))
        if pol_size > best_size:
            best_size = pol_size
            best_in_set = polished

    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]

    return result_vectors
