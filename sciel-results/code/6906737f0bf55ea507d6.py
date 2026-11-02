<<<<<<< SEARCH
    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap moves."""
        rng = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        # count_conflicts[v] = number of neighbors of v that are in in_set
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

            if move_type < 0.5 and len(not_in_set) > 0:
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

            elif move_type < 0.7 and len(in_set_indices) > 0:
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

            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[
                    (count_conflicts[not_in_set] <= 1)
                ]
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

            else:
                # 2-for-1 move
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    remaining = np.setdiff1d(in_set_indices, [v1])
                    if len(remaining) == 0:
                        stuck_counter += 1
                        continue
                    v2 = remaining[rng.integers(len(remaining))]

                    v_add_candidates = not_in_set[
                        (count_conflicts[not_in_set] <= 2)
                    ]
                    if len(v_add_candidates) == 0:
                        v_add_candidates = not_in_set
                    v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]

                    if len(other_conflicts) == 0:
                        to_remove = np.array([v1, v2])
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size -= 1
                        count_conflicts[neighbors[v_add]] += 1
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            # Cooling
            T = max(min_T, T * cooling)

            # Perturbation when stuck
            if stuck_counter > 150:
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

        # Final polish: greedily add any vertex with zero conflicts
        while True:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            zero_conf = not_in_set[count_conflicts[not_in_set] == 0]
            if len(zero_conf) == 0:
                break
            v = zero_conf[0]
            in_set[v] = True
            current_size += 1
            count_conflicts[neighbors[v]] += 1
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()

        return local_best_in_set, local_best_size
=======
    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap moves."""
        rng = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 6.0
        cooling = 0.999
        min_T = 0.03

        # count_conflicts[v] = number of neighbors of v that are in in_set
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

            if move_type < 0.55 and len(not_in_set) > 0:
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
                    accept = rng.random() < 0.4
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

            elif move_type < 0.72 and len(in_set_indices) > 0:
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

            elif move_type < 0.88 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[
                    (count_conflicts[not_in_set] <= 1)
                ]
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

            else:
                # 2-for-1 move
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    remaining = np.setdiff1d(in_set_indices, [v1])
                    if len(remaining) == 0:
                        stuck_counter += 1
                        continue
                    v2 = remaining[rng.integers(len(remaining))]

                    v_add_candidates = not_in_set[
                        (count_conflicts[not_in_set] <= 2)
                    ]
                    if len(v_add_candidates) == 0:
                        v_add_candidates = not_in_set
                    v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]

                    if len(other_conflicts) == 0:
                        to_remove = np.array([v1, v2])
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size -= 1
                        count_conflicts[neighbors[v_add]] += 1
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            # Cooling
            T = max(min_T, T * cooling)

            # Perturbation when stuck
            if stuck_counter > 120:
                n_perturb = rng.integers(2, 6)
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
                T = max(T, 3.0)

        # Final polish: greedily add any vertex with zero conflicts
        while True:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            zero_conf = not_in_set[count_conflicts[not_in_set] == 0]
            if len(zero_conf) == 0:
                break
            v = zero_conf[0]
            in_set[v] = True
            current_size += 1
            count_conflicts[neighbors[v]] += 1
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()

        return local_best_in_set, local_best_size
>>>>>>> REPLACE

