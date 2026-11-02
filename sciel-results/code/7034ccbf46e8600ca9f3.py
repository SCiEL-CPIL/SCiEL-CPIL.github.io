import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 118

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

    # Precompute neighbor lists as arrays for faster SA updates
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

    def greedy_max_degree():
        """Greedy independent set with max-degree heuristic (diversity)."""
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

    def local_search(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Hybrid Greedy + Simulated Annealing Local Search with cached not-in-set."""
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

        # Track membership with a boolean mask for fast updates
        in_mask = in_set.copy()

        # Cache not-in-set indices as a list for O(1) removal
        not_in_set_list = [int(i) for i in np.where(~in_set)[0]]
        # Map from node index to its position in not_in_set_list for O(1) removal
        pos_in_nis = np.full(N4, -1, dtype=np.int32)
        for idx, node in enumerate(not_in_set_list):
            pos_in_nis[node] = idx

        def remove_from_nis(v):
            """Remove node v from not_in_set_list in O(1)."""
            p = pos_in_nis[v]
            if p < 0:
                return
            last = not_in_set_list[-1]
            not_in_set_list[p] = last
            pos_in_nis[last] = p
            not_in_set_list.pop()
            pos_in_nis[v] = -1

        def add_to_nis(v):
            """Add node v to not_in_set_list in O(1)."""
            if pos_in_nis[v] >= 0:
                return
            pos_in_nis[v] = len(not_in_set_list)
            not_in_set_list.append(v)

        # Precompute neighbor arrays as Python lists for faster access
        neighbors_list = [list(n) for n in neighbors]

        while time.time() < sa_deadline:
            if len(not_in_set_list) == 0:
                break

            # Build array for vectorized min computation
            nis_arr = np.array(not_in_set_list, dtype=np.int32)
            counts = count_conflicts[nis_arr]
            min_count = int(np.min(counts))

            if min_count == 0:
                # Add a 0-conflict node
                zero_cands = nis_arr[counts == 0]
                v = int(zero_cands[rng.integers(len(zero_cands))])

                in_set[v] = True
                in_mask[v] = True
                current_size += 1
                for u in neighbors_list[v]:
                    count_conflicts[u] += 1
                remove_from_nis(v)

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # SA step: pick node with minimum conflict count
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
                # Remove neighbors of v that are in set
                nbrs_v = neighbors_list[v]
                to_remove = [u for u in nbrs_v if in_mask[u]]

                if to_remove:
                    for u in to_remove:
                        in_set[u] = False
                        in_mask[u] = False
                        for w in neighbors_list[u]:
                            count_conflicts[w] -= 1
                        add_to_nis(u)

                in_set[v] = True
                in_mask[v] = True
                current_size += delta
                for u in nbrs_v:
                    count_conflicts[u] += 1
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

    def exact_polish(in_set_init, max_time=5.0):
        """Exact local search: try all single additions and swaps with O(1) updates."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_mask = in_set.copy()
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        # Track not-in-set as a list with position map for O(1) ops
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

        # Precompute neighbor arrays as Python lists for faster access
        neighbors_list = [list(n) for n in neighbors]

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Try adding all nodes with 0 conflicts
            if len(not_in_set_list) > 0:
                nis_arr = np.array(not_in_set_list, dtype=np.int32)
                counts = count_conflicts[nis_arr]
                zero_cands = nis_arr[counts == 0]
                for v in zero_cands:
                    v = int(v)
                    in_set[v] = True
                    in_mask[v] = True
                    current_size += 1
                    for u in neighbors_list[v]:
                        count_conflicts[u] += 1
                    remove_from_nis(v)
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            # Try single swaps: for each node in set, temporarily remove and see if we can add more
            in_set_indices = np.where(in_set)[0].copy()
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break

                # Temporarily remove i
                in_set[i] = False
                in_mask[i] = False
                current_size -= 1
                for u in neighbors_list[i]:
                    count_conflicts[u] -= 1
                add_to_nis(int(i))

                # Find all nodes that can now be added (0 conflicts)
                if len(not_in_set_list) > 0:
                    nis_arr = np.array(not_in_set_list, dtype=np.int32)
                    counts = count_conflicts[nis_arr]
                    addable = nis_arr[counts == 0]
                    if len(addable) > 1:
                        # Add all addable nodes
                        added = []
                        for v in addable:
                            v = int(v)
                            in_set[v] = True
                            in_mask[v] = True
                            current_size += 1
                            for u in neighbors_list[v]:
                                count_conflicts[u] += 1
                            remove_from_nis(v)
                            added.append(v)
                            improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        else:
                            # Revert
                            for v in added:
                                in_set[v] = False
                                in_mask[v] = False
                                current_size -= 1
                                for u in neighbors_list[v]:
                                    count_conflicts[u] -= 1
                                add_to_nis(v)
                            in_set[i] = True
                            in_mask[i] = True
                            current_size += 1
                            for u in neighbors_list[i]:
                                count_conflicts[u] += 1
                            remove_from_nis(int(i))
                    elif len(addable) == 1:
                        v = int(addable[0])
                        if v != i:
                            in_set[v] = True
                            in_mask[v] = True
                            for u in neighbors_list[v]:
                                count_conflicts[u] += 1
                            remove_from_nis(v)
                            improved = True
                        else:
                            in_set[i] = True
                            in_mask[i] = True
                            current_size += 1
                            for u in neighbors_list[i]:
                                count_conflicts[u] += 1
                            remove_from_nis(int(i))
                    else:
                        in_set[i] = True
                        in_mask[i] = True
                        current_size += 1
                        for u in neighbors_list[i]:
                            count_conflicts[u] += 1
                        remove_from_nis(int(i))
                else:
                    in_set[i] = True
                    in_mask[i] = True
                    current_size += 1
                    for u in neighbors_list[i]:
                        count_conflicts[u] += 1
                    remove_from_nis(int(i))

        return local_best_in_set, local_best_size

    def two_for_one_polish(in_set_init, max_time=5.0):
        """Try 2-for-1 swaps: remove one node, add two if possible."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_mask = in_set.copy()
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

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

        # Precompute neighbor arrays as Python lists for faster access
        neighbors_list = [list(n) for n in neighbors]

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            in_set_indices = np.where(in_set)[0].copy()
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break

                # Temporarily remove i
                in_set[i] = False
                in_mask[i] = False
                current_size -= 1
                for u in neighbors_list[i]:
                    count_conflicts[u] -= 1
                add_to_nis(int(i))

                # Find all addable nodes
                if len(not_in_set_list) > 0:
                    nis_arr = np.array(not_in_set_list, dtype=np.int32)
                    counts = count_conflicts[nis_arr]
                    addable = nis_arr[counts == 0]

                    if len(addable) >= 2:
                        # Try adding first two addable nodes
                        v1 = int(addable[0])
                        v2 = int(addable[1])
                        in_set[v1] = True
                        in_mask[v1] = True
                        current_size += 1
                        for u in neighbors_list[v1]:
                            count_conflicts[u] += 1
                        remove_from_nis(v1)
                        
                        # Recheck for v2
                        if pos_in_nis[v2] >= 0 and count_conflicts[v2] == 0:
                            in_set[v2] = True
                            in_mask[v2] = True
                            current_size += 1
                            for u in neighbors_list[v2]:
                                count_conflicts[u] += 1
                            remove_from_nis(v2)
                            improved = True
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                        else:
                            # Revert v1
                            in_set[v1] = False
                            in_mask[v1] = False
                            current_size -= 1
                            for u in neighbors_list[v1]:
                                count_conflicts[u] -= 1
                            add_to_nis(v1)
                    elif len(addable) == 1:
                        v = int(addable[0])
                        if v != i:
                            in_set[v] = True
                            in_mask[v] = True
                            for u in neighbors_list[v]:
                                count_conflicts[u] += 1
                            remove_from_nis(v)
                            improved = True
                        else:
                            in_set[i] = True
                            in_mask[i] = True
                            current_size += 1
                            for u in neighbors_list[i]:
                                count_conflicts[u] += 1
                            remove_from_nis(int(i))
                    else:
                        in_set[i] = True
                        in_mask[i] = True
                        current_size += 1
                        for u in neighbors_list[i]:
                            count_conflicts[u] += 1
                        remove_from_nis(int(i))
                else:
                    in_set[i] = True
                    in_mask[i] = True
                    current_size += 1
                    for u in neighbors_list[i]:
                        count_conflicts[u] += 1
                    remove_from_nis(int(i))

        return local_best_in_set, local_best_size

    def crossover_resolve(parent_a, parent_b, max_time=5.0):
        """Merge two independent sets by taking union and resolving conflicts via local search."""
        # Start with the union of both sets
        union_set = parent_a | parent_b
        current_size = union_set.sum()
        
        # Build conflict counts
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_mask = union_set.copy()
        for i in np.where(union_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        # Not-in-set tracking
        not_in_set_list = [int(i) for i in np.where(~union_set)[0]]
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
        
        # Phase 1: Remove conflicting nodes greedily (keep lower-degree or random)
        resolve_deadline = time.time() + max_time * 0.4
        while time.time() < resolve_deadline:
            # Find nodes with conflicts
            conflicted = np.where(in_mask & (count_conflicts > 0))[0]
            if len(conflicted) == 0:
                break
            
            # Remove the conflicted node with highest conflict count
            max_conf = np.max(count_conflicts[conflicted])
            cands = conflicted[count_conflicts[conflicted] == max_conf]
            v = int(cands[rng.integers(len(cands))])
            
            in_set_v = in_mask.copy()
            in_set_v[v] = False
            in_mask[v] = False
            current_size -= 1
            count_conflicts[neighbors[v]] -= 1
            add_to_nis(v)
        
        # Phase 2: Local search to add back nodes
        local_best_size = current_size
        local_best_in_set = in_mask.copy()
        T = 2.0
        sa_deadline = time.time() + max_time * 0.6
        
        while time.time() < sa_deadline:
            if len(not_in_set_list) == 0:
                break
            
            nis_arr = np.array(not_in_set_list, dtype=np.int32)
            counts = count_conflicts[nis_arr]
            min_count = np.min(counts)
            
            if min_count == 0:
                zero_cands = nis_arr[counts == 0]
                v = int(zero_cands[rng.integers(len(zero_cands))])
                
                in_mask[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                remove_from_nis(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_mask.copy()
                continue
            
            # SA step
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
                    in_mask[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                        add_to_nis(int(u))
                
                in_mask[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                remove_from_nis(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_mask.copy()
            
            T *= 0.9995
            if T < 0.05:
                T = 0.05
        
        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 80
    population = []  # Store diverse solutions for crossover

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        # Alternate between greedy variants for diversity
        if n_restarts % 3 == 0:
            in_set = greedy_max_degree()
        else:
            in_set = greedy_min_degree()
        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.35, 10.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # Add to population for crossover (keep diverse solutions)
        if sa_size >= best_size * 0.95:
            population.append(sa_result.copy())
            if len(population) > 10:
                population.pop(0)

        # 3. Crossover: merge two solutions from population
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 3 and len(population) >= 2:
            # Pick two different solutions from population
            idx1 = rng.integers(len(population))
            idx2 = rng.integers(len(population))
            while idx2 == idx1 and len(population) > 1:
                idx2 = rng.integers(len(population))
            
            parent_a = population[int(idx1)]
            parent_b = population[int(idx2)]
            
            cross_time = min(remaining_time * 0.25, 6.0)
            if cross_time > 1:
                cross_result, cross_size = crossover_resolve(parent_a, parent_b, max_time=cross_time)
                if cross_size > best_size:
                    best_size = cross_size
                    best_in_set = cross_result.copy()
                    population.append(cross_result.copy())
                    if len(population) > 10:
                        population.pop(0)

        # 4. Polish pass on the best result
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polish_time = min(remaining_time * 0.3, 7.0)
            if polish_time > 1:
                polish_result, polish_size = local_search(
                    best_in_set, max_time=polish_time, min_T=0.02, cooling=0.9998
                )
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

        # 5. Exact polish periodically to squeeze out more
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 2 and n_restarts % 5 == 0 and best_size > 0:
            ep_time = min(remaining_time * 0.25, 3.5)
            if ep_time > 0.5:
                ep_result, ep_size = exact_polish(best_in_set, max_time=ep_time)
                if ep_size > best_size:
                    best_size = ep_size
                    best_in_set = ep_result.copy()

        # 6. 2-for-1 polish periodically
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 2 and n_restarts % 7 == 0 and best_size > 0:
            t21_time = min(remaining_time * 0.2, 2.5)
            if t21_time > 0.5:
                t21_result, t21_size = two_for_one_polish(best_in_set, max_time=t21_time)
                if t21_size > best_size:
                    best_size = t21_size
                    best_in_set = t21_result.copy()

    # Final exact polish on best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        final_polish_time = min(remaining_time * 0.85, 6.0)
        if final_polish_time > 0.5:
            final_result, final_size = exact_polish(best_in_set, max_time=final_polish_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_result

    # Final 2-for-1 polish
    remaining_time = deadline - 1 - time.time()
    if remaining_time > 1 and best_size > 0:
        final_t21_time = min(remaining_time * 0.9, 3.0)
        if final_t21_time > 0.5:
            final_t21_result, final_t21_size = two_for_one_polish(best_in_set, max_time=final_t21_time)
            if final_t21_size > best_size:
                best_size = final_t21_size
                best_in_set = final_t21_result

    # Final aggressive polish: run exact_polish again if time permits
    remaining_time = deadline - 0.5 - time.time()
    if remaining_time > 0.5 and best_size > 0:
        final_polish_time2 = min(remaining_time * 0.9, 2.0)
        if final_polish_time2 > 0.3:
            final_result2, final_size2 = exact_polish(best_in_set, max_time=final_polish_time2)
            if final_size2 > best_size:
                best_size = final_size2
                best_in_set = final_result2

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
