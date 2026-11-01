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

    def greedy_random():
        """Greedy independent set with random selection."""
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

        return in_set

    def local_search_enhanced(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Enhanced SA with O(1) not_in_set, 2-for-1 swaps, and perturbation."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        # Maintain not_in_set as a list with position tracking for O(1) removal
        not_in_set = list(np.where(~in_set)[0])
        pos_in_not_in_set = {}
        for idx, v in enumerate(not_in_set):
            pos_in_not_in_set[v] = idx

        def remove_from_not_in_set(v):
            """O(1) removal from not_in_set list."""
            if v in pos_in_not_in_set:
                idx = pos_in_not_in_set[v]
                last = not_in_set[-1]
                not_in_set[idx] = last
                pos_in_not_in_set[last] = idx
                not_in_set.pop()
                del pos_in_not_in_set[v]

        def add_to_not_in_set(v):
            """O(1) addition to not_in_set list."""
            if v not in pos_in_not_in_set:
                pos_in_not_in_set[v] = len(not_in_set)
                not_in_set.append(v)

        def try_add_node(v):
            """Try to add node v, removing conflicting nodes from in_set."""
            nonlocal current_size
            to_remove = np.where(conflicts[v] & in_set)[0]
            k = len(to_remove)
            delta = 1 - k

            if k > 0:
                in_set[to_remove] = False
                for u in to_remove:
                    count_conflicts[neighbors[u]] -= 1
                    add_to_not_in_set(int(u))

            in_set[v] = True
            current_size += delta
            count_conflicts[neighbors[v]] += 1
            remove_from_not_in_set(v)
            return delta

        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break

            # Sample a subset if not_in_set is large to speed up min finding
            if len(not_in_set) > 200:
                sample_size = 200
                sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                sample = np.array([not_in_set[i] for i in sample_indices])
            else:
                sample = np.array(not_in_set)

            counts = count_conflicts[sample]
            min_count = int(np.min(counts))

            if min_count == 0:
                # Add a 0-conflict node
                zero_mask = counts == 0
                if np.any(zero_mask):
                    zero_cands = sample[zero_mask]
                    v = int(zero_cands[rng.integers(len(zero_cands))])
                else:
                    v = int(not_in_set[rng.integers(len(not_in_set))])

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

            # SA step: pick node with weighted probability based on conflict count
            # Lower conflict count -> higher probability
            # Use softmax-like weighting
            weights = np.exp(-counts / max(T, 0.1))
            weights /= weights.sum()
            v = int(sample[rng.choice(len(sample), p=weights)])

            k = int(count_conflicts[v])
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

            # 2-for-1 swap: try removing 1 node and adding 2 nodes
            if stuck_counter > 500 and len(not_in_set) > 2 and len(np.where(in_set)[0]) > 2:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    # Pick a random node to remove
                    remove_v = int(in_set_indices[rng.integers(len(in_set_indices))])
                    
                    # Temporarily remove it
                    in_set[remove_v] = False
                    current_size -= 1
                    count_conflicts[neighbors[remove_v]] -= 1
                    add_to_not_in_set(remove_v)
                    
                    # Try to add two nodes that don't conflict with each other
                    # and don't conflict with remaining in_set
                    not_in_set_arr = np.array(not_in_set)
                    if len(not_in_set_arr) > 1:
                        counts = count_conflicts[not_in_set_arr]
                        # Find nodes with 0 conflicts
                        zero_cands = not_in_set_arr[counts == 0]
                        
                        if len(zero_cands) >= 2:
                            # Try all pairs from zero_cands (limited)
                            max_pairs = min(20, len(zero_cands))
                            pair_found = False
                            for _ in range(max_pairs):
                                i1 = rng.integers(len(zero_cands))
                                i2 = rng.integers(len(zero_cands))
                                if i1 == i2:
                                    continue
                                v1 = int(zero_cands[i1])
                                v2 = int(zero_cands[i2])
                                
                                # Check if v1 and v2 conflict
                                if not conflicts[v1, v2]:
                                    # Add both
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    current_size += 2
                                    count_conflicts[neighbors[v1]] += 1
                                    count_conflicts[neighbors[v2]] += 1
                                    remove_from_not_in_set(v1)
                                    remove_from_not_in_set(v2)
                                    pair_found = True
                                    break
                            
                            if not pair_found:
                                # Revert: add back remove_v
                                in_set[remove_v] = True
                                current_size += 1
                                count_conflicts[neighbors[remove_v]] += 1
                                remove_from_not_in_set(remove_v)
                        else:
                            # Revert
                            in_set[remove_v] = True
                            current_size += 1
                            count_conflicts[neighbors[remove_v]] += 1
                            remove_from_not_in_set(remove_v)

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0

            # Perturbation on long stall: randomly remove several nodes
            if stuck_counter > 2000:
                n_remove = min(4, int(np.where(in_set)[0].sum()))
                if n_remove > 0:
                    in_set_indices = np.where(in_set)[0]
                    to_perturb = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                    for u in to_perturb:
                        in_set[u] = False
                        current_size -= 1
                        count_conflicts[neighbors[u]] -= 1
                        add_to_not_in_set(int(u))
                    stuck_counter = 0
                    T = max(T, 2.0)

            if stuck_counter > 1000:
                T = max(T, 2.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_polish(in_set_init, max_time=5.0):
        """Exact local search: try all single additions and swaps."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Try adding all nodes with 0 conflicts
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

            # Try single swaps: for each node in set, temporarily remove and see if we can add more
            in_set_indices = np.where(in_set)[0].copy()
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break

                # Temporarily remove i
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1

                # Find all nodes that can now be added (0 conflicts)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) > 0:
                    counts = count_conflicts[not_in_set]
                    addable = not_in_set[counts == 0]
                    if len(addable) > 1:  # Need at least 2 to improve (1 adds back i's slot, net gain >= 1)
                        # Add all addable nodes
                        for v in addable:
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                            improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        else:
                            # Revert: remove all added nodes and restore i
                            for v in addable:
                                in_set[v] = False
                                current_size -= 1
                                count_conflicts[neighbors[v]] -= 1
                            in_set[i] = True
                            current_size += 1
                            count_conflicts[neighbors[i]] += 1
                    elif len(addable) == 1:
                        # Same size, but might help future iterations - keep the swap if it's different
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
                        # No improvement, restore i
                        in_set[i] = True
                        current_size += 1
                        count_conflicts[neighbors[i]] += 1
                else:
                    # No nodes to add, restore i
                    in_set[i] = True
                    current_size += 1
                    count_conflicts[neighbors[i]] += 1

        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 100

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        # Alternate between greedy variants for diversity
        if n_restarts % 4 == 0:
            in_set = greedy_max_degree()
        elif n_restarts % 4 == 1:
            in_set = greedy_min_degree()
        elif n_restarts % 4 == 2:
            in_set = greedy_random()
        else:
            in_set = greedy_min_degree()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.45, 14.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search_enhanced(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Polish pass on the SA result
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size >= current_size:
            polish_time = min(remaining_time * 0.35, 8.0)
            if polish_time > 1:
                polish_result, polish_size = local_search_enhanced(
                    sa_result, max_time=polish_time, min_T=0.02, cooling=0.9998
                )
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final exact polish on best solution
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

    return final_vectors

# EVOLVE-BLOCK-END
