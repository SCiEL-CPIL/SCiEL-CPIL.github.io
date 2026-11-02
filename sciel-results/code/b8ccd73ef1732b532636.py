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

    def constructive_init_e8e3():
        """Constructive initialization based on E8⊕E3 lattice structure.
        
        E8 in 8D: vectors with 4 nonzeros in {-1,0,1} with even number of -1s
        E3 in 3D: vectors with 2 nonzeros in {-1,0,1} (norm^2=2, scaled) or 1 nonzero in {-2,0,2}
        
        We construct a large independent set by:
        1. Taking E8-like vectors in first 8 coords (even parity of -1s)
        2. Taking complementary vectors in last 3 coords
        3. Ensuring no conflicts between groups
        """
        in_set = np.zeros(N4, dtype=bool)
        
        # Map from vector tuple to index
        vec_to_idx = {}
        for i, v in enumerate(v4_candidates):
            vec_to_idx[tuple(v)] = i
        
        # E8-like vectors: 4 nonzeros in first 8 coords, even number of -1s
        e8_vectors = []
        for positions in itertools.combinations(range(8), 4):
            for signs in itertools.product([1, -1], repeat=4):
                # Count -1s
                n_neg = sum(1 for s in signs if s == -1)
                if n_neg % 2 == 0:  # Even parity
                    v = [0] * D
                    for pos, s in zip(positions, signs):
                        v[pos] = s
                    e8_vectors.append(v)
        
        # E3-like vectors: 2 nonzeros in last 3 coords (positions 8,9,10)
        e3_vectors = []
        for positions in itertools.combinations(range(8, 11), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                e3_vectors.append(v)
        
        # Combine: E8 vector + E3 vector (but we need norm^2=4 total)
        # E8 vector has norm^2=4, E3 vector has norm^2=2, total norm^2=6. Too big.
        # Instead, use E8 vectors alone (norm^2=4) and E3 vectors alone (norm^2=2, not valid)
        # Or: use vectors with 2 nonzeros in first 8 coords + 2 nonzeros in last 3 coords (norm^2=4)
        
        # Better approach: construct vectors as (E8_root/√2) ⊕ (E3_root/√2) but keep integer
        # Use: 2 nonzeros in first 8 coords (norm^2=2) + 2 nonzeros in last 3 coords (norm^2=2) = norm^2=4
        
        mixed_vectors = []
        for pos8 in itertools.combinations(range(8), 2):
            for signs8 in itertools.product([1, -1], repeat=2):
                for pos3 in itertools.combinations(range(8, 11), 2):
                    for signs3 in itertools.product([1, -1], repeat=2):
                        v = [0] * D
                        for pos, s in zip(pos8, signs8):
                            v[pos] = s
                        for pos, s in zip(pos3, signs3):
                            v[pos] = s
                        mixed_vectors.append(v)
        
        # Greedily add vectors that don't conflict with already added ones
        # Start with E8-like vectors (4 nonzeros in first 8, even parity)
        added = set()
        for v in e8_vectors:
            v_tuple = tuple(v)
            if v_tuple not in vec_to_idx:
                continue
            idx = vec_to_idx[v_tuple]
            # Check if conflicts with any already added
            conflicts_with_added = False
            for added_idx in added:
                if conflicts[idx, added_idx]:
                    conflicts_with_added = True
                    break
            if not conflicts_with_added:
                in_set[idx] = True
                added.add(idx)
        
        # Try to add E3-like vectors (2 nonzeros in last 3 coords) - but norm^2=2, invalid
        # Instead, add mixed vectors (2+2) that don't conflict
        for v in mixed_vectors:
            v_tuple = tuple(v)
            if v_tuple not in vec_to_idx:
                continue
            idx = vec_to_idx[v_tuple]
            if in_set[idx]:
                continue
            # Check if conflicts with any already added
            conflicts_with_added = False
            for added_idx in added:
                if conflicts[idx, added_idx]:
                    conflicts_with_added = True
                    break
            if not conflicts_with_added:
                in_set[idx] = True
                added.add(idx)
        
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

        while time.time() < sa_deadline:
            if len(not_in_set_list) == 0:
                break

            counts = count_conflicts[not_in_set_list]
            min_count = np.min(counts)

            # Try 2-for-1 swap move with some probability
            if use_swaps and min_count >= 1 and rng.random() < swap_prob and len(not_in_set_list) >= 2:
                min_cands = not_in_set_list[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                u_cands = not_in_set_list[(count_conflicts[not_in_set_list] == 0)]
                u_cands = [u for u in u_cands if not conflicts[v, u]]
                if len(u_cands) > 0:
                    u = u_cands[rng.integers(len(u_cands))]
                    to_remove = np.unique(np.concatenate(
                        [np.where(conflicts[v] & in_set)[0],
                         np.where(conflicts[u] & in_set)[0]]))
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

                        not_in_set_list = not_in_set_list[
                            np.isin(not_in_set_list, np.concatenate(([v], [u])), invert=True)]

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
                zero_cands = not_in_set_list[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]

                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1

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
                to_remove = np.where(conflicts[v] & in_set)[0]

                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    not_in_set_list = not_in_set_list[~np.isin(not_in_set_list, to_remove)]

                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1

                not_in_set_list = not_in_set_list[not_in_set_list != v]

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

        while time.time() < polish_deadline:
            if len(not_in_set_list) == 0:
                break

            counts = count_conflicts[not_in_set_list]
            min_count = np.min(counts)

            if min_count == 0:
                zero_cands = not_in_set_list[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                not_in_set_list = not_in_set_list[not_in_set_list != v]
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
                to_remove = np.where(conflicts[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    not_in_set_list = not_in_set_list[~np.isin(not_in_set_list, to_remove)]
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                not_in_set_list = not_in_set_list[not_in_set_list != v]
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

    # First, try constructive initialization
    constructive_in_set = constructive_init_e8e3()
    constructive_size = constructive_in_set.sum()
    
    if constructive_size > best_size:
        best_size = constructive_size
        best_in_set = constructive_in_set.copy()

    # Run local search on constructive init
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 5 and constructive_size > 0:
        sa_time = min(remaining_time * 0.4, 15.0)
        if sa_time > 2:
            sa_result, sa_size = local_search(constructive_in_set, max_time=sa_time, use_swaps=True, swap_prob=0.15)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    # Multiple restarts with greedy + SA
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

        sa_time = min(remaining_time * 0.6, 12.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search(in_set, max_time=sa_time, use_swaps=True, swap_prob=0.15)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 8.0)
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
