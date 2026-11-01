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

    def local_search(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Hybrid Greedy + Simulated Annealing with 1-add and 2-swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

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

        # Cache not-in-set indices
        not_in_set = np.where(~in_set)[0]

        step = 0
        while time.time() < sa_deadline:
            step += 1
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)

            if min_count == 0:
                # Add a 0-conflict node
                zero_cands = not_in_set[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]

                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1

                # Remove v from not_in_set
                mask = not_in_set != v
                not_in_set = not_in_set[mask]

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # Every 10 steps, try a 2-swap move for better exploration
            if step % 10 == 0 and current_size >= 2:
                # Pick two random nodes to remove from in_set
                in_set_idx = np.where(in_set)[0]
                if len(in_set_idx) >= 2:
                    i1, i2 = rng.choice(in_set_idx, size=2, replace=False)
                    
                    # Temporarily remove both
                    in_set[i1] = False
                    in_set[i2] = False
                    current_size -= 2
                    count_conflicts[neighbors[i1]] -= 1
                    count_conflicts[neighbors[i2]] -= 1
                    
                    # Find addable nodes
                    not_in_set_now = np.where(~in_set)[0]
                    if len(not_in_set_now) > 0:
                        counts_now = count_conflicts[not_in_set_now]
                        addable = not_in_set_now[counts_now == 0]
                        
                        if len(addable) >= 1:
                            # Try to add 1 or 2 nodes
                            if len(addable) >= 2 and rng.random() < 0.3:
                                # Add 2 nodes
                                v1, v2 = rng.choice(addable, size=2, replace=False)
                                in_set[v1] = True
                                in_set[v2] = True
                                current_size += 2
                                count_conflicts[neighbors[v1]] += 1
                                count_conflicts[neighbors[v2]] += 1
                                delta = 0
                            else:
                                # Add 1 node
                                v = addable[rng.integers(len(addable))]
                                in_set[v] = True
                                current_size += 1
                                count_conflicts[neighbors[v]] += 1
                                delta = -1
                            
                            # Update not_in_set
                            mask = np.isin(not_in_set, [i1, i2], invert=True)
                            not_in_set = not_in_set[mask]
                            # Add back i1, i2 if not already there
                            new_items = [x for x in [i1, i2] if not in_set[x]]
                            if new_items:
                                not_in_set = np.concatenate([not_in_set, np.array(new_items)])
                            
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                            continue
                        else:
                            # No addable, restore
                            in_set[i1] = True
                            in_set[i2] = True
                            current_size += 2
                            count_conflicts[neighbors[i1]] += 1
                            count_conflicts[neighbors[i2]] += 1
                    
                    # If we didn't do a 2-swap, continue to 1-move
                    # (i1, i2 already restored above if no addable)
                    continue

            # SA step: pick node with minimum conflict count
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
                # Remove neighbors of v that are in set
                to_remove = np.where(conflicts[v] & in_set)[0]

                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1

                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1

                # Update not_in_set: remove v, add back to_remove
                if len(to_remove) > 0:
                    mask = not_in_set != v
                    not_in_set = np.concatenate([not_in_set[mask], to_remove])
                else:
                    mask = not_in_set != v
                    not_in_set = not_in_set[mask]

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
        """Exact local search: try all single additions and 1-swap moves (optimized)."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Try adding nodes with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                if len(zero_cands) > 0:
                    for v in zero_cands:
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()

            # Try 1-swap: remove one node i, add node v if it becomes conflict-free
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                # Temporarily remove i
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1

                # Only check nodes that conflict with i (they might now be addable)
                i_neighbors = neighbors[i]
                # Among i's neighbors, find those not in set with 0 remaining conflicts
                cand_mask = ~in_set[i_neighbors]
                if np.any(cand_mask):
                    cand_indices = i_neighbors[cand_mask]
                    cand_counts = count_conflicts[cand_indices]
                    addable = cand_indices[cand_counts == 0]
                    if len(addable) > 0:
                        # Pick the one that unblocks the most other nodes
                        # Score by number of neighbors it conflicts with that are not in set
                        best_v = addable[0]
                        best_score = -1
                        for v in addable:
                            v_nbrs = neighbors[v]
                            # Count how many of v's neighbors are not in set (potential future additions)
                            score = np.sum(~in_set[v_nbrs])
                            if score > best_score:
                                best_score = score
                                best_v = v
                        v = best_v
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        # Keep i removed and v added (net zero change in size)
                        continue
                # No swap possible, restore i
                in_set[i] = True
                current_size += 1
                count_conflicts[neighbors[i]] += 1

        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 80

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        # 1. Greedy Init with varying strategy
        if n_restarts % 3 == 0:
            # Standard min-degree
            in_set = greedy_min_degree()
        elif n_restarts % 3 == 1:
            # Randomized greedy: pick from top-3 min-degree nodes
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            degrees = neighbor_counts.copy()
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                # Pick randomly from top candidates
                top_k = min(3, len(cands))
                chosen = cands[rng.integers(top_k)]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        else:
            # Random seed for diversity
            rng_local = np.random.default_rng(n_restarts * 12345)
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            degrees = neighbor_counts.copy()
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                chosen = cands[rng_local.integers(len(cands))]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.45, 12.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Polish with exact method
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size >= current_size:
            polish_time = min(remaining_time * 0.35, 6.0)
            if polish_time > 1:
                polish_result, polish_size = exact_polish(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final polish on best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        # Multiple exact polish passes
        for pass_num in range(3):
            remaining_time = deadline - 1 - pass_num - time.time()
            if remaining_time < 0.5:
                break
            polish_time = min(remaining_time * 0.6, 4.0)
            if polish_time < 0.5:
                break
            polish_result, polish_size = exact_polish(best_in_set, max_time=polish_time)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
            else:
                # No improvement, try SA with very low temperature
                remaining_time = deadline - 1 - time.time()
                if remaining_time > 1:
                    sa_time = min(remaining_time * 0.5, 2.0)
                    if sa_time > 0.5:
                        sa_result, sa_size = local_search(
                            best_in_set, max_time=sa_time, min_T=0.01, cooling=0.9999
                        )
                        if sa_size > best_size:
                            best_size = sa_size
                            best_in_set = sa_result.copy()
                        break

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
