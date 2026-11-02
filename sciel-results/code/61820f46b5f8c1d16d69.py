"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

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
    N1 = len(vectors_v1)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix for W4 vectors: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists as arrays for faster SA updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    # Compute conflicts between W1 and W4 vectors
    # W1 vectors have norm^2=4, W4 have norm^2=4
    # Conflict if dot > 2
    cross_dots = vectors_v1 @ vectors_v4.T  # shape (N1, N4)
    cross_conflicts = cross_dots > 2  # shape (N1, N4)

    # For each W1 vector, find which W4 vectors conflict with it
    w1_conflicting_w4 = [np.where(cross_conflicts[i])[0] for i in range(N1)]

    # For each W4 vector, find which W1 vectors conflict with it
    w4_conflicting_w1 = [np.where(cross_conflicts[:, i])[0] for i in range(N4)]

    # Conflicts among W1 vectors
    # Two W1 vectors conflict if they have the same coordinate and same sign (dot=4)
    # or same coordinate opposite sign (dot=-4, no conflict)
    # Different coordinates: dot=0, no conflict
    w1_conflicts = np.zeros((N1, N1), dtype=bool)
    for i in range(N1):
        for j in range(i + 1, N1):
            dot = int(np.dot(vectors_v1[i], vectors_v1[j]))
            if dot > 2:
                w1_conflicts[i, j] = True
                w1_conflicts[j, i] = True

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set_w4 = np.zeros(N4, dtype=bool)
    best_in_set_w1 = np.zeros(N1, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic on W4 vectors."""
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

    def greedy_max_coverage():
        """Greedy: pick vertex covering most non-in-set vertices (max conflict spread)."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            # Pick vertex with max degree among alive (covers most)
            max_deg = np.max(neighbor_counts[alive_indices])
            cands = alive_indices[neighbor_counts[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

        return in_set

    def add_w1_vectors(in_set_w4):
        """Add W1 vectors that don't conflict with selected W4 or each other."""
        in_set_w1 = np.zeros(N1, dtype=bool)
        # For each W1, check if it conflicts with any selected W4
        for i in range(N1):
            conflicting_w4 = w1_conflicting_w4[i]
            if np.any(in_set_w4[conflicting_w4]):
                continue
            # Check conflicts with already selected W1
            if np.any(w1_conflicts[i] & in_set_w1):
                continue
            in_set_w1[i] = True
        return in_set_w1

    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict SA with add/remove, swap, and 2-for-1 moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 3.0
        cooling = 0.9997
        min_T = 0.03

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            # Choose move type probabilistically
            move_type = rng.random()
            if move_type < 0.50:
                # ADD/REMOVE move
                if len(not_in_set) == 0:
                    break
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = min_count
                delta = 1 - k
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
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
                stuck_counter += 0 if (accept and delta >= 0) else 1

            elif move_type < 0.80:
                # SWAP move: remove one in-set, add one not-in-set
                if len(not_in_set) == 0 or len(in_set_indices) == 0:
                    break
                # Pick a random in-set vertex to remove
                u = in_set_indices[rng.integers(len(in_set_indices))]
                # Pick a not-in-set vertex that conflicts with u (or random)
                # Prefer vertices that conflict with u to make it a meaningful swap
                conflicting_with_u = np.where(conflicts[u] & ~in_set)[0]
                if len(conflicting_with_u) > 0 and rng.random() < 0.7:
                    v = conflicting_with_u[rng.integers(len(conflicting_with_u))]
                else:
                    v = not_in_set[rng.integers(len(not_in_set))]

                # Compute delta: remove u, add v
                # After removing u: size -1
                # Adding v: conflicts with (in_set \ {u}) ∩ conflicts[v]
                in_set_minus_u = in_set.copy()
                in_set_minus_u[u] = False
                to_remove_for_v = np.where(conflicts[v] & in_set_minus_u)[0]
                delta = -1 + 1 - len(to_remove_for_v)
                # Actually: remove u (-1), add v (+1), remove to_remove_for_v (-len)
                # But to_remove_for_v doesn't include u since we removed it
                # Net change: -1 (remove u) + 1 (add v) - len(to_remove_for_v)
                delta = -1 + 1 - len(to_remove_for_v)

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    # Remove u
                    in_set[u] = False
                    for w in neighbors[u]:
                        count_conflicts[w] -= 1
                    # Remove to_remove_for_v
                    for w in to_remove_for_v:
                        in_set[w] = False
                        for z in neighbors[w]:
                            count_conflicts[z] -= 1
                    # Add v
                    in_set[v] = True
                    for w in neighbors[v]:
                        count_conflicts[w] += 1
                    current_size += delta
                stuck_counter += 0 if (accept and delta >= 0) else 1

            else:
                # 2-FOR-1 move: remove two in-set vertices, add one not-in-set
                if len(not_in_set) < 1 or len(in_set_indices) < 2:
                    break
                v = not_in_set[rng.integers(len(not_in_set))]
                # Find which in-set vertices conflict with v
                conflicting_in_set = np.where(conflicts[v] & in_set)[0]
                if len(conflicting_in_set) >= 2:
                    # Pick two to remove
                    idx1 = rng.integers(len(conflicting_in_set))
                    idx2 = rng.integers(len(conflicting_in_set) - 1)
                    if idx2 >= idx1:
                        idx2 += 1
                    u1 = conflicting_in_set[idx1]
                    u2 = conflicting_in_set[idx2]
                    # After removing u1, u2: does v still conflict with remaining in_set?
                    in_set_minus = in_set.copy()
                    in_set_minus[u1] = False
                    in_set_minus[u2] = False
                    still_conflict = np.where(conflicts[v] & in_set_minus)[0]
                    delta = -2 + 1 - len(still_conflict)
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.5
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    if accept:
                        for w in [u1, u2]:
                            in_set[w] = False
                            for z in neighbors[w]:
                                count_conflicts[z] -= 1
                        for w in still_conflict:
                            in_set[w] = False
                            for z in neighbors[w]:
                                count_conflicts[z] -= 1
                        in_set[v] = True
                        for w in neighbors[v]:
                            count_conflicts[w] += 1
                        current_size += delta
                    stuck_counter += 0 if (accept and delta >= 0) else 1
                else:
                    stuck_counter += 1

            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0

            if stuck_counter > 2000:
                T = max(T, 2.5)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def polish_zero_conflict(in_set_w4):
        """Add any W4 vectors with zero conflicts to the current set."""
        in_set = in_set_w4.copy()
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            # Find vertices with no neighbors in set
            zero_conf = not_in_set[np.array([np.sum(in_set[neighbors[v]]) == 0 for v in not_in_set])]
            if len(zero_conf) > 0:
                in_set[zero_conf] = True
                changed = True
        return in_set

    # Multiple restarts
    n_restarts = 0
    max_restarts = 60

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # Alternate between greedy strategies
        if n_restarts % 3 == 1:
            in_set = greedy_min_degree()
        elif n_restarts % 3 == 2:
            in_set = greedy_max_coverage()
        else:
            in_set = greedy_min_degree()

        current_size = in_set.sum()

        # 2. Polish with zero-conflict additions
        in_set = polish_zero_conflict(in_set)
        current_size = in_set.sum()

        # 3. Add W1 vectors
        in_set_w1 = add_w1_vectors(in_set)
        total_size = current_size + in_set_w1.sum()

        if total_size > best_size:
            best_size = total_size
            best_in_set_w4 = in_set.copy()
            best_in_set_w1 = in_set_w1.copy()

        # 4. SA Local Search with enhanced moves
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        sa_time = min(remaining_time * 0.5, 15.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)

        # Polish SA result
        sa_result = polish_zero_conflict(sa_result)
        sa_size = sa_result.sum()

        # Add W1 to SA result
        sa_in_set_w1 = add_w1_vectors(sa_result)
        sa_total = sa_size + sa_in_set_w1.sum()

        if sa_total > best_size:
            best_size = sa_total
            best_in_set_w4 = sa_result.copy()
            best_in_set_w1 = sa_in_set_w1.copy()

        # 5. Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_total > current_size:
            polish_time = min(remaining_time * 0.4, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                polish_result = polish_zero_conflict(polish_result)
                polish_size = polish_result.sum()
                polish_in_set_w1 = add_w1_vectors(polish_result)
                polish_total = polish_size + polish_in_set_w1.sum()
                if polish_total > best_size:
                    best_size = polish_total
                    best_in_set_w4 = polish_result.copy()
                    best_in_set_w1 = polish_in_set_w1.copy()

    # Final result
    final_v4_indices = np.where(best_in_set_w4)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_v1_indices = np.where(best_in_set_w1)[0]
    result_v1 = vectors_v1[final_v1_indices]

    if len(result_v4) > 0 and len(result_v1) > 0:
        final_vectors = np.vstack([result_v4, result_v1])
    elif len(result_v4) > 0:
        final_vectors = result_v4
    elif len(result_v1) > 0:
        final_vectors = result_v1
    else:
        final_vectors = vectors_v1  # fallback

    return final_vectors

# EVOLVE-BLOCK-END
