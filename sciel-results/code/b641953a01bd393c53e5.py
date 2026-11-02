"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 115

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

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree(seed_offset=0):
        """Greedy independent set with min-degree heuristic, randomized tie-breaking."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        local_rng = np.random.default_rng(42 + seed_offset)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]

            if len(cands) > 1:
                scores = np.zeros(len(cands))
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_nbrs = nbrs_c[alive[nbrs_c]]
                    if len(alive_nbrs) > 0:
                        scores[idx] = np.sum(degrees[alive_nbrs])
                    else:
                        scores[idx] = 0
                noise = local_rng.random(len(cands)) * 0.1
                scores += noise
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

    def sa_local_search(in_set_init, max_time=10.0):
        """Enhanced simulated annealing with diverse moves and kick mechanism."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 4.0
        cooling = 0.9993
        min_T = 0.05

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0
        kick_counter = 0

        while time.time() < sa_deadline:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            if stuck_counter > 3000:
                n_remove = max(1, int(0.15 * len(in_set_indices)))
                if n_remove < len(in_set_indices):
                    remove_idx = rng.choice(len(in_set_indices), size=n_remove, replace=False)
                    for ri in remove_idx:
                        v = in_set_indices[ri]
                        in_set[v] = False
                        count_conflicts[neighbors[v]] -= 1
                    current_size -= n_remove
                    stuck_counter = 0
                    kick_counter += 1
                    for v in not_in_set:
                        if count_conflicts[v] == 0:
                            in_set[v] = True
                            count_conflicts[neighbors[v]] += 1
                            current_size += 1
                    not_in_set = np.where(~in_set)[0]
                    in_set_indices = np.where(in_set)[0]
                    continue

            move_type = rng.random()

            if move_type < 0.45 and len(not_in_set) > 0:
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
                else:
                    stuck_counter += 1

            elif move_type < 0.70 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                in_degrees = count_conflicts[in_set_indices]
                weights = 1.0 / (1.0 + in_degrees.astype(np.float64))
                weights /= weights.sum()
                v_remove = in_set_indices[rng.choice(len(in_set_indices), p=weights)]

                add_counts = count_conflicts[not_in_set]
                add_weights = 1.0 / (1.0 + add_counts.astype(np.float64))
                add_weights /= add_weights.sum()
                v_add = not_in_set[rng.choice(len(not_in_set), p=add_weights)]

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
                    else:
                        stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) >= 2 and len(not_in_set) > 0:
                idx1, idx2 = rng.choice(len(in_set_indices), size=2, replace=False)
                v_rem1 = in_set_indices[idx1]
                v_rem2 = in_set_indices[idx2]

                add_counts = count_conflicts[not_in_set]
                add_weights = 1.0 / (1.0 + add_counts.astype(np.float64))
                add_weights /= add_weights.sum()
                v_add = not_in_set[rng.choice(len(not_in_set), p=add_weights)]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[~np.isin(conflicts_with, [v_rem1, v_rem2])]

                if len(other_conflicts) == 0:
                    to_remove = [v_rem1, v_rem2]
                    delta = 1 - 2
                    accept = rng.random() < np.exp(delta / T)
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            else:
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
                    stuck_counter += 1

            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0

            if stuck_counter > 1500:
                T = max(T, 3.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def polish_greedy_add(in_set_init, max_time=5.0):
        """Polish by greedily adding non-conflicting vertices."""
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time

        for trial in range(5):
            if time.time() >= polish_deadline:
                break

            not_in = np.where(~polish_set)[0]
            if len(not_in) == 0:
                break

            conflict_counts = np.array([np.sum(conflicts[v] & polish_set) for v in not_in])

            added_any = True
            while added_any and time.time() < polish_deadline:
                added_any = False
                zero_conflict = not_in[conflict_counts == 0]
                if len(zero_conflict) > 0:
                    for v in zero_conflict:
                        polish_set[v] = True
                        polish_size += 1
                    remaining = np.where(~polish_set)[0]
                    if len(remaining) > 0:
                        conflict_counts = np.array([np.sum(conflicts[v] & polish_set) for v in remaining])
                        not_in = remaining
                    else:
                        break
                    added_any = True
                else:
                    break

        return polish_set, polish_size

    def branch_and_bound_mis(conflict_matrix, n, initial_solution_size, max_time=30.0, seed=42):
        """
        Branch-and-bound for Maximum Independent Set.
        Uses greedy coloring of the induced subgraph for upper bounds.
        conflict_matrix: bool array, conflict_matrix[i,j] = True if i and j conflict
        Returns: (best_independent_set_size, best_vertex_set)
        """
        local_rng = np.random.default_rng(seed)
        best_size = initial_solution_size
        best_set = None

        def greedy_color_upper_bound(vertices, cm):
            """
            Greedy coloring of the induced subgraph on 'vertices' in conflict graph cm.
            Returns upper bound on the size of a maximum independent set in this subgraph.
            """
            if len(vertices) == 0:
                return 0
            
            # Build compatibility subgraph
            compat = np.zeros((len(vertices), len(vertices)), dtype=bool)
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if not cm[vertices[i], vertices[j]]:
                        compat[i, j] = True
                        compat[j, i] = True
            
            # Greedy coloring of compatibility graph
            color_of = np.full(len(vertices), -1, dtype=np.int32)
            n_colors = 0
            uncolored = np.arange(len(vertices))
            
            while len(uncolored) > 0:
                n_colors += 1
                new_uncolored = []
                for v in uncolored:
                    can_color = True
                    for u in uncolored:
                        if u != v and color_of[u] == n_colors and compat[v, u]:
                            can_color = False
                            break
                    if can_color:
                        color_of[v] = n_colors
                    else:
                        new_uncolored.append(v)
                uncolored = new_uncolored
            
            return n_colors

        # B&B search
        all_vertices = np.arange(n)
        current_set = set()
        remaining = set(all_vertices)
        
        node_count = 0
        max_nodes = 500000

        def bb_search(current_set, remaining, upper_bound, best_size_ref):
            nonlocal best_size, best_set, node_count
            if time.time() > max_time:
                return
            if len(remaining) == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            node_count += 1
            if node_count > max_nodes:
                return
            
            if len(current_set) + upper_bound <= best_size:
                return
            
            rem_list = list(remaining)
            if len(rem_list) == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            degs = np.array([sum(1 for u in remaining if u != v and conflict_matrix[v, u]) for v in rem_list])
            v = rem_list[np.argmax(degs)]
            
            new_set = current_set | {v}
            nbrs = set(np.where(conflict_matrix[v])[0])
            new_remaining = remaining - {v} - nbrs
            
            ub1 = len(new_set) + greedy_color_upper_bound(list(new_remaining), conflict_matrix)
            if ub1 > best_size:
                bb_search(new_set, new_remaining, ub1 - len(new_set), best_size)
            
            new_remaining2 = remaining - {v}
            ub2 = len(current_set) + greedy_color_upper_bound(list(new_remaining2), conflict_matrix)
            if ub2 > best_size:
                bb_search(current_set, new_remaining2, ub2 - len(current_set), best_size)
        
        initial_ub = greedy_color_upper_bound(list(all_vertices), conflict_matrix)
        if initial_ub > best_size:
            bb_search(set(), set(all_vertices), initial_ub, best_size)
        
        if best_set is None:
            best_set = set()
        
        return best_size, best_set

    # Run greedy + SA first to get a good initial solution
    n_restarts = 0
    max_restarts = 30

    while time.time() < deadline - 20 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree(n_restarts)
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 20 - time.time()
        if remaining_time < 3:
            break

        sa_time = min(remaining_time * 0.4, 8.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        remaining_time = deadline - 15 - time.time()
        if remaining_time > 5:
            polish_time = min(remaining_time * 0.3, 5.0)
            if polish_time > 1:
                polish_result, polish_size = polish_greedy_add(sa_result if sa_size >= current_size else in_set, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Now try branch-and-bound with the best solution as initial bound
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 10:
        bb_time = min(remaining_time * 0.8, 60.0)
        if bb_time > 5:
            try:
                bb_size, bb_set = branch_and_bound_mis(
                    conflicts, N4, best_size, 
                    max_time=bb_time, seed=123
                )
                if bb_size > best_size:
                    best_size = bb_size
                    best_in_set = np.zeros(N4, dtype=bool)
                    for v in bb_set:
                        best_in_set[v] = True
            except Exception:
                pass

    # Final polish on best found
    remaining_time = deadline - 3 - time.time()
    if remaining_time > 3:
        final_polish, final_size = polish_greedy_add(best_in_set, max_time=min(remaining_time * 0.6, 8.0))
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_polish.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
