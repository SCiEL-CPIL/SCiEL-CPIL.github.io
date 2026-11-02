"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

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

    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)

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
        """Enhanced simulated annealing with diverse moves."""
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

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            move_type = rng.random()

            if move_type < 0.5 and len(not_in_set) > 0:
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
                else:
                    stuck_counter += 1

            elif move_type < 0.7 and len(in_set_indices) > 0 and len(not_in_set) > 0:
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
                    else:
                        stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) >= 2 and len(not_in_set) > 0:
                idx1, idx2 = rng.choice(len(in_set_indices), size=2, replace=False)
                v_rem1 = in_set_indices[idx1]
                v_rem2 = in_set_indices[idx2]
                v_add = not_in_set[rng.integers(len(not_in_set))]

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

    def crossover_merge(set_a, set_b):
        """
        Merge two independent sets by greedily selecting from their union.
        The union contains many vertices; we pick a maximal independent set
        from it using a conflict-aware greedy strategy.
        """
        union_mask = set_a | set_b
        union_indices = np.where(union_mask)[0]
        if len(union_indices) == 0:
            return np.zeros(N4, dtype=bool), 0

        # Greedy: repeatedly pick the vertex in the union that has the fewest
        # conflicts with other vertices still in the union (and not yet removed).
        # This tends to keep more vertices.
        result = np.zeros(N4, dtype=bool)
        alive = union_mask.copy()

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            # For each alive vertex, count how many of its neighbors are also alive
            # (i.e., would be removed if we pick this vertex)
            # Pick the one with minimum "cost" = number of alive neighbors
            min_cost = np.inf
            chosen = -1
            for v in alive_indices:
                nbrs = neighbors[v]
                cost = np.sum(alive[nbrs])
                if cost < min_cost:
                    min_cost = cost
                    chosen = v
                elif cost == min_cost:
                    # Tie-break: prefer lower original degree
                    if neighbor_counts[v] < neighbor_counts[chosen]:
                        chosen = v

            result[chosen] = True
            # Remove chosen and its neighbors from alive
            nbrs = neighbors[chosen]
            alive[chosen] = False
            alive[nbrs] = False

        return result, int(result.sum())

    def polish_greedy_add(in_set_init, max_time=5.0):
        """Polish by greedily adding non-conflicting vertices in random order."""
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time

        for trial in range(3):
            if time.time() >= polish_deadline:
                break

            not_in = np.where(~polish_set)[0]
            if len(not_in) == 0:
                break

            order = rng.permutation(len(not_in))

            for idx in order:
                if time.time() >= polish_deadline:
                    break

                v = not_in[idx]

                if np.any(conflicts[v] & polish_set):
                    continue

                polish_set[v] = True
                polish_size += 1

        return polish_set, polish_size

    # Population for genetic algorithm
    population = []

    # Initialize population with diverse greedy solutions
    n_init = 6
    for i in range(n_init):
        if time.time() >= deadline - 10:
            break
        in_set = greedy_min_degree()
        sz = int(in_set.sum())
        population.append((in_set, sz))
        if sz > best_size:
            best_size = sz
            best_in_set = in_set.copy()

    # Genetic algorithm loop
    gen = 0
    max_gens = 30

    while time.time() < deadline - 10 and gen < max_gens:
        gen += 1

        # Sort population by size, keep top half
        population.sort(key=lambda x: x[1], reverse=True)
        pop_size = len(population)
        keep = max(2, pop_size // 2)
        population = population[:keep]

        # Generate children via crossover + SA polish
        children = []
        n_children = max(2, keep)

        for c in range(n_children):
            if time.time() >= deadline - 12:
                break

            # Pick two parents (tournament selection)
            p1 = population[rng.integers(len(population))][0]
            p2 = population[rng.integers(len(population))][0]

            # Crossover: merge via greedy from union
            child_set, child_size = crossover_merge(p1, p2)

            # SA polish on the child
            remaining = deadline - 12 - time.time()
            if remaining > 3:
                sa_time = min(remaining * 0.3, 8.0)
                if sa_time > 1:
                    sa_result, sa_size = sa_local_search(child_set, max_time=sa_time)
                    child_set = sa_result
                    child_size = sa_size

            # Greedy add polish
            remaining2 = deadline - 10 - time.time()
            if remaining2 > 2:
                pol_time = min(remaining2 * 0.3, 4.0)
                if pol_time > 0.5:
                    pol_result, pol_size = polish_greedy_add(child_set, max_time=pol_time)
                    if pol_size >= child_size:
                        child_set = pol_result
                        child_size = pol_size

            children.append((child_set, child_size))

            if child_size > best_size:
                best_size = child_size
                best_in_set = child_set.copy()

        # Update population: keep best parents + children
        population.extend(children)
        population.sort(key=lambda x: x[1], reverse=True)
        population = population[:max(4, keep)]

        # Also run SA on the current best to try to improve it
        if time.time() < deadline - 8:
            sa_time = min((deadline - 8 - time.time()) * 0.4, 10.0)
            if sa_time > 2:
                sa_best, sa_best_size = sa_local_search(best_in_set, max_time=sa_time)
                if sa_best_size > best_size:
                    best_size = sa_best_size
                    best_in_set = sa_best.copy()

    # Final polish
    if time.time() < deadline - 3:
        pol_time = min(deadline - 3 - time.time(), 5.0)
        if pol_time > 1:
            final_polish, final_size = polish_greedy_add(best_in_set, max_time=pol_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_polish.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
