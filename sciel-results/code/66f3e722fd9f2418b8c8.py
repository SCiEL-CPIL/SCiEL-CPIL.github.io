"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all D11 minimal vectors (squared norm 4)
    # Type 1: two coordinates ±2, rest 0 -> C(11,2)*4 = 220 vectors
    # Type 2: four coordinates ±1, rest 0 -> C(11,4)*16 = 5280 vectors
    v1_list = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    v1_list.append(v)

    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    all_vectors = np.array(v1_list + v4_list, dtype=np.int64)
    N = len(all_vectors)

    # Check if all vectors are mutually at distance >= 2
    dots = all_vectors @ all_vectors.T
    np.fill_diagonal(dots, 0)

    if np.max(dots) <= 2:
        return all_vectors

    # Fallback: need to find maximum independent set
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
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

    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        T = 3.0
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
            if len(not_in_set) == 0:
                break
            counts = count_conflicts[not_in_set]
            min_count = int(np.min(counts))
            min_cands = not_in_set[counts == min_count]
            v = int(min_cands[rng.integers(len(min_cands))])
            delta = 1 - min_count
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

    def genetic_crossover(parent1, parent2):
        """Merge two independent sets: keep non-conflicting vectors, resolve conflicts."""
        # Start with the union, then remove conflicts
        child = parent1 | parent2
        child_size = int(child.sum())

        # Resolve conflicts: iteratively remove the vector with higher conflict count
        changed = True
        max_iter = 1000
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            # Find all vectors in child that conflict with at least one other in child
            conflict_counts = np.zeros(N, dtype=np.int32)
            child_indices = np.where(child)[0]
            for i in child_indices:
                conflict_counts[neighbors[i]] += 1
            
            # We want to count conflicts only between nodes in child.
            # conflict_counts[i] is the number of neighbors of i that are in child.
            # So if child[i] is True, conflict_counts[i] is the number of conflicts for i.
            
            # Find vectors in child with conflicts
            conflicted_mask = (conflict_counts > 0) & child
            conflicted = np.where(conflicted_mask)[0]
            if len(conflicted) == 0:
                break

            # Remove the vector with the highest conflict count
            max_conf = int(np.max(conflict_counts[conflicted]))
            to_remove_cands = conflicted[conflict_counts[conflicted] == max_conf]
            
            # Prefer removing from the smaller parent if possible
            # Check which parents contain these candidates
            p1_in_cands = parent1[to_remove_cands]
            p2_in_cands = parent2[to_remove_cands]
            
            num_in_p1 = np.sum(p1_in_cands)
            num_in_p2 = np.sum(p2_in_cands)
            
            if num_in_p1 < num_in_p2:
                # Prefer removing from parent2 (keep more from P1)
                # Find candidates in P2
                p2_cands_indices = to_remove_cands[p2_in_cands]
                if len(p2_cands_indices) > 0:
                    v = int(p2_cands_indices[rng.integers(len(p2_cands_indices))])
                else:
                    v = int(to_remove_cands[rng.integers(len(to_remove_cands))])
            elif num_in_p2 < num_in_p1:
                # Prefer removing from parent1
                p1_cands_indices = to_remove_cands[p1_in_cands]
                if len(p1_cands_indices) > 0:
                    v = int(p1_cands_indices[rng.integers(len(p1_cands_indices))])
                else:
                    v = int(to_remove_cands[rng.integers(len(to_remove_cands))])
            else:
                v = int(to_remove_cands[rng.integers(len(to_remove_cands))])

            child[v] = False
            child_size -= 1
            changed = True

        return child, child_size

    def ga_local_search(in_set_init, max_time=8.0):
        """Local search that adds non-conflicting vectors and swaps to improve."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ga_deadline = time.time() + max_time

        # First pass: add all non-conflicting vectors
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        # Add zero-conflict vectors
        zero_conf = np.where((count_conflicts == 0) & ~in_set)[0]
        if len(zero_conf) > 0:
            in_set[zero_conf] = True
            current_size += len(zero_conf)
            for v in zero_conf:
                count_conflicts[neighbors[v]] += 1
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()

        # SA-style local search for swaps
        T = 2.0
        cooling = 0.999
        min_T = 0.1
        stuck_counter = 0

        while time.time() < ga_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = count_conflicts[not_in_set]
            min_count = int(np.min(counts))
            min_cands = not_in_set[counts == min_count]
            v = int(min_cands[rng.integers(len(min_cands))])
            delta = 1 - min_count
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
            if stuck_counter > 2000:
                T = max(T, 1.5)
                stuck_counter = 0
            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # Phase 1: Generate initial population via greedy
    population = []
    pop_sizes = []
    for _ in range(8):
        in_set = greedy_min_degree()
        sz = int(in_set.sum())
        population.append(in_set)
        pop_sizes.append(sz)
        if sz > best_size:
            best_size = sz
            best_in_set = in_set.copy()

    # Phase 2: Genetic algorithm with crossover
    gen = 0
    max_gens = 200
    while time.time() < deadline - 10 and gen < max_gens:
        gen += 1
        remaining = deadline - 10 - time.time()
        if remaining < 3:
            break

        # Tournament selection
        def tournament(k=3):
            indices = rng.integers(len(population), size=k)
            best_idx = indices[0]
            best_sz = pop_sizes[best_idx]
            for idx in indices[1:]:
                if pop_sizes[idx] > best_sz:
                    best_idx = idx
                    best_sz = pop_sizes[idx]
            return population[best_idx]

        # Crossover two parents
        p1 = tournament()
        p2 = tournament()
        child, child_size = genetic_crossover(p1, p2)

        # Local search on child
        child, child_size = ga_local_search(child, max_time=min(remaining * 0.3, 5.0))

        # Update population: replace worst if child is better
        worst_idx = int(np.argmin(pop_sizes))
        if child_size > pop_sizes[worst_idx]:
            population[worst_idx] = child
            pop_sizes[worst_idx] = child_size

        if child_size > best_size:
            best_size = child_size
            best_in_set = child.copy()

        # Also try SA on the best individual occasionally
        if gen % 5 == 0 and remaining > 8:
            best_idx = int(np.argmax(pop_sizes))
            sa_res, sa_sz = sa_local_search(population[best_idx], max_time=min(remaining * 0.4, 8.0))
            if sa_sz > best_size:
                best_size = sa_sz
                best_in_set = sa_res.copy()
            # Replace worst with SA result if better
            if sa_sz > pop_sizes[worst_idx]:
                population[worst_idx] = sa_res
                pop_sizes[worst_idx] = sa_sz

    # Final polish
    remaining = deadline - 3 - time.time()
    if remaining > 3:
        final_res, final_sz = sa_local_search(best_in_set, max_time=min(remaining, 15.0))
        if final_sz > best_size:
            best_size = final_sz
            best_in_set = final_res.copy()

    final_indices = np.where(best_in_set)[0]
    return all_vectors[final_indices]

# EVOLVE-BLOCK-END
