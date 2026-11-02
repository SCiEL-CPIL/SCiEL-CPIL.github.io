"""Rewritten program for kissing number in dimension 11."""
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

    # Compute conflict matrix: conflict if |dot| > 2 (distance^2 = 8 - 2*dot < 4 when dot > 2,
    # but we need distance >= max norm = 2, so distance^2 >= 4, so 8 - 2*dot >= 4 => dot <= 2)
    # Wait: distance^2 = ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 4 + 4 - 2*a.b = 8 - 2*a.b
    # We need distance >= 2, so distance^2 >= 4, so 8 - 2*a.b >= 4, so a.b <= 2.
    # Also need to consider negative dot products: if a.b is very negative, distance is large, which is fine.
    # The constraint is a.b <= 2. So conflict when a.b > 2.
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(12345)

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

    def sa_local_search(in_set_init, max_time=10.0):
        """Simulated annealing local search for maximum independent set."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 5.0
        cooling = 0.999
        min_T = 0.01

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

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 2000:
                T = max(T, 3.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def hill_climbing(in_set_init, max_time=10.0):
        """Hill climbing: only accept moves that increase size."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        hc_deadline = time.time() + max_time

        while time.time() < hc_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)

            if min_count > 0:
                break  # No improving move available

            min_cands = not_in_set[counts == 0]
            v = min_cands[rng.integers(len(min_cands))]

            in_set[v] = True
            current_size += 1
            count_conflicts[neighbors[v]] += 1

            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()

        return local_best_in_set, local_best_size

    # Multiple restarts with different strategies
    n_restarts = 0
    max_restarts = 100

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        # Alternate between SA and hill climbing based on restart number
        if n_restarts % 3 == 0:
            # Hill climbing
            hc_time = min(remaining_time * 0.7, 12.0)
            if hc_time < 2:
                hc_time = 2.0
            hc_result, hc_size = hill_climbing(in_set, max_time=hc_time)
            if hc_size > best_size:
                best_size = hc_size
                best_in_set = hc_result.copy()
        else:
            # Simulated annealing
            sa_time = min(remaining_time * 0.6, 15.0)
            if sa_time < 2:
                sa_time = 2.0
            sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

            # Second pass: hill climb on SA result
            remaining_time2 = deadline - 5 - time.time()
            if remaining_time2 > 3 and sa_size > current_size:
                hc_time2 = min(remaining_time2 * 0.4, 8.0)
                if hc_time2 > 1:
                    hc_result2, hc_size2 = hill_climbing(sa_result, max_time=hc_time2)
                    if hc_size2 > best_size:
                        best_size = hc_size2
                        best_in_set = hc_result2.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
