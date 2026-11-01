"""Kissing number in d=11: greedy min-degree independent set + local search with swaps."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Generate candidate vectors with norm^2 = 4
    # Type A: weight 4, entries in {-1,0,1}, norm^2 = 4
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Type B: weight 1, entry ±2, norm^2 = 4
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute dot product matrix
    dots = all_candidates @ all_candidates.T
    # All vectors have norm^2 = 4, so ||u||*||v|| = 4, threshold = 4/2 = 2
    # Conflict if dot > 2 (i.e., distance^2 < 4)
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    def greedy_min_degree(seed):
        """Greedy independent set with min-degree selection and random tie-breaking."""
        rng = random.Random(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 3:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.randrange(len(cands))]

            in_set[chosen] = True
            alive[chosen] = False

            nbrs = neighbors[chosen]
            alive[nbrs] = False

            # Update degrees: for each alive vertex, subtract number of neighbors removed
            rem = np.where(alive)[0]
            if len(rem) > 0:
                removed = np.zeros(N, dtype=bool)
                removed[chosen] = True
                removed[nbrs] = True
                for i in rem:
                    deg_decr = removed[neighbors[i]].sum()
                    degrees[i] -= deg_decr

        return in_set

    def local_search(in_set_init, seed, max_time=3.0):
        """Local search: add zero-conflict vertices, 1-for-1 swaps, 2-for-1 swaps, perturbation."""
        rng = random.Random(seed)
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        # Count conflicts for each vertex from the current set
        conflict_count = np.zeros(N, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            conflict_count[neighbors[i]] += 1

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time
        no_improve = 0

        while time.time() < ls_deadline:
            improved = False

            # Phase 1: Add vertices with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                zero_conf = not_in_set[conflict_count[not_in_set] == 0]
                while len(zero_conf) > 0 and time.time() < ls_deadline:
                    v = zero_conf[rng.randrange(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0

                    # Recompute zero_conf
                    not_in_set = np.where(~in_set)[0]
                    if len(not_in_set) == 0:
                        break
                    zero_conf = not_in_set[conflict_count[not_in_set] == 0]

            # Phase 2: Try 1-for-1 swaps
            if time.time() < ls_deadline:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]

                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    n_swap_tries = 0
                    max_swap_tries = 200
                    while n_swap_tries < max_swap_tries and time.time() < ls_deadline:
                        n_swap_tries += 1
                        u = in_set_indices[rng.randrange(len(in_set_indices))]
                        v_candidates = np.where(conflicts[u] & ~in_set)[0]
                        if len(v_candidates) == 0:
                            continue
                        v = v_candidates[rng.randrange(len(v_candidates))]
                        v_conflicts_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(v_conflicts_in_set) == 1 and v_conflicts_in_set[0] == u:
                            in_set[u] = False
                            in_set[v] = True
                            conflict_count[neighbors[u]] -= 1
                            conflict_count[neighbors[v]] += 1
                            improved = True
                            in_set_indices = np.where(in_set)[0]
                            not_in_set = np.where(~in_set)[0]

            # Phase 3: Try 2-for-1 swaps (remove 2, add 1 that conflicts with both removed)
            if time.time() < ls_deadline:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]

                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    n_2swap_tries = 0
                    max_2swap_tries = 100
                    while n_2swap_tries < max_2swap_tries and time.time() < ls_deadline:
                        n_2swap_tries += 1
                        # Pick a random vertex not in set
                        v = not_in_set[rng.randrange(len(not_in_set))]
                        v_conf = np.where(conflicts[v] & in_set)[0]
                        # v must conflict with exactly 2 set vertices (or we can remove a subset)
                        if len(v_conf) <= 2 and len(v_conf) >= 1:
                            # Check: if we remove all of v_conf, can we add v?
                            # v_conf has at most 2 elements
                            # Net gain: +1 - len(v_conf). Only do if len(v_conf) == 1 (already handled)
                            # For 2-for-1: len(v_conf)==2, net -1, skip unless it enables further adds
                            # Actually 2-for-1 means remove 2 add 1, net -1. That's bad.
                            # Let's skip this; instead try: remove 1, add 2 (1-for-2)
                            pass

            # Phase 3b: Try 1-for-2 swaps (remove 1, add 2 that each only conflict with the removed one)
            if time.time() < ls_deadline:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]

                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    n_1for2_tries = 0
                    max_1for2_tries = 50
                    while n_1for2_tries < max_1for2_tries and time.time() < ls_deadline:
                        n_1for2_tries += 1
                        u = in_set_indices[rng.randrange(len(in_set_indices))]
                        # Vertices not in set that conflict with u
                        v_candidates = np.where(conflicts[u] & ~in_set)[0]
                        if len(v_candidates) < 2:
                            continue
                        # Pick 2 random candidates
                        idx1, idx2 = rng.sample(range(len(v_candidates)), 2)
                        v1 = v_candidates[idx1]
                        v2 = v_candidates[idx2]
                        # Check v1 and v2 don't conflict with each other
                        if conflicts[v1, v2]:
                            continue
                        # Check v1 only conflicts with u in the set
                        v1_conf = np.where(conflicts[v1] & in_set)[0]
                        if len(v1_conf) != 1 or v1_conf[0] != u:
                            continue
                        # Check v2 only conflicts with u in the set
                        v2_conf = np.where(conflicts[v2] & in_set)[0]
                        if len(v2_conf) != 1 or v2_conf[0] != u:
                            continue
                        # Do the swap: remove u, add v1 and v2
                        in_set[u] = False
                        in_set[v1] = True
                        in_set[v2] = True
                        current_size += 1
                        conflict_count[neighbors[u]] -= 1
                        conflict_count[neighbors[v1]] += 1
                        conflict_count[neighbors[v2]] += 1
                        improved = True
                        in_set_indices = np.where(in_set)[0]
                        not_in_set = np.where(~in_set)[0]
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            no_improve = 0

            # Perturbation if stuck
            if not improved:
                no_improve += 1
                if no_improve > 20:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 3:
                        n_remove = min(4, len(in_set_indices))
                        to_remove = in_set_indices[rng.sample(len(in_set_indices), n_remove)]
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                        no_improve = 0
                    else:
                        break
            else:
                no_improve = 0

        return local_best_in_set, local_best_size

    # Multiple restarts
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    n_restarts = 0
    max_restarts = 2000

    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        seed = n_restarts * 7919 + 13

        in_set = greedy_min_degree(seed)
        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # Local search
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 1.0:
            break

        ls_time = min(remaining_time * 0.35, 2.5)
        if ls_time < 0.3:
            ls_time = 0.3

        ls_result, ls_size = local_search(in_set, seed, max_time=ls_time)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]

    return result_vectors

# EVOLVE-BLOCK-END
