"""Kissing number in dimension 11 - improved greedy + local search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import heapq
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all candidate vectors with norm^2 = 4
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
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists as sorted arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_min_degree(seed=None):
        """Greedy independent set with min-degree heuristic and random tie-breaking."""
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng

        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 3:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[local_rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Vectorized degree update
                removed_mask = np.zeros(N, dtype=bool)
                removed_mask[chosen] = True
                removed_mask[nbrs] = True
                for i in rem:
                    deg_decr = removed_mask[neighbors[i]].sum()
                    degrees[i] -= deg_decr

        return in_set

    def local_search(in_set_init, max_time=3.0):
        """Enhanced local search: add zero-conflict vertices, 1-for-1 swaps, 2-for-1 swaps."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        # Count conflicts for each vertex (how many in-set neighbors)
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
                    v = zero_conf[rng.integers(len(zero_conf))]
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
                    n_swaps_tried = 0
                    max_swap_tries = 200

                    while n_swaps_tried < max_swap_tries and time.time() < ls_deadline:
                        n_swaps_tried += 1
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        # Vertices not in set that conflict with u
                        v_candidates = np.where(conflicts[u] & ~in_set)[0]
                        if len(v_candidates) == 0:
                            continue

                        v = v_candidates[rng.integers(len(v_candidates))]
                        # Check if v conflicts with any other set vertex
                        v_conflicts_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(v_conflicts_in_set) == 1 and v_conflicts_in_set[0] == u:
                            # Do the swap
                            in_set[u] = False
                            in_set[v] = True
                            conflict_count[neighbors[u]] -= 1
                            conflict_count[neighbors[v]] += 1
                            improved = True
                            in_set_indices = np.where(in_set)[0]
                            not_in_set = np.where(~in_set)[0]

                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()

            # Phase 3: Try 2-for-1 swaps (remove 1, add 2)
            if time.time() < ls_deadline and rng.random() < 0.3:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]

                if len(in_set_indices) > 0 and len(not_in_set) > 1:
                    n_2for1_tried = 0
                    max_2for1_tries = 50

                    while n_2for1_tried < max_2for1_tries and time.time() < ls_deadline:
                        n_2for1_tried += 1
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        # Find vertices not in set that conflict with u
                        v_candidates = np.where(conflicts[u] & ~in_set)[0]
                        if len(v_candidates) < 2:
                            continue

                        # Pick two random candidates
                        idx1 = rng.integers(len(v_candidates))
                        idx2 = rng.integers(len(v_candidates))
                        if idx1 == idx2:
                            continue
                        v1 = v_candidates[idx1]
                        v2 = v_candidates[idx2]

                        # Check if v1 and v2 only conflict with u in the set
                        v1_conflicts_in_set = np.where(conflicts[v1] & in_set)[0]
                        v2_conflicts_in_set = np.where(conflicts[v2] & in_set)[0]
                        # Also check v1 and v2 don't conflict with each other
                        if (len(v1_conflicts_in_set) == 1 and v1_conflicts_in_set[0] == u and
                            len(v2_conflicts_in_set) == 1 and v2_conflicts_in_set[0] == u and
                            not conflicts[v1, v2]):
                            # Do the 2-for-1 swap
                            in_set[u] = False
                            in_set[v1] = True
                            in_set[v2] = True
                            conflict_count[neighbors[u]] -= 1
                            conflict_count[neighbors[v1]] += 1
                            conflict_count[neighbors[v2]] += 1
                            current_size += 1
                            improved = True
                            in_set_indices = np.where(in_set)[0]
                            not_in_set = np.where(~in_set)[0]

                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()

            if not improved:
                no_improve += 1
                if no_improve > 30:
                    # Perturb: remove 2-4 random vertices
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 2:
                        n_remove = min(4, len(in_set_indices))
                        to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
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

    # Multiple restarts with different strategies
    n_restarts = 0
    max_restarts = 1000

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # Use different random seeds for diversity
        seed = n_restarts * 7919 + 42
        in_set = greedy_min_degree(seed=seed)

        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # Local search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 1.0:
            break

        ls_time = min(remaining_time * 0.35, 4.0)
        if ls_time < 0.5:
            ls_time = 0.5

        ls_result, ls_size = local_search(in_set, max_time=ls_time)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

        if time.time() > deadline - 10:
            break

    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]

    return result_vectors

# EVOLVE-BLOCK-END
