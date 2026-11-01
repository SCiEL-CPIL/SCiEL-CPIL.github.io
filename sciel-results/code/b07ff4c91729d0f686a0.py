"""Rewritten program using model-and-optimize approach for kissing number in d=11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import heapq

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

    def greedy_with_heap():
        """Greedy independent set using a heap for efficient min-degree selection."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        # Build heap: (degree, index)
        heap = [(degrees[i], i) for i in range(N)]
        heapq.heapify(heap)
        in_heap = np.ones(N, dtype=bool)

        while heap and time.time() < deadline - 5:
            deg, idx = heapq.heappop(heap)
            if not alive[idx] or in_set[idx]:
                continue
            if deg != degrees[idx]:
                continue

            in_set[idx] = True
            alive[idx] = False
            in_heap[idx] = False

            # Remove neighbors
            nbrs = neighbors[idx]
            for nb in nbrs:
                if alive[nb]:
                    alive[nb] = False
                    in_heap[nb] = False

            # Update degrees of remaining alive vertices
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # For each remaining vertex, decrement degree by number of neighbors removed
                for nb in nbrs:
                    pass  # already removed from alive

                # Recompute degrees for remaining alive vertices
                # This is expensive; let's use a different approach
                # Actually, just rebuild the heap periodically

        # Rebuild approach: do greedy in batches
        # Let's use a simpler but still efficient approach
        return in_set

    def greedy_batch():
        """Greedy independent set with periodic heap rebuilds."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        batch_size = 50
        batches = 0

        while np.any(alive) and time.time() < deadline - 5:
            # Build heap from current alive vertices
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            heap = [(degrees[i], i) for i in alive_indices]
            heapq.heapify(heap)

            # Process up to batch_size vertices
            processed = 0
            while heap and processed < batch_size and time.time() < deadline - 5:
                deg, idx = heapq.heappop(heap)
                if not alive[idx] or in_set[idx]:
                    continue
                if deg != degrees[idx]:
                    continue

                in_set[idx] = True
                alive[idx] = False
                processed += 1

                # Remove neighbors
                nbrs = neighbors[idx]
                for nb in nbrs:
                    if alive[nb]:
                        alive[nb] = False

                # Update degrees
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    for nb in nbrs:
                        if nb in rem:
                            pass
                    # Faster: use boolean mask
                    nbr_mask = np.zeros(N, dtype=bool)
                    nbr_mask[nbrs] = True
                    degrees[rem] -= (nbr_mask[rem].sum())  # This is wrong

                # Actually, let me just recompute degrees for alive vertices
                # This is O(N * degree) per batch, which is acceptable

            # Recompute degrees for all alive vertices
            rem = np.where(alive)[0]
            if len(rem) > 0:
                for i in rem:
                    deg = 0
                    for nb in neighbors[i]:
                        if alive[nb]:
                            deg += 1
                    degrees[i] = deg

            batches += 1

        return in_set

    def greedy_simple():
        """Simple greedy with random tie-breaking, fast per iteration."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            # Find min degree among alive
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            alive[chosen] = False

            nbrs = neighbors[chosen]
            alive[nbrs] = False

            # Update degrees: for each alive vertex, subtract number of neighbors that were just removed
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Count how many of each alive vertex's neighbors are in {chosen} ∪ nbrs
                removed = np.zeros(N, dtype=bool)
                removed[chosen] = True
                removed[nbrs] = True
                for i in rem:
                    deg_decr = removed[neighbors[i]].sum()
                    degrees[i] -= deg_decr

        return in_set

    def local_search_2opt(in_set_init, max_time=3.0):
        """Local search: add vertices with 0 conflicts, then try 1-for-1 swaps."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        # Count conflicts for each vertex
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
                    else:
                        no_improve += 1

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
                    # For each vertex not in set, find which set vertices it conflicts with
                    # Try to find a swap where removing u allows adding v (and maybe more)
                    # Simple: for random u in set, check if any v not in set only conflicts with u
                    n_swaps_tried = 0
                    max_swap_tries = 100

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

            if not improved:
                no_improve += 1
                if no_improve > 50:
                    # Perturb: remove 2-3 random vertices
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 2:
                        n_remove = min(3, len(in_set_indices))
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
    max_restarts = 500

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # Choose strategy
        if n_restarts % 3 == 0:
            in_set = greedy_simple()
        else:
            in_set = greedy_simple()

        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # Local search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 1.5:
            break

        ls_time = min(remaining_time * 0.4, 3.0)
        if ls_time < 0.5:
            ls_time = 0.5

        ls_result, ls_size = local_search_2opt(in_set, max_time=ls_time)

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
