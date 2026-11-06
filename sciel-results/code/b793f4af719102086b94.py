import time
import random
import numpy as np

N = 64
TIME_LIMIT = 300.0


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 10.0

    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    total = N * N

    # Precompute squared distances: dist[i, j]
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = dx * dx + dy * dy

    def can_add_fast(idx, current_indices):
        """
        Vectorized check: can we add point idx to set current_indices without
        creating an isosceles triangle?
        Returns True if safe.
        """
        K = len(current_indices)
        if K < 2:
            return True

        # Distances from idx to all current points
        new_dists = dist[idx, current_indices]

        # Apex condition: all distances from idx to current points must be distinct
        if len(np.unique(new_dists)) != K:
            return False

        # Base/equal-edge conditions:
        # For each pair (i, j) in current, triangle (idx, i, j) is isosceles if:
        #   dist(idx,i) == dist(i,j)  OR  dist(idx,j) == dist(i,j)
        # We can vectorize: build sub_dist matrix and check
        sub_dist = dist[np.ix_(current_indices, current_indices)]

        # new_dists has shape (K,)
        # For pair (i,j): check new_dists[i] == sub_dist[i,j] or new_dists[j] == sub_dist[i,j]
        # Vectorize: create matrix where entry (i,j) is True if either condition holds
        # upper triangle indices
        i_idx, j_idx = np.triu_indices(K, k=1)
        d_i = new_dists[i_idx]
        d_j = new_dists[j_idx]
        d_ij = sub_dist[i_idx, j_idx]
        if np.any(d_i == d_ij) or np.any(d_j == d_ij):
            return False

        return True

    def greedy_random():
        """Greedily add points in random order using fast vectorized check."""
        current_indices = []
        remaining = list(range(total))
        random.shuffle(remaining)
        for idx in remaining:
            if time.time() > deadline:
                break
            if can_add_fast(idx, current_indices):
                current_indices.append(idx)
        return current_indices

    def local_search(initial_indices, time_budget):
        """
        Randomized swap-based local search with vectorized checks.
        """
        current = list(initial_indices)
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        end_time = time.time() + time_budget

        while time.time() < end_time:
            improved = False
            # Try a few swaps
            for _ in range(200):
                if time.time() > end_time:
                    break
                if not current:
                    break
                i = random.randrange(len(current))
                idx_remove = current[i]
                # Pick a random candidate not in current
                cand = random.randrange(total)
                if cand in current_set:
                    continue
                # Try swap
                temp_indices = [x for x in current if x != idx_remove]
                if can_add_fast(cand, temp_indices):
                    # Accept swap
                    current[i] = cand
                    current_set.discard(idx_remove)
                    current_set.add(cand)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
                    # After a swap, try to add more points greedily
                    remaining_pts = [j for j in range(total) if j not in current_set]
                    random.shuffle(remaining_pts)
                    added = 0
                    for add in remaining_pts:
                        if time.time() > end_time:
                            break
                        if added >= 10:
                            break
                        if can_add_fast(add, current):
                            current.append(add)
                            current_set.add(add)
                            added += 1
                            if len(current) > best_size:
                                best_size = len(current)
                                best = list(current)
            if not improved:
                # No improvement in this batch; break out if stuck
                break

        return best

    # Phase 1: Multiple greedy restarts
    best_indices = []
    best_size = 0
    n_restarts = 0
    while time.time() < deadline - 80:
        n_restarts += 1
        result = greedy_random()
        if len(result) > best_size:
            best_size = len(result)
            best_indices = result
        # Early exit if we hit a very good size
        if best_size > 200:
            break

    # Phase 2: Local search on best solution
    if best_size > 0 and time.time() < deadline - 20:
        remaining_time = max(10, deadline - 20 - time.time())
        improved_set = local_search(best_indices, remaining_time)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_indices = improved_set

    # Phase 3: Final greedy fill
    if time.time() < deadline - 5:
        current_set = set(best_indices)
        remaining_pts = [j for j in range(total) if j not in current_set]
        random.shuffle(remaining_pts)
        for idx in remaining_pts:
            if time.time() > deadline - 5:
                break
            if can_add_fast(idx, best_indices):
                best_indices.append(idx)
                current_set.add(idx)

    # Convert indices to (x, y) tuples
    result = [(int(xs[idx]), int(ys[idx])) for idx in best_indices]
    return result
