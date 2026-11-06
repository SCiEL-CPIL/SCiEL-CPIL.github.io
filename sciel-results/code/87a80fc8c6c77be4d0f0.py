import random
import time
import numpy as np

N = 64
TOTAL_POINTS = N * N
DEADLINE = 320.0


def construct():
    start_time = time.time()
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = TOTAL_POINTS

    # Precompute squared distances using numpy for speed
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)

    # Precompute for each point, the list of (distance, count) pairs for all other points
    # Actually, for the validity check, we need to check if any distance appears 2+ times
    # among the selected points.

    def count_conflicts(selected):
        """
        selected: numpy array of indices
        Returns number of isosceles triangles.
        """
        if len(selected) < 3:
            return 0
        count = 0
        for i in range(len(selected)):
            p = selected[i]
            # Distances from p to all other selected points
            dists = dist_matrix[p, selected]
            # Exclude self-distance (which is 0)
            dists = dists[dists != 0]
            # Count frequencies
            # For small K, we can just use a dict or sort
            # But numpy unique with return_counts is fast
            if len(dists) > 1:
                _, counts = np.unique(dists, return_counts=True)
                for c in counts:
                    if c >= 2:
                        count += c * (c - 1) // 2
        return count

    def find_best_swap(selected):
        """
        Find a swap that reduces conflicts the most.
        selected: numpy array of indices
        Returns (index_to_remove, index_to_add, new_conflicts) or None
        """
        # Current conflicts
        current_conf = count_conflicts(selected)
        if current_conf == 0:
            return None

        best_delta = 0
        best_remove = -1
        best_add = -1

        # For each point in selected, try swapping with a random point not in selected
        # To save time, we only check a few random candidates per point
        selected_set = set(selected.tolist())
        num_candidates = min(20, num_points - len(selected))

        for i in range(len(selected)):
            p = selected[i]
            # Calculate contribution of p to conflicts
            # Conflicts involving p as vertex:
            dists_p = dist_matrix[p, selected]
            dists_p = dists_p[dists_p != 0]
            _, counts_p = np.unique(dists_p, return_counts=True)
            conf_p_vertex = 0
            for c in counts_p:
                if c >= 2:
                    conf_p_vertex += c * (c - 1) // 2

            # Conflicts where p is a base point (vertex is another point q)
            # This is harder to compute locally. We just estimate by total conflicts.
            # Actually, let's just try the swap and count.
            # But that's expensive.
            # Heuristic: points with many duplicate distances are "bad"
            badness = conf_p_vertex
            if badness == 0:
                continue

            # Try swapping p with random points
            for _ in range(num_candidates):
                q = random.randint(0, num_points - 1)
                if q in selected_set:
                    continue

                # Create new selected set
                new_selected = selected.copy()
                new_selected[i] = q
                # Check if q causes new conflicts
                # Just count conflicts in the new set? No, too slow.
                # Let's use the incremental approach from the previous code?
                # Actually, for a 320s budget, we can afford to do a few full checks
                # if the set is small (< 50).
                # For now, let's just do a full count for the new set if it's small enough
                # or use a fast check.
                # Let's just do the full count for simplicity, but limit the number of swaps tried.
                new_conf = count_conflicts(new_selected)
                delta = current_conf - new_conf
                if delta > best_delta:
                    best_delta = delta
                    best_remove = i
                    best_add = q
                    if best_delta > 5:
                        break
            if best_delta > 5:
                break

        if best_remove != -1:
            return best_remove, best_add, best_delta
        return None

    def add_point(selected):
        """
        Try to add a point to selected to get a valid set of size +1.
        Returns new selected array if successful, else None.
        """
        current_size = len(selected)
        if current_size >= num_points:
            return None

        # Try adding random points
        selected_set = set(selected.tolist())
        for _ in range(500):
            q = random.randint(0, num_points - 1)
            if q in selected_set:
                continue
            new_selected = np.append(selected, q)
            if count_conflicts(new_selected) == 0:
                return new_selected
        return None

    # Simulated Annealing style search
    best_set = np.array([], dtype=np.int32)
    current_set = np.array([], dtype=np.int32)

    # Initialize with a small valid set
    random_indices = np.random.permutation(num_points)
    for i in range(10):
        candidate = random_indices[i:i+5]
        if count_conflicts(candidate) == 0:
            current_set = candidate
            break

    if len(current_set) == 0:
        # Fallback: just pick random points until valid or timeout
        while time.time() < start_time + DEADLINE - 5:
            candidate = np.random.choice(num_points, 10, replace=False)
            if count_conflicts(candidate) == 0:
                current_set = candidate
                break

    best_set = current_set.copy()

    while time.time() < start_time + DEADLINE - 5:
        # Try to improve current_set
        swap_result = find_best_swap(current_set)
        if swap_result:
            idx_remove, idx_add, delta = swap_result
            if delta > 0:
                current_set[idx_remove] = idx_add
                if count_conflicts(current_set) < count_conflicts(best_set):
                    best_set = current_set.copy()
        else:
            # No improving swap found, try to add a point
            new_set = add_point(current_set)
            if new_set is not None:
                current_set = new_set
                best_set = current_set.copy()
            else:
                # Perturb current_set
                # Swap a random point with a random other point
                if len(current_set) > 1:
                    i = random.randint(0, len(current_set) - 1)
                    j = random.randint(0, len(current_set) - 1)
                    if i != j:
                        current_set[i], current_set[j] = current_set[j], current_set[i]
                    else:
                        q = random.randint(0, num_points - 1)
                        if q not in current_set:
                            current_set[i] = q

    return [points[i] for i in best_set]
