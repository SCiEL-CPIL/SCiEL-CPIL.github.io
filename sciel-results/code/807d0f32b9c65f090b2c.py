import numpy as np
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280

    # Baseline: 1D Sidon-like set
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    best_set = [(x, 0) for x in xs]

    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = len(all_points)
    
    # Precompute full distance matrix
    all_pts64 = all_points.astype(np.int64)
    diff = all_pts64[:, np.newaxis, :] - all_pts64[np.newaxis, :, :]
    dist_matrix = np.sum(diff ** 2, axis=2).astype(np.int32)
    del diff
    
    # Optimized validity check using sets of distances
    # dist_sets[i] is a set of squared distances from point current_indices[i] to all other points in current_indices
    def is_valid_addition_optimized(p_idx, current_indices, dist_sets):
        """
        Check if point p_idx can be added to current set.
        current_indices: list of indices of points in the set.
        dist_sets: list of sets, dist_sets[i] contains distances from current_indices[i] to all other points in the set.
        """
        k = len(current_indices)
        if k < 2:
            return True
        
        # Get distances from p_idx to all current points
        d_p = dist_matrix[p_idx, current_indices]
        
        # Check for isosceles at p: any two distances equal
        # Use a set for O(k) check
        if len(set(d_p.tolist())) != len(d_p):
            return False
        
        # Check for isosceles at existing points
        # For each existing point i, check if d_p[i] is in dist_sets[i]
        # dist_sets[i] contains d(current_indices[i], current_indices[j]) for all j != i
        # If d_p[i] == d(current_indices[i], current_indices[j]), then triangle (p, current_indices[i], current_indices[j]) is isosceles at current_indices[i]
        for i in range(k):
            if d_p[i] in dist_sets[i]:
                return False
        
        return True

    def add_point(p_idx, current_indices, dist_sets):
        """Add point p_idx to the set and update dist_sets."""
        k = len(current_indices)
        # Get distances from p_idx to all current points
        d_p = dist_matrix[p_idx, current_indices]
        
        # Create new set for the new point p_idx
        new_set = set(d_p.tolist())
        
        # Update existing sets
        for i in range(k):
            dist_sets[i].add(d_p[i])
        
        # Add new point and its set
        current_indices.append(p_idx)
        dist_sets.append(new_set)

    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Use a smarter ordering
        if restart % 3 == 0:
            # Random
            perm = np.random.permutation(num_points)
        elif restart % 3 == 1:
            # Prioritize points with larger x+y or x*y (corners)
            scores = all_points[:, 0] * all_points[:, 1]
            perm = np.argsort(-scores) + np.random.randn(num_points) * 10
            perm = np.argsort(perm)
        else:
            # Prioritize points with larger max(x, N-1-x, y, N-1-y) (edges/corners)
            edge_score = np.maximum.reduce([
                all_points[:, 0].astype(float),
                (N - 1 - all_points[:, 0]).astype(float),
                all_points[:, 1].astype(float),
                (N - 1 - all_points[:, 1]).astype(float)
            ])
            perm = np.argsort(-edge_score) + np.random.randn(num_points) * 5
            perm = np.argsort(perm)
        
        current_indices = []
        dist_sets = []
        current_list = []

        for idx in perm:
            if time.time() > deadline:
                break
            if is_valid_addition_optimized(idx, current_indices, dist_sets):
                add_point(idx, current_indices, dist_sets)
                current_list.append((int(all_points[idx, 0]), int(all_points[idx, 1])))

        if len(current_list) > len(best_set):
            best_set = current_list[:]

    return best_set
# EVOLVE-BLOCK-END
