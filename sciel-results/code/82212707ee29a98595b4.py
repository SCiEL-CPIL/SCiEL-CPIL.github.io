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

    # Precompute all squared distances between all pairs of grid points
    # dist[i, j] = squared distance between point i and point j
    # For memory efficiency with N=64 (4096 points), we use int32
    all_pts64 = all_points.astype(np.int64)
    # Compute pairwise squared distances in chunks to manage memory
    # Actually 4096x4096 int32 is ~64MB, manageable
    # But let's be smart: we only need to check if adding point p creates an isosceles triangle
    # with any two existing points. So we need:
    # For new point p and existing points q1, q2:
    #   dist(p,q1) == dist(p,q2)  -> isosceles at p
    #   dist(p,q1) == dist(q1,q2) -> isosceles at q1
    #   dist(p,q2) == dist(q1,q2) -> isosceles at q2

    def is_valid_addition_fast(p_idx, pts_indices, dist_matrix):
        """Check if adding point p_idx to the set of points pts_indices creates an isosceles triangle.
        pts_indices: list/array of indices into all_points
        dist_matrix: precomputed pairwise squared distance matrix
        """
        k = len(pts_indices)
        if k < 2:
            return True
        
        # Get distances from p to all existing points
        d_p = dist_matrix[p_idx, pts_indices]  # shape (k,)
        
        # Check: any two distances from p equal? (isosceles at p)
        # Use sorting trick: if any duplicates in d_p, then invalid
        if k >= 2:
            d_sorted = np.sort(d_p)
            if np.any(d_sorted[:-1] == d_sorted[1:]):
                return False
        
        # Check: for any pair (i,j) in existing points, 
        #   dist(p, pi) == dist(pi, pj) or dist(p, pj) == dist(pi, pj)
        # This means: for each existing pair, the distance between them must not equal
        # either of the distances from p to those points.
        # 
        # More efficient: collect all pairwise distances among existing points,
        # and check if any equals any d_p value.
        
        # Get all pairwise distances among existing points
        pts_arr = np.array(pts_indices, dtype=np.int32)
        i_idx, j_idx = np.triu_indices(k, k=1)
        d_ij = dist_matrix[pts_arr[i_idx], pts_arr[j_idx]]  # shape (k*(k-1)/2,)
        
        # Check if any d_p value appears in d_ij
        if len(d_ij) > 0:
            # Use set intersection for speed
            d_p_set = set(d_p.tolist())
            d_ij_set = set(d_ij.tolist())
            if d_p_set & d_ij_set:
                return False
        
        return True

    # Precompute full distance matrix
    # dist[i,j] = (xi-xj)^2 + (yi-yj)^2
    diffs = all_pts64[:, np.newaxis, :] - all_pts64[np.newaxis, :, :]
    dist_matrix = np.sum(diffs ** 2, axis=2).astype(np.int32)
    del diffs, all_pts64

    # Heuristic ordering: prefer points that are "far" from center or have unique distance profiles
    # Better: use a greedy approach where we pick the point with fewest conflicts first
    # For each point, compute how many other points share a distance with it
    # Points with more unique distances are better candidates to add early

    # Simpler heuristic: random but with some structure
    # Try multiple random seeds and orderings

    num_restarts = 100
    best_size = len(best_set)

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Different ordering strategies per restart
        if restart % 3 == 0:
            perm = np.random.permutation(num_points)
        elif restart % 3 == 1:
            # Sort by x+y (diagonal order)
            sums = all_points[:, 0] + all_points[:, 1]
            perm = np.argsort(sums + np.random.rand(num_points) * 0.1)
        else:
            # Sort by distance from center
            center = np.array([N/2, N/2])
            dists_from_center = np.sum((all_points.astype(np.float64) - center) ** 2, axis=1)
            perm = np.argsort(dists_from_center + np.random.rand(num_points) * N)

        current_indices = []
        current_arr = np.empty((0, 2), dtype=np.int32)

        for idx in perm:
            if time.time() > deadline:
                break
            if is_valid_addition_fast(int(idx), current_indices, dist_matrix):
                current_indices.append(int(idx))
                current_arr = np.vstack([current_arr, all_points[idx]])

        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_set = [(int(all_points[i, 0]), int(all_points[i, 1])) for i in current_indices]

    # If we have time, do a local improvement pass on the best set
    if time.time() < deadline - 20 and best_size > 0:
        # Try to add more points to the best set
        best_indices = [i for i in range(num_points) 
                       if (all_points[i, 0], all_points[i, 1]) in set(best_set)]
        
        # Try adding each non-selected point
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for i in range(num_points):
                if time.time() > deadline - 5:
                    break
                if i in best_indices:
                    continue
                if is_valid_addition_fast(i, best_indices, dist_matrix):
                    best_indices.append(i)
                    best_set.append((int(all_points[i, 0]), int(all_points[i, 1])))
                    added = True

    return best_set
# EVOLVE-BLOCK-END
