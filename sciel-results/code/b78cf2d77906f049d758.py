import time
import random
import numpy as np

N = 64
TOTAL = N * N


def construct():
    start_time = time.time()
    deadline = start_time + 280.0

    # Precompute all grid points
    xs = np.arange(N, dtype=np.int64)
    ys = np.arange(N, dtype=np.int64)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    all_points = np.column_stack([XX.ravel(), YY.ravel()])  # shape (4096, 2)

    # For each point i, compute squared distances to all other points
    # dists[i, j] = squared distance between point i and point j
    # We only need upper triangle for the conflict graph
    # Memory: 4096 * 4096 * 8 bytes = ~128 MB for int64, or use int32 (max dist ~ 2*63^2 = 7938)
    # Use int32 to save memory
    dists = np.zeros((TOTAL, TOTAL), dtype=np.int32)
    for i in range(TOTAL):
        dx = all_points[:, 0].astype(np.int64) - all_points[i, 0]
        dy = all_points[:, 1].astype(np.int64) - all_points[i, 1]
        d = dx * dx + dy * dy
        dists[i, :] = d.astype(np.int32)

    # Build conflict graph: two points i, j are in conflict if there exists
    # a third point k such that dist(i,k) == dist(j,k) (making isosceles with apex k)
    # OR dist(i,k) == dist(i,j) and dist(j,k) == dist(i,j) (making isosceles with base ij)
    # Actually, three points a, b, c form isosceles if any two of the three pairwise distances are equal.
    # So i and j conflict if:
    #   1. There exists k != i, j such that dist(i,k) == dist(j,k)  [isosceles with apex k]
    #   2. There exists k != i, j such that dist(i,k) == dist(i,j)  [isosceles with base from j to k, equal legs from i]
    #      which means dist(i,k) == dist(i,j), i.e., k is on the circle centered at i through j
    #   3. There exists k != i, j such that dist(j,k) == dist(i,j)  [isosceles with base from i to k, equal legs from j]
    #      which means dist(j,k) == dist(i,j), i.e., k is on the circle centered at j through i
    #
    # Actually condition 1 covers apex at k. Conditions 2 and 3 cover apex at i and j respectively.
    # So i,j conflict iff:
    #   - exists k: dist(i,k) == dist(j,k)  [apex at k]
    #   - exists k: dist(i,k) == dist(i,j)  [apex at i, base jk]  -- but this means k is equidistant... no.
    #
    # Let me reconsider. Three points a,b,c form isosceles triangle if:
    #   dist(a,b) == dist(a,c) OR dist(a,b) == dist(b,c) OR dist(a,c) == dist(b,c)
    #
    # So for pair (i,j), they conflict if there exists k != i,j such that:
    #   dist(i,j) == dist(i,k)  OR  dist(i,j) == dist(j,k)  OR  dist(i,k) == dist(j,k)
    #
    # This is a large conflict graph. With 4096 points, the conflict graph has ~16M edges potentially.
    # Building it explicitly would be expensive. Instead, let's use a greedy approach with fast checks.

    # Alternative: use a bitset-based approach where for each point we track which distances are "used"
    # by the current set. A new point p can be added iff:
    #   1. All distances from p to existing points are distinct
    #   2. For each existing point q, dist(p,q) is not equal to any dist(q, r) for existing r
    #
    # This is what the original does. Let's optimize it with numpy.

    best_set = []
    best_size = 0

    def check_and_add_fast(p_idx, in_set, points_arr, dist_matrix):
        """
        Check if point p_idx can be added to the set.
        in_set: boolean array of size TOTAL
        points_arr: k x 2 array of points in the set
        dist_matrix: k x k array of pairwise squared distances
        Returns (valid, new_points_arr, new_dist_matrix) or (False, None, None).
        """
        k = len(points_arr)
        p = all_points[p_idx]

        if k == 0:
            new_points = np.array([[p[0], p[1]]], dtype=np.int32)
            new_dist = np.zeros((1, 1), dtype=np.int32)
            return True, new_points, new_dist

        # Compute distances from p to all existing points
        dx = points_arr[:, 0].astype(np.int64) - p[0]
        dy = points_arr[:, 1].astype(np.int64) - p[1]
        dists_new = dx * dx + dy * dy  # shape (k,)

        # Check 1: all distances from p must be distinct
        if len(np.unique(dists_new)) != k:
            return False, None, None

        # Check 2: for each existing point i, dists_new[i] must not appear in
        # dist_matrix[i, j] for any j != i
        # Vectorized: for each i, check if dists_new[i] is in the row of dist_matrix
        # This is equivalent to checking if any (dist_matrix[i,:i] == dists_new[i]) or
        # (dist_matrix[i,i+1:] == dists_new[i])
        # We can vectorize this: create a boolean matrix where entry (i,j) is
        # True if dist_matrix[i,j] == dists_new[i]
        # But that's k x k comparisons. For k up to ~50, this is fine.

        for i in range(k):
            if i > 0:
                if (dist_matrix[i, :i] == dists_new[i]).any():
                    return False, None, None
            if i < k - 1:
                if (dist_matrix[i, i+1:] == dists_new[i]).any():
                    return False, None, None

        # Build new arrays
        new_points = np.vstack([points_arr, [[p[0], p[1]]]])
        new_dist = np.zeros((k+1, k+1), dtype=np.int32)
        new_dist[:k, :k] = dist_matrix
        new_dist[:k, k] = dists_new
        new_dist[k, :k] = dists_new
        return True, new_points, new_dist

    def greedy_with_order(order, time_limit):
        """Greedily add points in the given order."""
        points_arr = np.zeros((0, 2), dtype=np.int32)
        dist_matrix = np.zeros((0, 0), dtype=np.int32)
        in_set = np.zeros(TOTAL, dtype=bool)

        for idx in order:
            if time.time() > time_limit:
                break
            if in_set[idx]:
                continue
            p = all_points[idx]
            valid, new_pts, new_dm = check_and_add_fast(idx, in_set, points_arr, dist_matrix)
            if valid:
                points_arr = new_pts
                dist_matrix = new_dm
                in_set[idx] = True

        return points_arr, dist_matrix, in_set

    def swap_improve(points_arr, dist_matrix, in_set, time_limit):
        """Try to improve by removing one point and adding one or two others."""
        k = len(points_arr)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(points_arr)
            if k == 0:
                break

            # Try removing each point and adding up to 2 new ones
            for remove_idx in range(k):
                if time.time() > time_limit:
                    break

                remaining_idx = [i for i in range(k) if i != remove_idx]
                rem_points = points_arr[remaining_idx]
                rem_dm = dist_matrix[np.ix_(remaining_idx, remaining_idx)]

                # Find candidates not in set
                removed_pt = points_arr[remove_idx]
                removed_flat = int(removed_pt[0]) * N + int(removed_pt[1])
                in_set[removed_flat] = False

                candidates = np.where(~in_set)[0]
                random.shuffle(candidates)
                added = 0
                new_points = rem_points
                new_dm = rem_dm

                for c_idx in candidates:
                    if added >= 2:
                        break
                    valid, np2, ndm2 = check_and_add_fast(c_idx, in_set, new_points, new_dm)
                    if valid:
                        new_points = np2
                        new_dm = ndm2
                        in_set[c_idx] = True
                        added += 1

                if len(new_points) > k - 1:
                    if len(new_points) > k:
                        points_arr = new_points
                        dist_matrix = new_dm
                        improved = True
                        break
                    # Equal size: accept with some probability to escape local optima
                    elif random.random() < 0.3:
                        points_arr = new_points
                        dist_matrix = new_dm
                else:
                    # Restore removed point
                    in_set[removed_flat] = True

                # Reset in_set for consistency
                in_set[:] = False
                for pt in points_arr:
                    in_set[int(pt[0]) * N + int(pt[1])] = True

        return points_arr, dist_matrix, in_set

    num_restarts = 3000

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Generate diverse orderings
        if restart % 6 == 0:
            order = np.random.permutation(TOTAL)
        elif restart % 6 == 1:
            order = np.arange(TOTAL)
        elif restart % 6 == 2:
            order = np.arange(TOTAL).reshape(N, N).T.ravel()
        elif restart % 6 == 3:
            # Diagonal order
            order_list = []
            for d in range(2 * N - 1):
                for x in range(max(0, d - N + 1), min(N, d + 1)):
                    y = d - x
                    if 0 <= y < N:
                        order_list.append(x * N + y)
            order = np.array(order_list)
        elif restart % 6 == 4:
            # Spiral from center
            cx, cy = N // 2, N // 2
            order = np.argsort((all_points[:, 0] - cx)**2 + (all_points[:, 1] - cy)**2)
        else:
            # Random with spacing bias
            order = np.random.permutation(TOTAL)

        points_arr, dist_matrix, in_set = greedy_with_order(order, deadline)

        # Second greedy pass with different order
        if time.time() < deadline - 5:
            order2 = np.random.permutation(TOTAL)
            # Continue from current set
            for idx in order2:
                if time.time() > deadline - 5:
                    break
                if in_set[idx]:
                    continue
                valid, new_pts, new_dm = check_and_add_fast(idx, in_set, points_arr, dist_matrix)
                if valid:
                    points_arr = new_pts
                    dist_matrix = new_dm
                    in_set[idx] = True

        # Local search with swaps
        if time.time() < deadline - 10 and len(points_arr) >= 3:
            points_arr, dist_matrix, in_set = swap_improve(points_arr, dist_matrix, in_set, deadline - 5)

        if len(points_arr) > best_size:
            best_size = len(points_arr)
            best_set = [(int(pt[0]), int(pt[1])) for pt in points_arr]

    return best_set
