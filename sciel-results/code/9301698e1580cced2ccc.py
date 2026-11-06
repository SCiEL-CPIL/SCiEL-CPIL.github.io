"""Rewritten program: minimal maximal no-isosceles set via randomized greedy + local search."""
import time
import random
import numpy as np

N = 16

# Deadline guard: keep a safety margin before the 300s limit
_DEADLINE = 300.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _dist_sq_vec(points, p):
    """
    Compute squared distances from point p to all points in array points.
    points: (m, 2) array, p: (2,) array
    returns: (m,) array of squared distances
    """
    diff = points - p
    return np.sum(diff * diff, axis=1)


def _is_isosceles_free(p, chosen_pts):
    """
    Check if adding point p to chosen_pts creates any isosceles triangle.
    chosen_pts: (m, 2) array of already chosen points.
    p: (2,) array.
    Returns True if p can be added (no isosceles triangle formed).
    """
    m = len(chosen_pts)
    if m < 2:
        return True
    
    # Distances from p to each chosen point
    d_p = _dist_sq_vec(chosen_pts, p)  # (m,)
    
    # For each pair (i, j) in chosen, check if d_p[i] == dist(i,j) or d_p[j] == dist(i,j) or d_p[i] == d_p[j]
    # d_p[i] == d_p[j]: p is equidistant from i and j -> isosceles with apex at p
    # d_p[i] == dist(i,j): triangle (p, i, j) has side p-i equal to i-j -> isosceles
    # d_p[j] == dist(i,j): triangle (p, i, j) has side p-j equal to i-j -> isosceles
    
    # Precompute all pairwise squared distances among chosen points
    # dist[i,j] for i < j
    # We need to check for each new p:
    # 1) For any pair (i,j): d_p[i] == d_p[j] (p equidistant from i and j)
    # 2) For any pair (i,j): d_p[i] == dist_sq(i,j)
    # 3) For any pair (i,j): d_p[j] == dist_sq(i,j)
    
    # Efficient check:
    # For condition 1: any duplicate values in d_p means isosceles. But wait, that's only if there exist i,j with d_p[i]==d_p[j].
    # Actually condition 1 is: exists i != j such that d_p[i] == d_p[j]. This means p is on the perpendicular bisector of some pair.
    
    # Let's check condition 1 first (fastest):
    # If any two distances from p to chosen points are equal, we have an isosceles triangle (apex at p).
    # Use a set to check for duplicates
    if len(set(d_p.tolist())) != m:
        return False
    
    # Conditions 2 and 3: for each pair (i,j), check if d_p[i] == dist(i,j) or d_p[j] == dist(i,j)
    # Precompute pairwise distances
    # For efficiency with small m, just loop
    for i in range(m):
        for j in range(i + 1, m):
            dij = _dist_sq_vec(chosen_pts[i:j+1], chosen_pts[i])[0]  # dist between i and j
            # Actually let's compute directly
            dij = (chosen_pts[i][0] - chosen_pts[j][0])**2 + (chosen_pts[i][1] - chosen_pts[j][1])**2
            if d_p[i] == dij or d_p[j] == dij:
                return False
    return True


def _is_isosceles_free_fast(p, chosen_pts, dist_matrix):
    """
    Fast check using precomputed distance matrix among chosen points.
    p: (2,) array
    chosen_pts: (m, 2) array
    dist_matrix: (m, m) symmetric matrix of squared distances, dist_matrix[i,j] = dist_sq(i,j)
    """
    m = len(chosen_pts)
    if m < 2:
        return True
    
    d_p = _dist_sq_vec(chosen_pts, p)  # (m,)
    
    # Condition 1: any two distances from p equal
    if len(set(d_p.tolist())) != m:
        return False
    
    # Conditions 2 and 3: for each pair (i,j), d_p[i] == dist(i,j) or d_p[j] == dist(i,j)
    # dist_matrix[i,j] is the squared distance between chosen_pts[i] and chosen_pts[j]
    # For each i, check if d_p[i] appears in dist_matrix[i, :] (excluding self, but dist to self is 0)
    # Actually: for pair (i,j), we need d_p[i] == dist_matrix[i,j] OR d_p[j] == dist_matrix[i,j]
    # Equivalent: for each i, for each j != i, check d_p[i] == dist_matrix[i,j]
    # We can vectorize: for each i, check if d_p[i] is in the set of dist_matrix[i, :]
    
    for i in range(m):
        # Check if d_p[i] equals any dist_matrix[i, j] for j != i
        # dist_matrix[i, i] = 0, d_p[i] is distance from p to chosen[i], could be 0 if p == chosen[i] but p is new
        mask = np.ones(m, dtype=bool)
        mask[i] = False
        if np.any(d_p[i] == dist_matrix[i, mask]):
            return False
    
    return True


def _greedy_randomized(order, max_time=None):
    """
    Greedy construction in given order.
    order: list of grid point indices (0 to N*N-1)
    Returns list of chosen points as (x, y) tuples.
    """
    if max_time is None:
        max_time = _remaining()
    
    # Map index to (x, y)
    def idx_to_pt(idx):
        return (idx % N, idx // N)
    
    chosen = []
    chosen_arr = np.empty((0, 2), dtype=np.int64)
    dist_matrix = np.empty((0, 0), dtype=np.int64)
    
    for idx in order:
        if _remaining() < max_time:
            break
        p = np.array(idx_to_pt(idx), dtype=np.int64)
        
        m = len(chosen)
        if m < 2:
            # No isosceles possible with fewer than 2 points
            chosen.append(tuple(p))
            if m == 0:
                chosen_arr = p.reshape(1, 2)
            else:
                # m == 1
                d = (chosen_arr[0][0] - p[0])**2 + (chosen_arr[0][1] - p[1])**2
                dist_matrix = np.array([[0, d], [d, 0]], dtype=np.int64)
                chosen_arr = np.vstack([chosen_arr, p])
            continue
        
        # Check if p is isosceles-free
        if _is_isosceles_free_fast(p, chosen_arr, dist_matrix):
            # Add p
            chosen.append(tuple(p))
            # Update dist_matrix
            d_new = _dist_sq_vec(chosen_arr, p)  # (m,)
            new_row = np.zeros(m + 1, dtype=np.int64)
            new_row[:m] = d_new
            new_row[m] = 0
            dist_matrix = np.vstack([dist_matrix, new_row])
            dist_matrix = np.hstack([dist_matrix, new_row.reshape(-1, 1)])
            chosen_arr = np.vstack([chosen_arr, p])
    
    return chosen


def _try_remove_one(chosen_pts, dist_matrix):
    """
    Try to remove one point from the set to see if we can get a smaller valid set.
    This is not directly useful for maximality, but for local search on the set size.
    Actually, for a maximal no-isosceles set, we want to minimize size while maintaining maximality.
    Removing a point might break maximality (other points could be added).
    This is complex. Skip for now.
    """
    pass


def construct():
    """
    Find a minimal maximal no-isosceles set.
    Returns list of (x, y) tuples.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Precompute all grid points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    all_indices = list(range(N * N))
    
    best = None
    best_len = N * N + 1
    
    # Run multiple randomized greedy attempts
    rng = random.Random(42)
    
    # First, try a deterministic greedy in row-major order as baseline
    baseline_order = list(range(N * N))
    result = _greedy_randomized(baseline_order)
    best = result
    best_len = len(result)
    
    # Now try randomized orders
    attempts = 0
    while _remaining() > 10.0:
        attempts += 1
        # Random shuffle
        order = all_indices[:]
        rng.shuffle(order)
        
        result = _greedy_randomized(order)
        if len(result) < best_len:
            best = result
            best_len = len(result)
            # Reset start time? No, keep global deadline
            # We could try to do local improvement here but let's keep it simple
            # Try a few more shuffles from this good result?
            # For now, just continue searching
        
        # Periodically check time
        if attempts % 100 == 0 and _remaining() < 30.0:
            break
    
    return best


if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(result)
