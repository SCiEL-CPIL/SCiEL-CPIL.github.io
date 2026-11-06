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
    
    # Precompute squared distances between all pairs of grid points
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
    
    # Precompute full distance matrix
    diff = all_pts64[:, np.newaxis, :] - all_pts64[np.newaxis, :, :]
    dist_matrix = np.sum(diff ** 2, axis=2).astype(np.int32)
    del diff
    
    def is_valid_addition_fast(p_idx, current_indices_list):
        """Check if point p_idx can be added to current set (given as list of indices)."""
        k = len(current_indices_list)
        if k < 2:
            return True
        
        # Convert to numpy array for consistent and fast indexing
        curr_idx_arr = np.array(current_indices_list, dtype=np.int64)
        
        # Get distances from p_idx to all current points
        d_p = dist_matrix[p_idx, curr_idx_arr]  # shape (k,)
        
        # Check for isosceles at p: any two distances equal
        # np.unique is efficient for small k
        if len(np.unique(d_p)) != len(d_p):
            return False
        
        # Check for isosceles at existing points: 
        # For each pair (i,j) in current set, check if d_p[i] == dist[i,j] or d_p[j] == dist[i,j]
        i_idx, j_idx = np.triu_indices(k, k=1)
        # Use the numpy array for fancy indexing
        d_ij = dist_matrix[curr_idx_arr[i_idx], curr_idx_arr[j_idx]]
        d_p_i = d_p[i_idx]
        d_p_j = d_p[j_idx]
        
        if np.any(d_p_i == d_ij) or np.any(d_p_j == d_ij):
            return False
        
        return True

    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Use a smarter ordering: start with random, but bias towards corners/edges
        # which tend to have fewer conflicts
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
        
        current_list = []
        current_indices = []

        for idx in perm:
            if time.time() > deadline:
                break
            if is_valid_addition_fast(idx, current_indices):
                current_list.append((int(all_points[idx, 0]), int(all_points[idx, 1])))
                current_indices.append(idx)

        if len(current_list) > len(best_set):
            best_set = current_list[:]

    return best_set
# EVOLVE-BLOCK-END
