# EVOLVE-BLOCK-START
import numpy as np
import time

def _count_convex7_fast(points):
    """Fast count of convex 7-gons using vectorized operations."""
    n = len(points)
    # Sort points lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[order]
    
    total_count = 0
    
    for i in range(n):
        pivot = sorted_points[i]
        later_indices = np.arange(i + 1, n)
        if len(later_indices) < 6:
            continue
            
        later_points = sorted_points[later_indices]
        
        # Sort these points by angle around pivot
        angles = np.arctan2(later_points[:, 1] - pivot[1], later_points[:, 0] - pivot[0])
        angle_order = np.argsort(angles)
        sorted_later_points = later_points[angle_order]
        
        m = len(sorted_later_points)
        
        # Precompute cross products for all triples
        # Cross product (a,b,c) = (b-a) x (c-b) = (bx-ax)(cy-by) - (by-ay)(cx-bx)
        # We need T1[i,j]: turn(pivot, p_i, p_j)
        # T2[i,j,k]: turn(p_i, p_j, p_k)
        # T3[i,j]: turn(p_i, p_j, pivot)
        # T4[i,j]: turn(p_j, pivot, p_i)
        
        # Vectorized cross product computation
        # For T1: (pivot, p_i, p_j)
        # (p_i - pivot) x (p_j - p_i) = (p_ix - px)(p_jy - p_iy) - (p_iy - py)(p_jx - p_ix)
        pi = sorted_later_points[:, 0]
        pj = sorted_later_points[:, 1]
        
        # T1[i,j] = (pi[i]-px)*(pj[j]-pi[i]) - (pj[i]-py)*(pi[j]-pi[i]) > 1e-9
        # This is tricky to vectorize directly as a matrix because of the mixed indices.
        
        # Let's compute all cross products for triples (i,j,k) in the sorted_later_points
        # Cross(i,j,k) = (xj-xi)(yk-yj) - (yj-yi)(xk-xj)
        
        # We can compute a 3D array of cross products for all i,j,k
        # x = pi, y = pj
        # dx_ij = xj - xi
        # dy_ij = yj - yi
        # dx_jk = xk - xj
        # dy_jk = yk - yj
        # cross = dx_ij * dy_jk - dy_ij * dx_jk
        
        # This is m^3 which is 32^3 = 32768. Fine.
        
        x = pi
        y = pj
        
        # dx_ij[i,j] = x[j] - x[i]
        dx_ij = x[np.newaxis, :] - x[:, np.newaxis]
        dy_ij = y[np.newaxis, :] - y[:, np.newaxis]
        
        # dx_jk[j,k] = x[k] - x[j]
        dx_jk = x[np.newaxis, :] - x[:, np.newaxis]
        dy_jk = y[np.newaxis, :] - y[:, np.newaxis]
        
        # cross[i,j,k] = dx_ij[i,j] * dy_jk[j,k] - dy_ij[i,j] * dx_jk[j,k]
        # Need to align dimensions.
        # dx_ij is (m, m) with indices [i, j]
        # dy_jk is (m, m) with indices [j, k]
        # We want cross[i,j,k] = dx_ij[i,j] * dy_jk[j,k] - dy_ij[i,j] * dx_jk[j,k]
        
        # dx_ij[:, :, np.newaxis] is (m, m, 1)
        # dy_jk[np.newaxis, :, :] is (1, m, m)
        # Product is (m, m, m) with indices [i, j, k]
        
        cross = dx_ij[:, :, np.newaxis] * dy_jk[np.newaxis, :, :] - dy_ij[:, :, np.newaxis] * dx_jk[np.newaxis, :, :]
        
        # T2[i,j,k] = cross[i,j,k] > 1e-9
        T2 = cross > 1e-9
        
        # T1[i,j] = turn(pivot, p_i, p_j)
        # (p_i - pivot) x (p_j - p_i) = (x[i]-px)*(y[j]-y[i]) - (y[i]-py)*(x[j]-x[i])
        px, py = pivot
        T1 = (x[np.newaxis, :] - px) * (y[:, np.newaxis] - y[np.newaxis, :]) - (y[np.newaxis, :] - py) * (x[:, np.newaxis] - x[np.newaxis, :]) > 1e-9
        
        # T3[i,j] = turn(p_i, p_j, pivot)
        # (p_j - p_i) x (pivot - p_j) = (x[j]-x[i])*(py-y[j]) - (y[j]-y[i])*(px-x[j])
        T3 = (x[np.newaxis, :] - x[:, np.newaxis]) * (py - y[np.newaxis, :]) - (y[np.newaxis, :] - y[:, np.newaxis]) * (px - x[np.newaxis, :]) > 1e-9
        
        # T4[i,j] = turn(p_j, pivot, p_i)
        # (pivot - p_j) x (p_i - pivot) = (px-x[j])*(y[i]-py) - (py-y[j])*(x[i]-px)
        T4 = (px - x[np.newaxis, :]) * (y[:, np.newaxis] - py) - (py - y[np.newaxis, :]) * (x[:, np.newaxis] - px) > 1e-9
        
        # DP
        # dp[first, prev, curr]
        # l=2: dp2[first, first, curr] = T1[first, curr]
        
        dp = np.zeros((m, m, m), dtype=np.int64)
        # dp[a, a, b] = 1 if T1[a, b]
        # We can do this by setting dp[:, 0, :] but indices are tricky.
        # Let's just use a loop or advanced indexing.
        # Actually, we can create a matrix T1 and then set dp[i, i, j] = T1[i, j]
        # This is not directly vectorizable easily without a loop or einsum trick.
        # Given m is small, a loop is fine.
        for a in range(m):
            for b in range(m):
                if T1[a, b]:
                    dp[a, a, b] = 1
        
        # Iterate l=3 to 6
        for l in range(3, 7):
            # new_dp[first, prev, curr] = sum_pp dp[first, pp, prev] * T2[pp, prev, curr]
            new_dp = np.einsum('fpq,pqr->fqr', dp, T2, optimize=True)
            dp = new_dp
        
        # Check closing turns
        # valid_mask[first, prev, curr] = T3[prev, curr] * T4[first, curr]
        valid_mask = T3[np.newaxis, :, :] * T4[:, np.newaxis, :]
        count = np.sum(dp * valid_mask)
        total_count += count
        
    return total_count

def _is_valid_fast(points):
    """Fast validity check."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    
    # Distance check
    # Compute all pairwise squared distances
    # 33x33 matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Check off-diagonal
    mask = ~np.eye(33, dtype=bool)
    if np.min(dist_sq[mask]) < 1e-12:
        return False
        
    # Collinearity check
    # For each triple, compute area.
    # This is harder to vectorize fully, but we can loop over i.
    for i in range(33):
        p1 = points[i]
        # Vectors from p1 to all other points
        v1 = points - p1
        # Cross product of v1[j] and v1[k]
        # We need to check if any pair (j, k) has cross product close to 0.
        # Cross(j, k) = v1[j][0]*v1[k][1] - v1[j][1]*v1[k][0]
        # This is the determinant of the 2x2 matrix formed by v1[j] and v1[k].
        # We can compute the matrix of cross products for all j, k.
        cross = v1[:, 0][:, np.newaxis] * v1[:, 1][np.newaxis, :] - v1[:, 1][:, np.newaxis] * v1[:, 0][np.newaxis, :]
        # Check if any off-diagonal element is close to 0.
        # Note: j and k must be different from i and each other.
        # But v1[i] is 0, so cross[i, k] = 0. We should exclude i.
        mask = ~np.eye(33, dtype=bool)
        mask[i, :] = False
        mask[:, i] = False
        if np.min(np.abs(cross[mask])) < 1e-9:
            return False
            
    return True

def _generate_initial_points():
    """Generate initial points using a clustered strategy."""
    n = 33
    points = np.zeros((n, 2))
    
    # Cluster 1: 1 point at (0.1, 0.1)
    points[0] = [0.1, 0.1]
    
    # Cluster 2: 2 points near (0.9, 0.1)
    points[1] = [0.8, 0.1]
    points[2] = [0.9, 0.2]
    
    # Cluster 3: 4 points near (0.9, 0.9)
    points[3] = [0.8, 0.8]
    points[4] = [0.9, 0.8]
    points[5] = [0.9, 0.9]
    points[6] = [0.8, 0.9]
    
    # Cluster 4: 8 points near (0.1, 0.9)
    for i in range(8):
        t = i / 7.0
        points[7+i] = [0.1 + 0.1*t, 0.8 + 0.1*t]
        
    # Cluster 5: 16 points near (0.5, 0.0)
    for i in range(16):
        t = i / 15.0
        points[15+i] = [0.3 + 0.4*t, 0.05 + 0.05*np.sin(t*np.pi)]
        
    # Cluster 6: 12 points near (0.5, 1.0)
    for i in range(12):
        t = i / 11.0
        points[31-i] = [0.3 + 0.4*t, 0.95 - 0.05*np.sin(t*np.pi)]
        
    return points

def _hill_climb(points, time_limit):
    """Hill climbing optimization."""
    n = 33
    start_time = time.time()
    best_points = points.copy()
    best_count = _count_convex7_fast(points)
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = np.random.randint(n)
        old_point = points[idx].copy()
        
        # Move to a new random position
        new_point = np.random.rand(2)
        points[idx] = new_point
        
        if not _is_valid_fast(points):
            points[idx] = old_point
            continue
            
        new_count = _count_convex7_fast(points)
        
        if new_count < best_count:
            best_count = new_count
            best_points = points.copy()
            if best_count == 1:
                break
        else:
            points[idx] = old_point
            
    return best_points, best_count

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 600
    
    np.random.seed(42)
    
    # Generate initial points
    points = _generate_initial_points()
    
    if not _is_valid_fast(points):
        points = np.random.rand(n, 2)
    
    # Hill climbing
    points, _ = _hill_climb(points, deadline - time.time() - 10)
    
    return points

# EVOLVE-BLOCK-END
