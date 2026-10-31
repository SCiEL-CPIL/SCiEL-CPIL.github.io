import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Precompute indices for pairwise distances
    # We fix p0 and p1. The variable points are p2 ... p15 (indices 2 to 15).
    # Total points N=16. Variables are 2*(N-2) = 28.
    
    def get_dists_sq(var_pts):
        """
        Compute squared distances for all pairs.
        var_pts: array of shape (N-2, 2) representing points 2..15.
        Returns array of squared distances for all 120 pairs.
        """
        # Reconstruct full points
        # p0 = (0,0), p1 = (1,0)
        full_pts = np.vstack((
            np.array([[0.0, 0.0]]),
            np.array([[1.0, 0.0]]),
            var_pts
        ))
        
        # Compute pairwise squared distances
        diff = full_pts[:, np.newaxis, :] - full_pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Get upper triangle indices
        iu = np.triu_indices(N, 1)
        return dists_sq[iu]

    def smooth_objective(var_pts, beta=20.0):
        """
        Smooth surrogate for (max_dist / min_dist)^2.
        We minimize log(max_d) - log(min_d).
        Using log-sum-exp for stability.
        """
        d_sq = get_dists_sq(var_pts)
        
        # Check for very small distances (collision)
        if np.any(d_sq < 1e-12):
            return 1e10
            
        d = np.sqrt(d_sq)
        log_d = np.log(d)
        
        # Soft max of log_d
        m_max = np.max(log_d)
        log_sum_max = np.log(np.sum(np.exp(beta * (log_d - m_max))))
        soft_max_log_d = m_max + (1.0 / beta) * log_sum_max
        
        # Soft min of log_d
        m_min = np.min(log_d)
        log_sum_min = np.log(np.sum(np.exp(-beta * (log_d - m_min))))
        soft_min_log_d = m_min - (1.0 / beta) * log_sum_min
        
        # Difference in log space is log(ratio)
        log_ratio = soft_max_log_d - soft_min_log_d
        
        # We want to minimize (ratio)^2, which is exp(2 * log_ratio).
        # Minimizing log_ratio is equivalent and numerically better.
        # However, the task asks for the raw value score.
        # For optimization, minimizing log_ratio is smooth and monotonic.
        return log_ratio

    def true_objective(var_pts):
        """
        Exact objective for evaluation.
        """
        d_sq = get_dists_sq(var_pts)
        if np.any(d_sq < 1e-12):
            return 1e10
        d = np.sqrt(d_sq)
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def make_hex_init():
        """Generate a hexagonal lattice cluster."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        
        # Sort by distance to origin to get compact cluster
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        pts = pts[order[:N]]
        
        # Scale and align so that the closest pair is at distance 1?
        # We fix p0=(0,0), p1=(1,0).
        # We need to map the cluster such that two points are at these positions.
        # Find the closest pair in the generated cluster
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        min_idx = np.argmin(dists_sq)
        i, j = np.unravel_index(min_idx, (N, N))
        
        p_i = pts[i]
        p_j = pts[j]
        
        # Vector from i to j
        v = p_j - p_i
        dist = np.linalg.norm(v)
        
        # Rotation matrix to align v with (1, 0)
        # v / dist is the unit vector.
        # We want to rotate the whole set so that p_i -> (0,0) and p_j -> (1,0).
        # 1. Translate so p_i is at origin
        pts_trans = pts - p_i
        # 2. Rotate so that (p_j - p_i) is along x-axis
        angle = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle), np.cos(-angle)]])
        pts_rot = (R @ pts_trans.T).T
        # 3. Scale so that distance is 1
        pts_scaled = pts_rot / dist
        
        # Now pts_scaled[i] should be (0,0) and pts_scaled[j] should be (1,0)
        # We need to reorder points so that index 0 is i and index 1 is j.
        # This is complex. Instead, we just use this as an initial guess for the variables.
        # But our variable space assumes p0 and p1 are fixed at (0,0) and (1,0).
        # So we must identify which two points in the optimized cluster correspond to the fixed ones.
        # It's easier to just generate a random cluster, find the closest pair, and map it.
        
        # Re-order the array so that the closest pair is at indices 0 and 1.
        # Create a permutation
        perm = list(range(N))
        perm[0] = i
        perm[1] = j
        # Ensure i != j (guaranteed)
        # Remove i and j from the rest of the permutation list to avoid duplicates
        rest = [k for k in range(N) if k != i and k != j]
        perm = [i, j] + rest
        
        pts_permuted = pts_scaled[perm]
        
        # The variables are points 2 to 15
        var_pts = pts_permuted[2:]
        return var_pts

    def make_square_init():
        """Generate a square grid cluster."""
        side = 4
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append([i, j])
        pts = np.array(pts[:N], dtype=float)
        pts -= pts.mean(axis=0)
        
        # Find closest pair
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_idx = np.argmin(dists_sq)
        i, j = np.unravel_index(min_idx, (N, N))
        
        p_i = pts[i]
        p_j = pts[j]
        v = p_j - p_i
        dist = np.linalg.norm(v)
        angle = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle), np.cos(-angle)]])
        pts_trans = pts - p_i
        pts_rot = (R @ pts_trans.T).T
        pts_scaled = pts_rot / dist
        
        perm = [i, j] + [k for k in range(N) if k != i and k != j]
        pts_permuted = pts_scaled[perm]
        var_pts = pts_permuted[2:]
        return var_pts

    def make_random_init():
        """Generate a random cluster."""
        pts = np.random.randn(N, D)
        # Find closest pair
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_idx = np.argmin(dists_sq)
        i, j = np.unravel_index(min_idx, (N, N))
        
        p_i = pts[i]
        p_j = pts[j]
        v = p_j - p_i
        dist = np.linalg.norm(v)
        if dist < 1e-6:
            return np.random.randn(N-2, D)
        angle = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle), np.cos(-angle)]])
        pts_trans = pts - p_i
        pts_rot = (R @ pts_trans.T).T
        pts_scaled = pts_rot / dist
        
        perm = [i, j] + [k for k in range(N) if k != i and k != j]
        pts_permuted = pts_scaled[perm]
        var_pts = pts_permuted[2:]
        return var_pts

    # Initial candidates
    inits = [
        make_hex_init(),
        make_square_init(),
        make_random_init(),
        make_random_init(),
        make_random_init()
    ]

    best_val = np.inf
    best_var_pts = inits[0].copy()

    # Optimization loop
    # We use L-BFGS-B with a smooth objective
    # Beta controls the sharpness of the soft-max/min.
    # Start with lower beta for global basin, increase for refinement.
    
    for beta in [5.0, 10.0, 20.0, 40.0]:
        if time.time() > deadline - 10:
            break
            
        for init in inits:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    lambda x: smooth_objective(x, beta=beta),
                    init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                # Evaluate true objective
                val = true_objective(res.x)
                if val < best_val:
                    best_val = val
                    best_var_pts = res.x.copy()
                    
            except Exception as e:
                pass

    # Final polish with very high beta (sharp) and more iterations
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: smooth_objective(x, beta=100.0),
                best_var_pts,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            val = true_objective(res.x)
            if val < best_val:
                best_val = val
                best_var_pts = res.x.copy()
        except Exception:
            pass

    # Reconstruct full points for return
    # p0 = (0,0), p1 = (1,0)
    full_pts = np.vstack((
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 0.0]]),
        best_var_pts
    ))
    
    return full_pts
