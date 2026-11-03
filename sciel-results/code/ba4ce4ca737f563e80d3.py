"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares


def _hex_init(n):
    """Generate a hexagonal lattice packing of n circles in the unit disk."""
    # Start with a reasonable spacing estimate
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    for _ in range(200):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) >= n:
            # Sort by distance from center, take the n closest
            d = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(d, kind='stable')[:n]
            return pts[idx]
        spacing *= 0.99
    # Fallback: simple grid
    m = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(-0.9, 0.9, m)
    ys = np.linspace(-0.9, 0.9, m)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    d = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(d, kind='stable')[:n]
    return pts[idx]


def _compute_radius(pts):
    """Compute the packing radius for a set of points."""
    n = len(pts)
    if n < 2:
        return 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])) if n == 1 else 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    # Distance to boundary
    boundary_dist = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    return 0.5 * min(min_pair_dist, np.min(boundary_dist))


def _residuals(x, n, t):
    """Residuals for least_squares: minimize negative distances."""
    pts = x.reshape(-1, 2)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Get pairwise distances for i < j
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    
    # Boundary distances
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Residuals: we want to maximize the minimum, so minimize (t - d) for d < t
    # Use a soft formulation: residuals = max(0, t - d) for each constraint
    pair_res = np.maximum(0.0, t - pair_dists)
    bnd_res = np.maximum(0.0, t - boundary_dists)
    
    return np.concatenate([pair_res, bnd_res])


def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    # Step 1: Hexagonal initialization
    pts = _hex_init(N)
    r_init = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = r_init
    
    # Step 2: Optimize using least_squares
    # We'll use an auxiliary variable t and minimize residuals
    # But least_squares doesn't easily handle auxiliary variables in the objective.
    # Instead, we'll directly maximize the minimum distance via a smooth surrogate.
    
    def _soft_min_logsumexp(vals, alpha):
        """Soft minimum using log-sum-exp."""
        # soft_min = -1/alpha * log(sum(exp(-alpha * vals)))
        m = np.min(vals)
        return m - np.log(np.sum(np.exp(-alpha * (vals - m)))) / alpha
    
    def _objective_and_grad(x):
        """Objective: negative soft-min of all distances, with gradient."""
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        iu = np.triu_indices(n, k=1)
        pair_dist_sq = dist_sq[iu]
        pair_dist = np.sqrt(pair_dist_sq)
        
        # Boundary distances
        r = np.hypot(pts[:, 0], pts[:, 1])
        bnd_dist = 1.0 - r
        
        # All distances
        all_dists = np.concatenate([pair_dist, bnd_dist])
        
        # Use soft-min with moderate alpha
        alpha = 50.0
        m = np.min(all_dists)
        # Soft min approximation
        soft_min = m - np.log(np.sum(np.exp(-alpha * (all_dists - m)))) / alpha
        
        # Objective: we want to maximize soft_min, so minimize -soft_min
        obj = -soft_min
        
        # Gradient computation (numerical for simplicity, but we can do analytical)
        # For now, use numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            pts_plus = x_plus.reshape(-1, 2)
            diff_p = pts_plus[:, np.newaxis, :] - pts_plus[np.newaxis, :, :]
            dist_sq_p = np.sum(diff_p ** 2, axis=2)
            np.fill_diagonal(dist_sq_p, np.inf)
            pair_dist_p = np.sqrt(dist_sq_p[iu])
            bnd_dist_p = 1.0 - np.hypot(pts_plus[:, 0], pts_plus[:, 1])
            all_dists_p = np.concatenate([pair_dist_p, bnd_dist_p])
            m_p = np.min(all_dists_p)
            soft_min_p = m_p - np.log(np.sum(np.exp(-alpha * (all_dists_p - m_p)))) / alpha
            grad[i] = (-soft_min_p - obj) / eps
        
        return obj, grad
    
    # Use a simpler approach: direct least_squares on the constraint residuals
    # with an iterative increase of t
    
    # Phase 1: Coarse optimization
    current_t = r_init * 2.0
    for phase in range(3):
        if time.time() > deadline - 30.0:
            break
        
        # Define residuals function for least_squares
        def _res(x):
            pts = x.reshape(-1, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            iu = np.triu_indices(N, k=1)
            pair_dists = dist[iu]
            boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
            
            # We want all distances >= current_t
            # Residuals: for distances < current_t, push them up
            pair_res = np.maximum(0.0, current_t - pair_dists)
            bnd_res = np.maximum(0.0, current_t - boundary_dists)
            
            # Add a small regularization to keep points spread
            return np.concatenate([pair_res, bnd_res])
        
        try:
            result = least_squares(_res, pts.ravel(), method='trf', 
                                   max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
                pts = new_pts
            current_t = max(current_t * 1.02, best_r * 2.0)
        except Exception:
            pass
    
    # Phase 2: Fine polish with very tight tolerances
    if time.time() < deadline - 10.0:
        # Use a sharper soft-min for final polish
        def _polish_residuals(x):
            pts = x.reshape(-1, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            iu = np.triu_indices(N, k=1)
            pair_dists = dist[iu]
            boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
            
            t = best_r * 1.001
            pair_res = np.maximum(0.0, t - pair_dists)
            bnd_res = np.maximum(0.0, t - boundary_dists)
            return np.concatenate([pair_res, bnd_res])
        
        try:
            result = least_squares(_polish_residuals, best_pts.ravel(), method='trf',
                                   max_nfev=200, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
        except Exception:
            pass
    
    # Final validation and return
    final_pts = best_pts
    # Ensure all points are inside the unit circle
    d = np.hypot(final_pts[:, 0], final_pts[:, 1])
    final_pts = final_pts[d <= 1.0]
    
    # If we lost some points, add them back near the center
    if len(final_pts) < N:
        missing = N - len(final_pts)
        extra = np.random.randn(missing, 2) * 0.01
        final_pts = np.vstack([final_pts, extra])
    
    return final_pts[:N]
# EVOLVE-BLOCK-END
