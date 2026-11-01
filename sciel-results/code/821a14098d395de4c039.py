import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def compute_radius(pts):
        """Compute the common radius: min of half pairwise distances and boundary distances."""
        # Use scipy's pdist for efficient pairwise distances
        from scipy.spatial.distance import pdist
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    def hex_lattice_init(n, spacing=None):
        """Generate hexagonal lattice points inside the unit disk."""
        if spacing is None:
            spacing = 2.0 / (np.sqrt(n) + 1.5)
        
        while True:
            k = int(1.0 / spacing) + 3
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                # Select n closest to center
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.97

    def smooth_min(x, alpha=50.0):
        """Smooth approximation to min(x) using log-sum-exp."""
        x = np.asarray(x)
        x_clipped = np.clip(x, -100, 100)
        return -np.log(np.sum(np.exp(-alpha * x_clipped))) / alpha

    def neg_smooth_min(x_flat, alpha=50.0):
        """Objective: negative of smooth min of all constraints (to minimize)."""
        if time.time() > deadline:
            return 1e10
        
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        
        # Compute all pairwise distances efficiently
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        all_vals = np.concatenate([pair_vals, boundary])
        
        return -smooth_min(all_vals, alpha)

    def compute_gradient(x_flat, alpha=50.0):
        """Compute gradient of the smooth min objective."""
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        all_vals = np.concatenate([pair_vals, boundary])
        
        x_clipped = np.clip(all_vals, -100, 100)
        exp_vals = np.exp(-alpha * x_clipped)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(x_flat)
        
        # Pair contributions
        # For pair (i,j): d(dist_ij/2)/dx_i = (x_i - x_j) / (2 * dist_ij)
        # d(dist_ij/2)/dx_j = (x_j - x_i) / (2 * dist_ij)
        
        pair_count = len(pair_vals)
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dist[i, j]
            if d < 1e-10:
                continue
            w = weights[idx] * alpha  # d(-logsumexp)/dx = alpha * weight * dx
            # grad contribution from -smooth_min = +weights * alpha * d(constraint)/dx
            grad[2*i] += w * (pts[i, 0] - pts[j, 0]) / (2.0 * d)
            grad[2*i+1] += w * (pts[i, 1] - pts[j, 1]) / (2.0 * d)
            grad[2*j] += w * (pts[j, 0] - pts[i, 0]) / (2.0 * d)
            grad[2*j+1] += w * (pts[j, 1] - pts[i, 1]) / (2.0 * d)
        
        # Boundary contributions
        # d(1 - |c_i|)/dx_i = -x_i / |c_i|
        for i in range(n):
            r = r_pts[i]
            if r < 1e-10:
                continue
            w = weights[pair_count + i] * alpha
            grad[2*i] += w * (-pts[i, 0] / r)
            grad[2*i+1] += w * (-pts[i, 1] / r)
        
        return grad

    # Step 1: Generate initial hexagonal lattice
    pts = hex_lattice_init(N)
    
    # Step 2: Compute initial radius
    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Step 3: L-BFGS-B optimization with smooth surrogate
    from scipy.optimize import minimize
    
    x0 = pts.flatten()
    
    result = minimize(
        neg_smooth_min,
        x0,
        jac=compute_gradient,
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
    )
    
    if result.fun < neg_smooth_min(x0):
        pts_new = result.x.reshape(-1, 2)
        # Ensure points are inside the disk
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = r_new > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
        
        r_new_val = compute_radius(pts_new)
        if r_new_val > best_radius:
            best_radius = r_new_val
            best_pts = pts_new.copy()
    
    # Step 4: Local search with small perturbations
    if time.time() < deadline - 15:
        np.random.seed(42)
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            # Random small perturbation with decreasing magnitude
            scale = 0.002 * (1.0 - trial / 30.0) + 0.0005
            noise = np.random.normal(0, scale, size=best_pts.shape)
            pts_perturbed = best_pts + noise
            
            # Keep inside disk
            r_p = np.hypot(pts_perturbed[:, 0], pts_perturbed[:, 1])
            mask = r_p > 0.999
            if np.any(mask):
                pts_perturbed[mask] = pts_perturbed[mask] / r_p[mask, np.newaxis] * 0.999
            
            r_pert = compute_radius(pts_perturbed)
            if r_pert > best_radius:
                best_radius = r_pert
                best_pts = pts_perturbed.copy()
    
    # Step 5: Final polish with tighter optimization if time permits
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        x0 = best_pts.flatten()
        result2 = minimize(
            neg_smooth_min,
            x0,
            jac=compute_gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
        )
        
        if result2.fun < neg_smooth_min(x0):
            pts_new = result2.x.reshape(-1, 2)
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask = r_new > 1.0
            if np.any(mask):
                pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
            
            r_new_val = compute_radius(pts_new)
            if r_new_val > best_radius:
                best_radius = r_new_val
                best_pts = pts_new.copy()
    
    # Final validation: ensure all points are strictly inside and distinct
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask_out = r_final >= 1.0
    if np.any(mask_out):
        best_pts[mask_out] = best_pts[mask_out] / r_final[mask_out, np.newaxis] * 0.9999
    
    # Check for coincident points
    from scipy.spatial.distance import pdist
    min_dist = np.min(pdist(best_pts))
    if min_dist < 1e-6:
        # Add small random jitter to coincident points
        np.random.seed(123)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-6:
                    best_pts[j] += np.random.normal(0, 1e-4, size=2)
    
    return best_pts
