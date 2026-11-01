import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Step 1: Generate hexagonal lattice points
    def hex_lattice_init(n):
        # Try different spacings to get at least n points in unit disk
        best_points = None
        for trial in range(20):
            spacing = 2.0 / (np.sqrt(n) + 1.0) * (1.0 - trial * 0.02)
            if spacing <= 0:
                continue
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing / 2.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= n:
                # Select n closest to center
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists)[:n]
                if best_points is None or np.min(dists[idx]) < np.min(np.hypot(best_points[:, 0], best_points[:, 1])):
                    best_points = points[idx].copy()
        return best_points
    
    pts = hex_lattice_init(N)
    
    # Step 2: Compute initial radius
    def compute_radius(pts):
        # Min pair distance / 2
        from scipy.spatial.distance import pdist
        min_pair = np.min(pdist(pts)) / 2.0
        # Min distance to boundary
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)
    
    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Step 3: Repulsion-based optimization
    # Use a simple iterative method
    n_iter_max = 50000
    for it in range(n_iter_max):
        if time.time() > deadline:
            break
        
        # Compute forces
        # Repulsion between pairs
        # For efficiency, use vectorized computation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        dist = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(dist, 1.0)  # avoid division by zero
        
        # Force on each point: sum of (pts[i] - pts[j]) / dist^3
        # This is like Coulomb repulsion
        inv_dist3 = 1.0 / (dist ** 3)
        inv_dist3[np.diag_indices(N)] = 0.0
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)  # (N, 2)
        
        # Boundary force: push inward if too close to edge
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary_margin = 1.0 - r_pts
        # If boundary_margin is small, push inward
        # Force proportional to how close to boundary
        boundary_force = np.zeros_like(pts)
        for i in range(N):
            if r_pts[i] > 0:
                # Push toward center
                boundary_force[i] = -pts[i] / r_pts[i] * max(0, 0.1 - boundary_margin[i]) ** 2 * 10.0
        
        # Combined force
        total_force = forces + boundary_force
        
        # Adaptive step size
        step_size = 0.001 * (1.0 + it * 0.0001)
        # Limit step size
        force_mag = np.linalg.norm(total_force, axis=1)
        max_step = 0.05
        scale = np.minimum(step_size, max_step / (force_mag + 1e-10))
        pts_new = pts + total_force * scale[:, np.newaxis]
        
        # Keep points inside the disk
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = r_new > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.999
        
        # Check if this improves the radius
        r_new_val = compute_radius(pts_new)
        if r_new_val > best_radius:
            best_radius = r_new_val
            best_pts = pts_new.copy()
            pts = pts_new.copy()
        else:
            # Revert or take a smaller step
            pts = pts_new.copy()  # still accept for exploration
    
    # Final: return best
    return best_pts
