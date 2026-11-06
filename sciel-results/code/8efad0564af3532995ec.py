import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    def generate_lattice():
        best_points = None
        best_r = 0
        
        # Try a few different lattice scales
        for scale_factor in [0.9, 0.95, 1.0, 1.05, 1.1]:
            r_est = np.sqrt(0.9 / N) * scale_factor
            spacing = 2.0 * r_est
            
            k = int(1.0 / spacing) + 2
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        rows.append((x, y))
            
            points = np.array(rows)
            if len(points) < N:
                continue
                
            # Select N points closest to center
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')
            points = points[idx[:N]]
            
            # Calculate current radius
            d_min = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(points[i] - points[j])
                    if d < d_min:
                        d_min = d
                d_bound = 1.0 - np.linalg.norm(points[i])
                if d_bound < d_min:
                    d_min = d_bound
            
            r_cur = d_min / 2.0
            
            if r_cur > best_r:
                best_r = r_cur
                best_points = points.copy()
                
        return best_points, best_r

    points, r = generate_lattice()
    
    if points is None:
        # Fallback: Random points
        np.random.seed(42)
        points = np.random.rand(N, 2) * 2 - 1
        mask = np.hypot(points[:, 0], points[:, 1]) < 1.0
        while np.sum(mask) < N:
            new_pts = np.random.rand(N, 2) * 2 - 1
            new_mask = np.hypot(new_pts[:, 0], new_pts[:, 1]) < 1.0
            points = np.vstack([points, new_pts[new_mask]])
            mask = new_mask
        points = points[:N]
        d_min = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < d_min:
                    d_min = d
            d_bound = 1.0 - np.linalg.norm(points[i])
            if d_bound < d_min:
                d_min = d_bound
        r = d_min / 2.0

    best_r = r
    best_points = points.copy()
    
    def compute_r(pts):
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_bound_dist = np.min(1.0 - radii)
        
        return min(min_pair_dist, min_bound_dist) / 2.0

    def relax(pts, target_r, steps=50, lr=0.01):
        pts = pts.copy()
        for step in range(steps):
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion using KDTree for efficiency
            tree = cKDTree(pts)
            # We need to check neighbors within 2*target_r
            # Using query_ball_point or similar
            # For simplicity and robustness in this context, we stick to vectorized
            # full matrix but only if N is small enough, or use KDTree for neighbors.
            # Given N=300, full matrix is 90k, which is fast in NumPy.
            # Let's stick to the full matrix approach for correctness and simplicity,
            # as KDTree neighbor iteration can be tricky to vectorize forces for.
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            dists_safe = np.where(dists == 0, 1e-6, dists)
            unit_vecs = diff / dists_safe[:, :, np.newaxis]
            
            overlap = np.maximum(0, 2.0 * target_r - dists)
            
            forces += np.sum(overlap[:, :, np.newaxis] * unit_vecs, axis=1)
            
            # Boundary constraint
            radii = np.hypot(pts[:, 0], pts[:, 1])
            bound_overlap = np.maximum(0, target_r - (1.0 - radii))
            radii_safe = np.where(radii == 0, 1e-6, radii)
            unit_center = -pts / radii_safe[:, np.newaxis]
            forces += bound_overlap[:, np.newaxis] * unit_center
            
            # Update
            pts += lr * forces
            
            # Project back to inside unit circle
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            mask_out = new_radii > 1.0
            if np.any(mask_out):
                pts[mask_out] *= (1.0 / new_radii[mask_out])[:, np.newaxis]
                
        return pts

    # Iterative increase of r
    current_r = r
    step_r = current_r * 0.001
    max_iterations = 1000
    
    for _ in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        target_r = current_r + step_r
        
        new_pts = relax(best_points, target_r, steps=20, lr=0.05)
        new_r = compute_r(new_pts)
        
        if new_r >= target_r - 1e-6:
            current_r = target_r
            best_points = new_pts
            best_r = current_r
        else:
            step_r *= 0.8
            if step_r < current_r * 1e-5:
                break
                
    # Final polish
    if time.time() < deadline - 10.0:
        step_r = best_r * 0.0001
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            target_r = best_r + step_r
            new_pts = relax(best_points, target_r, steps=10, lr=0.01)
            new_r = compute_r(new_pts)
            if new_r >= target_r - 1e-7:
                best_r = target_r
                best_points = new_pts
            else:
                break

    return best_points
