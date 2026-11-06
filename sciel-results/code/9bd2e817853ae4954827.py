"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Golden angle spiral initialization (sunflower model)
    def _golden_spiral(n):
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        angles = np.arange(n) * golden_angle
        radii = np.sqrt(np.arange(n) / n)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y])
    
    points = _golden_spiral(N)
    
    def _compute_min_radius(pts):
        if len(pts) == 0:
            return 0.0
        # Boundary constraint
        dists = np.sqrt(np.sum(pts**2, axis=1))
        boundary_r = np.min(1.0 - dists)
        if boundary_r < 0:
            return 0.0
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        pair_r = min_pair_dist / 2.0
        return min(boundary_r, pair_r)
    
    def _force_relaxation(pts, max_iters=5000, tol=1e-10):
        n = len(pts)
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = _compute_min_radius(pts)
        
        # Adaptive step size
        step = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion (only when closer than some threshold)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, 1e10)
            
            # Only consider pairs that are too close
            close_mask = dist_matrix < 0.2  # threshold
            for i in range(n):
                for j in range(i+1, n):
                    if close_mask[i, j]:
                        d = dist_matrix[i, j]
                        if d < 1e-10:
                            # Random small push
                            forces[i, 0] += np.random.randn() * 0.001
                            forces[i, 1] += np.random.randn() * 0.001
                            forces[j, 0] -= np.random.randn() * 0.001
                            forces[j, 1] -= np.random.randn() * 0.001
                        else:
                            force_mag = 1.0 / (d * d)
                            unit_vec = diff[i, j] / d
                            forces[i] += unit_vec * force_mag
                            forces[j] -= unit_vec * force_mag
            
            # Boundary attraction (push back if outside)
            radii = np.sqrt(np.sum(pts**2, axis=1))
            boundary_force = np.zeros_like(pts)
            for i in range(n):
                if radii[i] > 0.9:
                    # Push toward center
                    boundary_force[i] = -pts[i] / radii[i] * (radii[i] - 0.9)
            
            forces += boundary_force
            
            # Update positions
            pts_new = pts + step * forces
            
            # Check score
            score = _compute_min_radius(pts_new)
            if score > best_score:
                best_score = score
                best_pts = pts_new.copy()
                pts = pts_new
                step = min(step * 1.05, 0.05)
            else:
                step = max(step * 0.95, 0.001)
            
            # Convergence check
            if np.max(np.abs(pts_new - pts)) < tol:
                break
        
        return best_pts, best_score
    
    # Run relaxation with multiple restarts
    best_points = points.copy()
    best_score = _compute_min_radius(points)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = points
        else:
            # Perturb initial configuration
            pts = points + np.random.randn(N, 2) * 0.05
            pts = np.clip(pts, -1.0, 1.0)
        
        result_pts, result_score = _force_relaxation(pts)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts
    
    # Final polish: local optimization with smaller steps
    if time.time() < deadline - 5:
        best_points, best_score = _force_relaxation(best_points, max_iters=2000, tol=1e-12)
    
    return best_points
# EVOLVE-BLOCK-END
