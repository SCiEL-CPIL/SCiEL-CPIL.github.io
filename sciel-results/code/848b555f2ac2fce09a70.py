import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    best_score = 0.0
    best_points = None
    
    # Run multiple initializations to avoid local minima
    # 10 runs of ~14s each
    num_runs = 10
    
    for run_id in range(num_runs):
        if time.time() > deadline:
            break
            
        # 1. Initialization
        # Hexagonal grid init
        spacing = 0.15
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0]
        
        if len(points) < N:
            while len(points) < N:
                p = np.random.uniform(-1, 1, size=2)
                if np.linalg.norm(p) < 1:
                    points = np.vstack([points, p])
        else:
            np.random.shuffle(points)
            points = points[:N]
            
        # Add noise
        points += np.random.normal(0, 0.02, size=(N, 2))
        # Project to circle
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * np.minimum(norms, 0.9)
        
        # 2. Optimization
        r = 0.1
        delta_r = 0.0002
        alpha = 0.02
        max_iter = 10000
        tol = 1e-6
        
        # Track progress
        last_r_increase = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Distance to boundary
            dist_bound = 1.0 - np.linalg.norm(points, axis=1)
            
            # Violations
            mask = ~np.eye(N, dtype=bool)
            violations_pair = np.where(mask, np.maximum(0, 2*r - dist), 0)
            violations_bound = np.maximum(0, r - dist_bound)
            
            total_violation = np.sum(violations_pair) + np.sum(violations_bound)
            
            if total_violation < tol:
                # Feasible
                r += delta_r
                last_r_increase = it
                
                # Update best score
                min_d = np.min(dist[mask]) / 2.0
                min_b = np.min(dist_bound)
                score = min(min_d, min_b)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
            else:
                # Relax
                forces = np.zeros_like(points)
                
                # Pairwise forces
                safe_dist = np.where(dist > 1e-8, dist, 1e-8)
                force_factor = violations_pair / safe_dist
                np.fill_diagonal(force_factor, 0)
                
                forces += np.sum(diff * force_factor[:, :, np.newaxis], axis=1)
                
                # Boundary forces
                b_viol = violations_bound
                norms = np.linalg.norm(points, axis=1)
                safe_norms = np.where(norms > 1e-8, norms, 1e-8)
                forces -= points * (b_viol / safe_norms)[:, np.newaxis]
                
                # Update
                points += alpha * forces
                
                # Project to circle
                norms = np.linalg.norm(points, axis=1)
                mask_out = norms > 1.0
                if np.any(mask_out):
                    points[mask_out] = points[mask_out] / norms[mask_out, np.newaxis]
                    
                # Noise to escape local minima
                # Increase noise if stuck
                if it - last_r_increase > 200:
                    noise_scale = 0.01
                else:
                    noise_scale = 0.001
                    
                points += np.random.normal(0, noise_scale, size=(N, 2))
                
                # Re-project
                norms = np.linalg.norm(points, axis=1)
                mask_out = norms > 1.0
                if np.any(mask_out):
                    points[mask_out] = points[mask_out] / norms[mask_out, np.newaxis]

        # End of optimization for this run
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        min_d = np.min(dist[mask]) / 2.0
        dist_bound = 1.0 - np.linalg.norm(points, axis=1)
        min_b = np.min(dist_bound)
        score = min(min_d, min_b)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()

    if best_points is None:
        # Fallback
        spacing = 0.2
        k = 5
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0]
        np.random.shuffle(points)
        best_points = points[:N]
        
    return best_points
