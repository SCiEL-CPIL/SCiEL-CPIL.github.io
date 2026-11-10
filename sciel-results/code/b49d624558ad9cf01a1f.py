import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random
import sys

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Use diverse initial configurations (Fibonacci Sphere, Icosahedron, Random).
    2. For each configuration, run Nelder-Mead on the true (d_min/d_max)^2 objective.
       Nelder-Mead handles the non-smoothness of min/max better than gradient methods.
    3. Polish the result with BFGS on a smooth surrogate to ensure high precision.
    4. Keep track of the best result.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # 30s margin before 630s kill

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Center points to remove translational invariance
        points = points - np.mean(points, axis=0)
        
        # Compute pairwise squared distances
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        ratio_sq = d_min_sq / d_max_sq
        return ratio_sq

    def normalize_rms(points):
        """Normalize points to have unit RMS radius."""
        points = points - np.mean(points, axis=0)
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def smooth_surrogate(points):
        """
        Smooth surrogate for optimization.
        We want to maximize min(d) and minimize max(d).
        Objective to minimize: -log(min_d) + log(max_d)
        Approximation: 
        min_d approx by -1/k * log(sum(exp(-k*d^2)))
        max_d approx by 1/k * log(sum(exp(k*d^2)))
        """
        points = points - np.mean(points, axis=0)
        dists_sq = pdist(points, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        k = 50.0
        # -log(min) ~ 1/k * log(sum(exp(-k*d^2)))
        log_min_term = np.log(np.sum(np.exp(-k * dists_sq))) / k
        # log(max) ~ 1/k * log(sum(exp(k*d^2)))
        log_max_term = np.log(np.sum(np.exp(k * dists_sq))) / k
        
        return log_max_term - log_min_term

    best_score = 0.0
    best_points = None

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Fibonacci Sphere (Uniform distribution)
    def fibonacci_sphere(n, r=1.0):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x * r, y * r, z * r])
        return np.array(points)
    
    initial_configs.append(fibonacci_sphere(n))

    # 2. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts.astype(float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 points at poles or random
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    initial_configs.append(np.vstack([ico_verts, p1, p2]))
    
    # Random perturbations of icosahedron
    for _ in range(5):
        ico_pert = ico_verts + np.random.randn(12, 3) * 0.1
        p1_pert = p1 + np.random.randn(1, 3) * 0.1
        p2_pert = p2 + np.random.randn(1, 3) * 0.1
        initial_configs.append(np.vstack([ico_pert, p1_pert, p2_pert]))

    # 3. Random points on sphere
    for _ in range(10):
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)

    # Optimization Loop
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
            
        try:
            # Phase 1: Nelder-Mead on True Objective
            # NM is slow but robust to non-smoothness
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res_nm = minimize(
                nm_obj, 
                init_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            
            current_pts = normalize_rms(res_nm.x.reshape(n, d))
            
            # Phase 2: Polish with BFGS on Smooth Surrogate
            # This helps refine the solution to high precision
            def polish_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return smooth_surrogate(pts)

            res_polish = minimize(
                polish_obj,
                current_pts.flatten(),
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-10}
            )
            
            polished_pts = normalize_rms(res_polish.x.reshape(n, d))
            
            # Calculate score on polished points
            score = true_score(polished_pts)
            
            # If polish didn't improve, try NM again from polished point?
            # Or just keep the best of NM and Polish
            nm_score = true_score(current_pts)
            
            if score > nm_score:
                final_pts = polished_pts
                final_score = score
            else:
                final_pts = current_pts
                final_score = nm_score

            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
        except Exception as e:
            continue

    # Fallback if all failed
    if best_points is None:
        best_points = fibonacci_sphere(n)

    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)

    return best_points
