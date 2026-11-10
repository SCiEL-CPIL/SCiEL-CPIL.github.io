import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def get_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Objective to minimize: -log(d_min) or similar? 
    # Actually, maximizing sum of logs is smoother.
    # Let's use a combination: maximize min distance primarily.
    # We will use a custom optimization loop using Gradient Ascent on the sphere.
    
    def repulsion_force(points, power=3.0):
        """
        Calculate repulsive forces on each point.
        Force on i is sum over j of (p_i - p_j) / ||p_i - p_j||^(power+1) * ||p_i-p_j||^power ?
        Standard Coulomb: F = (p_i - p_j) / ||p_i - p_j||^3
        This corresponds to potential U = 1/r.
        Gradient of sum of 1/r is sum of (p_i-p_j)/r^3.
        We want to maximize distances, so we move in direction of force.
        """
        forces = np.zeros_like(points)
        n_pts = len(points)
        for i in range(n_pts):
            for j in range(n_pts):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    # Avoid division by zero, add small random jitter
                    forces[i] += np.random.normal(0, 1e-6, 3)
                    continue
                # Force magnitude proportional to 1/dist^2, direction diff/dist
                # F = diff / dist^3
                forces[i] += diff / (dist ** 3)
        return forces

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-12] = 1.0
        return points / norms

    def optimize_sphere(points, n_iter=2000, step_size=0.05, decay=0.999):
        pts = points.copy()
        current_score = get_score(pts)
        
        for i in range(n_iter):
            forces = repulsion_force(pts)
            # Project forces to tangent space? 
            # Simply adding force and renormalizing is a valid Riemannian gradient step approximation
            # F_tangent = F - (F . p) p
            # But renormalizing after p + eta*F handles the radial component implicitly.
            
            # Adaptive step size: if score decreases, reduce step
            new_pts = pts + step_size * forces
            new_pts = project_to_sphere(new_pts)
            
            new_score = get_score(new_pts)
            
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
            else:
                # Reduce step size if stuck or moving wrong
                step_size *= 0.9
                # If step size is very small, break
                if step_size < 1e-8:
                    break

        return pts, current_score

    best_points = None
    best_score = -1.0
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for r in range(n_restarts):
        if r == 0:
            # Fibonacci Sphere
            phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
            theta = np.pi * (1 + 5**0.5) * np.arange(n)
            points = np.column_stack((
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ))
        elif r == 1:
            # Random
            points = np.random.normal(0, 1, size=(n, 3))
            points = project_to_sphere(points)
        else:
            # Perturbed Fibonacci
            phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
            theta = np.pi * (1 + 5**0.5) * np.arange(n)
            points = np.column_stack((
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ))
            points = points + np.random.normal(0, 0.2, size=(n, 3))
            points = project_to_sphere(points)
            
        optimized_pts, score = optimize_sphere(points, n_iter=3000, step_size=0.1)
        
        if score > best_score:
            best_score = score
            best_points = optimized_pts

    return best_points
