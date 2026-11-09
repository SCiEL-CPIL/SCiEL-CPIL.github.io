import numpy as np
from scipy.spatial.distance import pdist

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

    def fibonacci_sphere(n):
        phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * np.arange(n)
        return np.column_stack((
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ))

    def optimize_repulsion(init_points):
        points = init_points.copy()
        steps = 5000
        base_lr = 0.05
        decay = 0.999
        
        for i in range(steps):
            lr = base_lr * (decay ** i)
            forces = np.zeros_like(points)
            
            # Calculate repulsive forces (Coulomb-like 1/r^2 force)
            for i_idx in range(n):
                for j_idx in range(n):
                    if i_idx == j_idx:
                        continue
                    diff = points[i_idx] - points[j_idx]
                    dist_sq = np.dot(diff, diff)
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        force = np.random.normal(size=3) * 0.1
                    else:
                        force = diff / (dist_sq * dist)
                    forces[i_idx] += force
            
            # Update points
            for i_idx in range(n):
                p = points[i_idx]
                f = forces[i_idx]
                
                # Project force to tangent plane
                f_dot_p = np.dot(f, p)
                f_tangent = f - f_dot_p * p
                
                # Move
                points[i_idx] += lr * f_tangent
                
                # Normalize
                norm = np.linalg.norm(points[i_idx])
                if norm > 1e-10:
                    points[i_idx] /= norm
                else:
                    points[i_idx] = np.array([1.0, 0.0, 0.0])
                    
        return points

    best_points = None
    best_score = -1.0
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for r in range(n_restarts):
        if r == 0:
            init_pts = fibonacci_sphere(n)
        else:
            # Random initialization
            init_pts = np.random.normal(size=(n, 3))
            init_pts /= np.linalg.norm(init_pts, axis=1, keepdims=True)
            
        # Optimize
        opt_pts = optimize_repulsion(init_pts)
        
        # Score
        score = get_score(opt_pts)
        
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
            
    return best_points
