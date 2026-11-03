import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Helper to normalize points
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    # Helper to compute covering radius and worst facet
    def get_covering_radius_and_worst(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_simplex = None
            worst_center = None
            
            for simplex in hull.simplices:
                facet_pts = pts[simplex]
                v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                
                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n /= n_norm
                
                if n @ v0 < 0:
                    n = -n
                
                dist = np.arccos(np.clip(n @ v0, -1, 1))
                
                if dist > max_rad:
                    max_rad = dist
                    worst_simplex = simplex
                    worst_center = n
                    
            return max_rad, worst_simplex, worst_center
        except Exception:
            return np.pi, None, None

    # Initial points: Fibonacci Spiral
    def init_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return normalize(points)

    best_points = init_fibonacci()
    best_radius, _, _ = get_covering_radius_and_worst(best_points)
    
    points = best_points.copy()
    
    # Simulated Annealing with gradient-based moves
    temp = 0.02
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        current_radius, worst_simplex, worst_center = get_covering_radius_and_worst(points)
        
        if worst_simplex is None:
            break
            
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Generate candidate move: perturb worst facet vertices
        if worst_simplex is not None and worst_center is not None:
            candidate = points.copy()
            
            # Move worst facet vertices towards center
            for idx in worst_simplex:
                candidate[idx] = candidate[idx] + 0.1 * (worst_center - candidate[idx])
            
            # Add small random perturbation to all points
            noise = temp * np.random.randn(N, 3)
            candidate += noise
            
            # Normalize
            candidate = normalize(candidate)
            
            # Evaluate candidate
            cand_radius, _, _ = get_covering_radius_and_worst(candidate)
            
            # Accept if better, or with probability exp(-delta/temp) if worse
            delta = cand_radius - current_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-10)):
                points = candidate
                current_radius = cand_radius
            
            temp *= cooling
            
            if temp < 1e-5:
                break
    
    return best_points
