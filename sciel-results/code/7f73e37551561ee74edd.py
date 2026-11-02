import numpy as np
from scipy.spatial import ConvexHull

def construct():
    N = 50
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def compute_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_rad = 0.0
            for tri in facets:
                a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                # Ensure normal points outward
                if np.dot(n, a) < 0:
                    n = -n
                dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
                rad = np.arccos(dot_val)
                if rad > max_rad:
                    max_rad = rad
            return max_rad
        except Exception:
            return np.inf

    current_points = points.copy()
    current_cost = compute_covering_radius(current_points)
    best_points = current_points.copy()
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature (in radians)
    cooling_rate = 0.995
    min_T = 1e-6
    max_iterations = 5000
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iterations):
        if T < min_T:
            break
            
        # Select a random point to move
        idx = rng.integers(0, N)
        old_point = current_points[idx].copy()
        
        # Generate a new random point on the sphere
        # Using uniform distribution on sphere
        u = rng.random()
        v = rng.random()
        theta = 2.0 * np.pi * u
        phi = np.arccos(2.0 * v - 1.0)
        new_point = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Apply move
        current_points[idx] = new_point
        
        # Compute new cost
        new_cost = compute_covering_radius(current_points)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_points = current_points.copy()
        else:
            # Reject move
            current_points[idx] = old_point
            
        # Cool down
        T *= cooling_rate
        
        # Optional: Local polish every few thousand steps? 
        # Given time constraints, maybe just run SA for a long time.
        
    # Final local optimization on the best point found
    points = best_points
    for it in range(500):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        facets = hull.simplices
        radii = np.zeros(len(facets))
        centers = np.zeros((len(facets), 3))
        
        for k, tri in enumerate(facets):
            a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            if np.dot(n, a) < 0:
                u = -n
            else:
                u = n
            centers[k] = u
            dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
            radii[k] = np.arccos(dot_val)
            
        current_max_rad = np.max(radii)
        
        push = np.zeros((N, 3))
        weight = np.zeros(N)
        threshold = current_max_rad - 0.05 * np.pi
        
        for k, tri in enumerate(facets):
            if radii[k] > threshold:
                w = radii[k]
                for idx in tri:
                    push[idx] += w * centers[k]
                    weight[idx] += w
                    
        for i in range(N):
            if weight[i] > 1e-10:
                target = push[i] / weight[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target /= norm_t
                    step = 0.01
                    new_p = points[i] + step * (target - points[i])
                    new_p /= np.linalg.norm(new_p)
                    points[i] = new_p

    return points
