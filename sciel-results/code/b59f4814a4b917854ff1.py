"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    A_tri = 0.5 * 1.0 * h  # area = sqrt(3)/4
    
    def point_in_triangle(p):
        """Check if point is inside or on the boundary of the triangle."""
        x, y = p
        # Using barycentric coordinates
        d0 = np.cross(V[1] - V[0], p - V[0])
        d1 = np.cross(V[2] - V[1], p - V[1])
        d2 = np.cross(V[0] - V[2], p - V[2])
        # All should have same sign for CCW triangle
        return d0 >= -1e-10 and d1 >= -1e-10 and d2 >= -1e-10
    
    def project_to_triangle(p):
        """Project point onto the triangle if outside."""
        if point_in_triangle(p):
            return p
        # Simple projection: scale towards centroid
        centroid = np.mean(V, axis=0)
        for t in np.linspace(1.0, 0.0, 100):
            q = centroid + t * (p - centroid)
            if point_in_triangle(q):
                return q
        return centroid
    
    def min_triangle_area(points):
        """Compute the minimum area of any triangle formed by 3 points."""
        n = len(points)
        min_area = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                    if area < min_area:
                        min_area = area
        return min_area
    
    def soft_min_area(points, alpha=20.0):
        """
        Smooth surrogate for max-min: we want to maximize the min area.
        Equivalent to minimizing -min(area). Use log-sum-exp for smoothness.
        We minimize the soft-min of areas, which approximates the true min.
        """
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                    areas.append(area)
        areas = np.array(areas)
        if len(areas) == 0:
            return 0.0
        # Soft min via log-sum-exp: soft_min = -log(sum(exp(-alpha * x))) / alpha
        # This is a smooth approximation of min(x)
        shifted = -alpha * areas
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = -log_sum / alpha
        return soft_min
    
    def objective_flat(x):
        """Objective for optimization: minimize negative soft-min area."""
        if time.time() > deadline - 5.0:
            return 1e10  # large penalty
        points = x.reshape(-1, 2)
        # Penalize points outside triangle
        penalty = 0.0
        for p in points:
            if not point_in_triangle(p):
                # Distance from triangle (approximate)
                centroid = np.mean(V, axis=0)
                penalty += 100.0 * np.sum((p - centroid) ** 2)
        val = -soft_min_area(points, alpha=30.0) + penalty
        return val
    
    def gradient_flat(x):
        """Numerical gradient."""
        if time.time() > deadline - 5.0:
            return np.zeros_like(x)
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = objective_flat(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (objective_flat(x_plus) - f0) / eps
        return grad
    
    # Initial points: vertices + centroid + midpoints + some interior
    # Start with a reasonable initial configuration
    centroid = np.mean(V, axis=0)
    midpoints = [(V[i] + V[j]) / 2 for i, j in [(0,1), (1,2), (2,0)]]
    
    # 11 points: 3 vertices, 3 midpoints, 1 centroid, 4 more
    initial_points = []
    initial_points.extend(V)
    initial_points.extend(midpoints)
    initial_points.append(centroid)
    # Add 4 more points near the center but spread out
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        r = 0.15
        px = centroid[0] + r * np.cos(angle)
        py = centroid[1] + r * np.sin(angle)
        initial_points.append([px, py])
    
    initial_points = np.array(initial_points)
    best_points = initial_points.copy()
    best_score = min_triangle_area(initial_points)
    
    # Run multiple optimization attempts with different initializations
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() > deadline - 15.0:
            break
        
        if attempt == 0:
            x0 = initial_points.flatten()
        else:
            # Perturb initial points
            rng = np.random.RandomState(attempt * 123 + 7)
            x0 = initial_points.copy()
            noise = rng.normal(0, 0.05, size=x0.shape)
            x0 = x0 + noise
            # Project to triangle
            for i in range(len(x0)):
                x0[i] = project_to_triangle(x0[i])
            x0 = x0.flatten()
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x in [0,1]
            bounds.append((0.0, h))   # y in [0, h]
        
        try:
            result = minimize(
                objective_flat,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < objective_flat(x0):
                points_opt = result.x.reshape(-1, 2)
                # Project to triangle
                for i in range(len(points_opt)):
                    points_opt[i] = project_to_triangle(points_opt[i])
                
                score = min_triangle_area(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    
        except Exception:
            pass
    
    # Final polish: try a few more iterations with higher alpha
    if time.time() < deadline - 10.0:
        x0 = best_points.flatten()
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, h))
        
        try:
            def objective_polish(x):
                if time.time() > deadline - 5.0:
                    return 1e10
                points = x.reshape(-1, 2)
                penalty = 0.0
                for p in points:
                    if not point_in_triangle(p):
                        centroid = np.mean(V, axis=0)
                        penalty += 100.0 * np.sum((p - centroid) ** 2)
                val = -soft_min_area(points, alpha=50.0) + penalty
                return val
            
            result = minimize(
                objective_polish,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success:
                points_opt = result.x.reshape(-1, 2)
                for i in range(len(points_opt)):
                    points_opt[i] = project_to_triangle(points_opt[i])
                
                score = min_triangle_area(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    
        except Exception:
            pass
    
    # Ensure all points are strictly inside
    for i in range(N):
        best_points[i] = project_to_triangle(best_points[i])
    
    return best_points
# EVOLVE-BLOCK-END
