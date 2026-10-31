"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import random


def construct():
    """
    Place 11 points inside an equilateral triangle with vertices
    (0,0), (1,0), (0.5, sqrt(3)/2) to maximize the minimum triangle area.
    """
    sqrt3 = np.sqrt(3)
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, sqrt3 / 2.0])
    
    def in_triangle(p):
        """Check if point p is inside the triangle (including boundary)."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, A, B)
        d2 = sign(p, B, C)
        d3 = sign(p, C, A)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def min_triangle_area(points):
        """Compute the minimum area of any triangle formed by three of the points."""
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
        return min_area
    
    def objective(points_flat):
        """Objective function to minimize: negative of the minimum triangle area."""
        points = points_flat.reshape(-1, 2)
        
        # Check if all points are inside the triangle
        for p in points:
            if not in_triangle(p):
                return 1e6
        
        # Add a small penalty for points too close to the boundary
        eps = 1e-5
        for p in points:
            # Distance to edge AB (y=0)
            if p[1] < eps:
                return 1e6
            # Distance to edge BC
            # Line BC: from (1,0) to (0.5, sqrt3/2)
            # Equation: sqrt3/2 * x + 1/2 * y = sqrt3/4
            # Distance = |sqrt3/2 * x + 1/2 * y - sqrt3/4| / sqrt(3/4 + 1/4) = |sqrt3/2 * x + 1/2 * y - sqrt3/4|
            d_bc = abs(sqrt3/2 * p[0] + 0.5 * p[1] - sqrt3/4)
            if d_bc < eps:
                return 1e6
            # Distance to edge CA
            # Line CA: from (0.5, sqrt3/2) to (0,0)
            # Equation: y = sqrt3 * x
            # Distance = |sqrt3 * x - y| / sqrt(3 + 1) = |sqrt3 * x - y| / 2
            d_ca = abs(sqrt3 * p[0] - p[1]) / 2.0
            if d_ca < eps:
                return 1e6
        
        min_area = min_triangle_area(points)
        return -min_area
    
    def optimize_from_start(start_points):
        """Optimize from a given starting configuration."""
        x0 = np.array(start_points).flatten()
        
        try:
            result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', 
                                               options={'maxiter': 200, 'ftol': 1e-10})
            opt_points = result.x.reshape(-1, 2)
            
            # Verify all points are inside
            valid = True
            for p in opt_points:
                if not in_triangle(p):
                    valid = False
                    break
            
            if valid:
                return opt_points
            else:
                return None
        except Exception:
            return None
    
    # Generate initial configurations
    def generate_config_1():
        """Grid-like pattern."""
        pts = [
            [0.05, 0.05],
            [0.95, 0.05],
            [0.5, 0.4],
            [0.25, 0.05],
            [0.75, 0.05],
            [0.15, 0.25],
            [0.85, 0.25],
            [0.5, 0.15],
            [0.35, 0.35],
            [0.65, 0.35],
            [0.5, 0.05],
        ]
        return pts
    
    def generate_random_config(seed):
        """Random configuration."""
        rng = random.Random(seed)
        points = []
        attempts = 0
        while len(points) < N and attempts < 10000:
            attempts += 1
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, sqrt3 / 2.0 * v])
        return points
    
    def generate_known_config():
        """Known good configuration."""
        h = sqrt3 / 2.0
        pts = [
            [0.1, 0.05],
            [0.9, 0.05],
            [0.5, h - 0.05],
            [0.3, 0.05],
            [0.7, 0.05],
            [0.2, 0.3],
            [0.8, 0.3],
            [0.5, 0.15],
            [0.4, 0.4],
            [0.6, 0.4],
            [0.5, 0.05],
        ]
        return pts
    
    # Try multiple configurations
    configs = []
    configs.append(generate_config_1())
    configs.append(generate_known_config())
    for seed in [1, 7, 42, 123, 999]:
        configs.append(generate_random_config(seed))
    
    best_points = None
    best_score = -float('inf')
    
    for start_pts in configs:
        if len(start_pts) != N:
            continue
        opt_pts = optimize_from_start(start_pts)
        if opt_pts is not None:
            score = min_triangle_area(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts
    
    # If no optimization succeeded, use the best initial config
    if best_points is None:
        best_points = np.array(configs[0])
    
    return best_points
# EVOLVE-BLOCK-END
