import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Center-to-center distance for touching unit hexagons
    d = np.sqrt(3)
    
    # Generate candidate center configurations
    configs = []
    
    # Linear row
    linear_centers = [(i - N//2) * d for i in range(N)]
    configs.append(('linear', linear_centers))
    
    # Hexagonal cluster: pick 11 closest points from triangular lattice
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = d * (q + r/2)
            y = d * np.sqrt(3)/2 * r
            dist = (abs(q) + abs(r) + abs(q+r)) / 2
            if dist <= 2:
                lattice_points.append((x, y))
    
    lattice_points_sorted = sorted(lattice_points, key=lambda p: p[0]**2 + p[1]**2)
    cluster_centers = [p[0] for p in lattice_points_sorted[:N]]  # x-coords
    cluster_ys = [p[1] for p in lattice_points_sorted[:N]]
    configs.append(('cluster_x', cluster_centers))
    configs.append(('cluster_y', cluster_ys))
    
    # Store full 2D centers for cluster
    cluster_2d = [(x, y) for x, y in lattice_points_sorted[:N]]
    
    best_side = float('inf')
    best_result = None
    
    # For a set of 2D centers, find optimal outer hexagon
    def optimize_outer_hexagon(centers_2d, time_limit=5.0):
        """Find optimal outer hexagon for given inner centers"""
        start_opt = time.time()
        
        def required_side(cx, cy, theta):
            """Compute minimum side length for outer hexagon to contain all inner hexagons"""
            max_s = 0.0
            for (px, py) in centers_2d:
                # Inner hexagon vertices (assuming rotation 0)
                for k in range(6):
                    angle = k * np.pi / 3
                    vx = px + np.cos(angle)
                    vy = py + np.sin(angle)
                    
                    # Check if this vertex is inside outer hexagon
                    dx = vx - cx
                    dy = vy - cy
                    cos_t = np.cos(-theta)
                    sin_t = np.sin(-theta)
                    rx = dx * cos_t - dy * sin_t
                    ry = dx * sin_t + dy * cos_t
                    
                    # Six side normals for flat-top hexagon
                    for a in [np.pi/6 + m * np.pi/3 for m in range(6)]:
                        val = rx * np.cos(a) + ry * np.sin(a)
                        if val > 0:
                            s_req = val * 2.0 / np.sqrt(3)
                            if s_req > max_s:
                                max_s = s_req
            return max_s
        
        # Try multiple starting points
        best_local = float('inf')
        best_params = None
        
        for cx0 in [-3, -1, 0, 1, 3]:
            for cy0 in [-3, -1, 0, 1, 3]:
                for theta0 in [0, np.pi/6, np.pi/3, np.pi/2]:
                    if time.time() - start_opt > time_limit:
                        return best_local, best_params
                    
                    x0 = np.array([cx0, cy0, theta0])
                    try:
                        result = minimize(required_side, x0, method='Nelder-Mead',
                                         options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                        if result.fun < best_local:
                            best_local = result.fun
                            best_params = result.x
                    except:
                        pass
        
        return best_local, best_params
    
    # Process each config
    # For linear: centers are on x-axis
    linear_2d = [(x, 0.0) for x in linear_centers]
    side, params = optimize_outer_hexagon(linear_2d, time_limit=10.0)
    if params is not None and side < best_side:
        best_side = side
        best_result = (linear_2d, params)
    
    # For cluster
    if time.time() < deadline - 30:
        side, params = optimize_outer_hexagon(cluster_2d, time_limit=20.0)
        if params is not None and side < best_side:
            best_side = side
            best_result = (cluster_2d, params)
    
    # Try random perturbations of cluster
    np.random.seed(42)
    for trial in range(10):
        if time.time() > deadline - 20:
            break
        
        noisy_2d = [(x + np.random.normal(0, 0.2), y + np.random.normal(0, 0.2)) 
                   for x, y in cluster_2d]
        side, params = optimize_outer_hexagon(noisy_2d, time_limit=5.0)
        if params is not None and side < best_side:
            best_side = side
            best_result = (noisy_2d, params)
    
    # Fallback
    if best_result is None:
        inner = [[(i - N//2) * d, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    centers_2d, params = best_result
    cx, cy, theta = params
    
    inner = [[x, y, 0.0] for x, y in centers_2d]
    return inner, (cx, cy), best_side, theta * 180.0 / np.pi
# EVOLVE-BLOCK-END
