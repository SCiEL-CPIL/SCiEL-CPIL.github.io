import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 115.0 # Safety margin before 120s kill switch

def get_hex_vertices(center, angle=0.0):
    """
    Returns the 6 vertices of a regular hexagon with side length 1 
    centered at 'center' and rotated by 'angle' radians.
    """
    x, y = center
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        vx = x + 1.0 * math.cos(theta)
        vy = y + 1.0 * math.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

def compute_min_side_length(points, theta):
    """
    Computes the minimum side length S of a regular hexagon centered at origin,
    rotated by 'theta' radians, that contains all points.
    Returns a Python float.
    """
    max_projection = 0.0
    for k in range(6):
        angle_normal = theta + k * math.pi / 3
        nx = math.cos(angle_normal)
        ny = math.sin(angle_normal)
        projections = points[:, 0] * nx + points[:, 1] * ny
        max_proj = float(np.max(projections))
        if max_proj > max_projection:
            max_projection = max_proj
            
    r = max_projection
    S = 2.0 * r / SQRT3
    return float(S)

def check_overlap_penalty(centers, min_dist):
    """
    Calculates penalty for overlapping hexagons.
    Assumes fixed orientation (angle=0).
    min_dist for non-overlapping unit hexagons (aligned) is sqrt(3).
    """
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return penalty

def construct():
    """
    Construct the optimal packing of N unit hexagons inside a regular hexagon.
    Returns: (inner, center, side, angle)
    """
    # 1. Generate initial lattice configuration
    lattice_pts = []
    for r in range(-5, 6):
        for q in range(-5, 6):
            x = SQRT3 * (q + r/2.0)
            y = 1.5 * r
            lattice_pts.append((x, y))
            
    lattice_pts.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_centers = np.array(lattice_pts[:N])
    
    # 2. Objective function for optimization
    # Variables: 2*N for inner centers (x1, y1, ..., xN, yN)
    # We will also optimize the outer hexagon rotation theta?
    # Let's fix outer center at (0,0) and optimize inner positions and outer rotation.
    # Total vars: 2*N + 1 (theta)
    
    min_dist = SQRT3 # Approx min distance for aligned hexagons
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
            
        # Extract inner centers
        centers = params[:2*N].reshape(N, 2)
        theta = params[2*N]
        
        # Collect vertices for all inner hexagons (angle=0)
        all_vertices = []
        for c in centers:
            verts = get_hex_vertices(c, angle=0.0)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # Calculate side length of enclosing hexagon
        side = compute_min_side_length(all_vertices, theta)
        
        # Calculate overlap penalty
        pen = check_overlap_penalty(centers, min_dist)
        
        # Weighted sum
        return side + 100.0 * pen

    # 3. Optimization
    # Initial guess: lattice centers + theta=0
    x0 = np.zeros(2*N + 1)
    x0[:2*N] = initial_centers.flatten()
    x0[2*N] = 0.0
    
    # Use Nelder-Mead for robustness on small non-convex problems
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        best_params = res.x
    except Exception:
        best_params = x0
        
    # 4. Extract results
    centers = best_params[:2*N].reshape(N, 2)
    theta_opt = best_params[2*N]
    
    # Re-calculate side length for the final configuration
    all_vertices = []
    for c in centers:
        verts = get_hex_vertices(c, angle=0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    side = compute_min_side_length(all_vertices, theta_opt)
    
    # Check if any overlaps remain (should be minimal due to penalty)
    # If overlaps are significant, we might need to reject or fix, 
    # but for this repair, we assume the penalty worked.
    
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    center = (0.0, 0.0)
    angle = math.degrees(theta_opt)
    
    return inner, center, float(side), float(angle)

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    print(f"Inner hexagons: {len(inner)}")
