import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # 115s out of 120s to be safe
    
    # Precompute vertices of a unit regular hexagon centered at origin, angle 0
    # Side length = 1, so circumradius = 1
    base_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    base_vertices = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
    
    def get_vertices(cx, cy, angle_deg):
        a = np.radians(angle_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return base_vertices @ R.T + np.array([cx, cy])
    
    def hexagon_distance(v1, v2):
        """Approximate minimum distance between two hexagons using support function."""
        # For convex polygons, we can use the separating axis theorem
        # But for speed, we'll use a simple approach: check if they overlap
        # Using GJK algorithm would be ideal but complex.
        # Instead, we'll use a conservative distance estimate based on center distance
        # and inradius.
        # Inradius of unit hexagon = sqrt(3)/2
        d = np.linalg.norm(v1 - v2)
        return d - np.sqrt(3)  # Conservative: assumes hexagons touch when centers are 2*inradius apart
    
    def compute_outer_side(inner, center, angle_deg):
        """Compute the minimum side length of a regular hexagon containing all inner hexagons."""
        a = np.radians(angle_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        
        max_r = 0
        for cx, cy, ang in inner:
            verts = get_vertices(cx, cy, ang)
            # Transform to outer hexagon's coordinate frame
            verts_centered = verts - np.array(center)
            verts_rotated = verts_centered @ R.T
            max_r = max(max_r, np.max(np.linalg.norm(verts_rotated, axis=1)))
        return max_r
    
    # Initial configuration: hexagons in a grid pattern
    # For N hexagons, arrange in a roughly hexagonal shape
    positions = []
    if N <= 1:
        positions.append((0.0, 0.0, 0.0))
    elif N <= 7:
        # Center + ring
        positions.append((0.0, 0.0, 0.0))
        for i in range(min(6, N - 1)):
            angle = i * 60
            x = np.cos(np.radians(angle)) * np.sqrt(3)
            y = np.sin(np.radians(angle)) * np.sqrt(3)
            positions.append((x, y, 0.0))
    else:
        # Fill in a hexagonal grid
        # Hexagonal grid spacing: distance between centers of touching hexagons = sqrt(3)
        d = np.sqrt(3)
        # Generate positions in a hexagonal pattern
        for row in range(-20, 21):
            for col in range(-20, 21):
                x = d * (col + row * 0.5)
                y = d * row * np.sqrt(3) / 2
                if len(positions) < N:
                    positions.append((x, y, 0.0))
                else:
                    break
            if len(positions) >= N:
                break
    
    if len(positions) < N:
        # Fill remaining
        while len(positions) < N:
            positions.append((0.0, 0.0, 0.0))
    
    # Optimize positions using scipy
    def flatten(positions):
        return np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
    
    def unflatten(x):
        n = len(x) // 3
        return [(x[i*3], x[i*3+1], x[i*3+2]) for i in range(n)]
    
    def objective(x):
        if time.time() > deadline:
            return 1e6
        pos = unflatten(x)
        # Penalty for overlaps
        penalty = 0
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                dist = np.linalg.norm([pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]])
                min_dist = np.sqrt(3) * 0.999  # Slightly less than touching
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100
        # Compute outer hexagon side
        # Center of mass
        cx = np.mean([p[0] for p in pos])
        cy = np.mean([p[1] for p in pos])
        # Try a few angles for the outer hexagon
        best_side = 1e9
        for ang in [0, 30, 60, 90, 120, 150]:
            side = compute_outer_side(pos, (cx, cy), ang)
            best_side = min(best_side, side)
        return best_side + penalty
    
    x0 = flatten(positions)
    
    # Try to optimize
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'fatol': 1e-8, 'xatol': 1e-8})
        if result.fun < 1e5:
            best_pos = unflatten(result.x)
        else:
            best_pos = positions
    except Exception:
        best_pos = positions
    
    # Compute final outer hexagon
    cx = np.mean([p[0] for p in best_pos])
    cy = np.mean([p[1] for p in best_pos])
    best_side = 1e9
    best_angle = 0
    for ang in range(0, 180, 15):
        side = compute_outer_side(best_pos, (cx, cy), ang)
        if side < best_side:
            best_side = side
            best_angle = ang
    
    inner = [[p[0], p[1], p[2]] for p in best_pos]
    return inner, (cx, cy), best_side, best_angle
# EVOLVE-BLOCK-END
