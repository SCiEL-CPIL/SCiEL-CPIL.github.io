import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    # 1. Define inner hexagon centers
    # Layout: A compact "cross" shape on the hexagonal grid
    # Axial coordinates (q, r)
    points_axial = [
        (0, 0),   # Center
        (1, 0), (-1, 0), (2, 0), (-2, 0),  # Horizontal axis
        (0, 1), (-1, 1),  # Top-leftish
        (0, -1), (1, -1), # Bottom-rightish
        (0, 2),   # Top
        (0, -2)   # Bottom
    ]
    
    # Convert axial to cartesian
    # Distance between centers of touching unit hexagons is sqrt(3)
    d = np.sqrt(3)
    centers = []
    for q, r in points_axial:
        x = d * (q + r / 2.0)
        y = d * (3.0 / 2.0) * r
        centers.append((x, y))
    centers = np.array(centers)
    
    # 2. Define the outer hexagon containment check
    # Outer hexagon parameters: cx, cy, S, angle
    # Vertices of outer hexagon
    def is_inside(cx, cy, S, angle, p):
        # Translate point to center
        px = p[0] - cx
        py = p[1] - cy
        
        # Rotate point by -angle to align hexagon
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        
        # Check against 6 sides of a regular hexagon of side S centered at origin
        # Distance from center to side is S * sqrt(3) / 2
        # The normals to the sides are at angles 30, 90, 150, 210, 270, 330 degrees
        # (Assuming vertex at 0 deg, so flat top/bottom? No, vertex at 0 means pointy top/bottom)
        # If vertex is at angle `angle`, the sides are perpendicular to vectors at angle + 30 + k*60
        
        # Let's use the standard inequality for a regular polygon
        # For a hexagon with vertices at angles theta_k = angle + k*pi/3
        # The constraint is max_k |dot(p, normal_k)| <= S*sqrt(3)/2
        # Normals are at angles angle + pi/6 + k*pi/3
        
        normals = []
        for k in range(6):
            theta = angle + np.pi/6 + k * np.pi/3
            nx = np.cos(theta)
            ny = np.sin(theta)
            normals.append((nx, ny))
            
        max_dist = 0
        for nx, ny in normals:
            dist = abs(rx * nx + ry * ny)
            if dist > max_dist:
                max_dist = dist
                
        return max_dist <= S * np.sqrt(3) / 2

    def objective(params):
        cx, cy, S, angle = params
        # We want to minimize S
        # But we must satisfy constraints. 
        # Penalty method: return S + penalty if points are outside
        
        penalty = 0
        for p in centers:
            if not is_inside(cx, cy, S, angle, p):
                # Calculate violation
                px = p[0] - cx
                py = p[1] - cy
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                
                for k in range(6):
                    theta = angle + np.pi/6 + k * np.pi/3
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    dist = abs(rx * nx + ry * ny)
                    limit = S * np.sqrt(3) / 2
                    if dist > limit:
                        penalty += (dist - limit)
                        
        return S + 100 * penalty

    # Initial guess
    cx, cy = 0.0, 0.0
    S_init = 3.5
    angle_init = 0.0
    
    x0 = [cx, cy, S_init, angle_init]
    
    # Optimization
    best_S = S_init
    best_params = x0
    try:
        # Use Nelder-Mead as it handles non-smooth penalties well and doesn't need gradients
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000})
        
        if res.success or res.fun < objective(x0):
            best_params = res.x
            best_S = res.x[2]
            
            # Verify validity
            if all(is_inside(best_params[0], best_params[1], best_S, best_params[3], p) for p in centers):
                pass
            else:
                # Fallback to initial valid bounding box if optimizer failed
                best_S = 2 * np.sqrt(3)
                best_params = [0.0, 0.0, best_S, 0.0]
                
    except Exception as e:
        # Fallback
        best_S = 2 * np.sqrt(3)
        best_params = [0.0, 0.0, best_S, 0.0]
        
    # Time check
    if time.time() > deadline:
        # Return best found so far
        pass

    cx, cy, S, angle = best_params
    
    # Format inner hexagons
    # inner: list of [x, y, angle_degrees]
    # All inner hexagons are aligned (angle 0)
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (cx, cy), S, np.degrees(angle)

