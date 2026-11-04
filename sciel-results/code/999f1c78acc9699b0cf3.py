"""Packing 12 unit hexagons in a regular hexagon via optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, angle_deg, side):
    """Check if point (px,py) is inside hexagon with center (cx,cy), angle, side."""
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # In local frame, hexagon has vertices at angles 0, 60, ..., 300
    # The hexagon is the intersection of 6 half-planes
    # Edge normals at angles 30, 90, 150, 210, 270, 330 degrees (for flat-top)
    # Actually for pointy-top (vertex at angle 0), normals are at 90, 150, 210, 270, 330, 30
    # Apothem = side * cos(pi/6) = side * sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        normal_angle = np.pi / 6 + i * np.pi / 3  # 30, 90, 150, ...
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-9:
            return False
    return True

def hexagon_in_container(cx, cy, angle_deg, container_cx, container_cy, container_side, container_angle):
    """Check if unit hexagon is inside container hexagon."""
    verts = hex_vertices(cx, cy, angle_deg, 1.0)
    for vx, vy in verts:
        if not point_in_hexagon(vx, vy, container_cx, container_cy, container_angle, container_side):
            return False
    return True

def min_distance_hexagons(cx1, cy1, a1, cx2, cy2, a2):
    """Compute minimum distance between two unit hexagons."""
    v1 = hex_vertices(cx1, cy1, a1, 1.0)
    v2 = hex_vertices(cx2, cy2, a2, 1.0)
    min_dist = np.inf
    # Check all vertex-vertex distances
    for p in v1:
        for q in v2:
            d = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
            if d < min_dist:
                min_dist = d
    # Also check if one hexagon is inside the other (distance should be negative in that case)
    # For simplicity, check if any vertex of v1 is inside v2 or vice versa
    for p in v1:
        if point_in_hexagon(p[0], p[1], cx2, cy2, a2, 1.0):
            return 0.0  # overlap
    for q in v2:
        if point_in_hexagon(q[0], q[1], cx1, cy1, a1, 1.0):
            return 0.0  # overlap
    return min_dist

def objective(x):
    """Minimize container side length. x = [cx, cy, s, phi, then for each hex: x_i, y_i, theta_i]"""
    cx, cy, s, phi = x[0], x[1], x[2], x[3]
    if s < 1.0:
        return 1e6
    hex_data = x[4:].reshape(N, 3)
    # Check all hexagons are inside container
    for i in range(N):
        hx, hy, ha = hex_data[i]
        if not hexagon_in_container(hx, hy, ha, cx, cy, phi, s):
            return 1e6 + abs(s - 1.0)
    # Check non-overlap
    for i in range(N):
        for j in range(i+1, N):
            d = min_distance_hexagons(hex_data[i][0], hex_data[i][1], hex_data[i][2],
                                       hex_data[j][0], hex_data[j][1], hex_data[j][2])
            if d < 1e-6:
                return 1e6 + abs(s - 1.0)
    return s

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal arrangement
    # Place 7 in a hexagonal pattern (1 center + 6 around), then 5 more
    # For parallel hexagons, spacing is 2*sqrt(3)/2 = sqrt(3) in one direction, 2 in another
    # Actually for flat-top hexagons, horizontal spacing is 3/2 * side * 2 = 3, vertical is sqrt(3)
    # Let me use a simple grid
    
    best_result = None
    best_score = np.inf
    
    # Multiple restarts
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Initial placement
        if restart == 0:
            # Hexagonal pattern: 2-3-4-3-2 rows won't work for 12
            # Try: 4 rows of 3, or 3 rows of 4
            # For parallel hexagons, use offset grid
            # Row spacing: sqrt(3), col spacing: 1.5, offset alternate rows by 0.75
            positions = []
            # 3 rows x 4 cols
            for row in range(3):
                for col in range(4):
                    x_pos = col * 1.5 + (0.75 if row % 2 else 0)
                    y_pos = row * np.sqrt(3)
                    positions.append([x_pos, y_pos, 0.0])
            # Center them
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            cx0 = (min(xs) + max(xs)) / 2
            cy0 = (min(ys) + max(ys)) / 2
            positions = [[p[0] - cx0, p[1] - cy0, p[2]] for p in positions]
            init = [0.0, 0.0, 8.0, 0.0] + [v for p in positions for v in p]
        else:
            # Perturbed initial
            if best_result is not None:
                init = list(best_result)
                for i in range(4, len(init)):
                    init[i] += rng.normal(0, 0.5)
                init[2] += rng.normal(0, 0.5)
            else:
                init = [0.0, 0.0, 10.0, 0.0] + [rng.normal(0, 5) for _ in range(N*3)]
        
        # Bounds
        bounds = [(-10, 10), (-10, 10), (1.0, 30.0), (-180, 180)]
        for i in range(N):
            bounds += [(-10, 10), (-10, 10), (-180, 180)]
        
        try:
            result = minimize(objective, init, method='SLSQP', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-8})
        except Exception:
            continue
        
        if result.success or result.fun < best_score:
            score = result.fun
            if score < best_score and score < 1e5:
                best_score = score
                best_result = result.x
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    if best_result is None:
        # Fallback
        positions = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return positions, (0.0, 0.0), 40.0, 0.0
    
    cx, cy, s, phi = best_result[0], best_result[1], best_result[2], best_result[3]
    hex_data = best_result[4:].reshape(N, 3)
    inner = [[float(h[0]), float(h[1]), float(h[2])] for h in hex_data]
    
    return inner, (float(cx), float(cy)), float(s), float(phi)
