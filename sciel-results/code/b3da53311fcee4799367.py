import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = angle_rad + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def sat_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT."""
    n1 = len(v1)
    n2 = len(v2)
    
    def project(vertices, axis):
        proj = vertices @ axis
        return proj.min(), proj.max()
    
    for i in range(n1):
        # Edge normal for polygon 1
        edge = v1[(i+1)%n1] - v1[i]
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis /= norm
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        if max1 < min2 or max2 < min1:
            return False
    
    for i in range(n2):
        edge = v2[(i+1)%n2] - v2[i]
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis /= norm
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon local coords
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local coords, hexagon has vertices at angles 0,60,120,180,240,300
    # The hexagon can be defined by 6 half-planes
    # For a unit hexagon (side=1), the apothem is sqrt(3)/2
    # The hexagon contains points where for each of 6 directions, the projection is <= apothem
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(60 * k + 30)  # Normal directions
        proj = lx * np.cos(theta) + ly * np.sin(theta)
        if proj > apothem + 1e-9:
            return False
    return True

def outer_hexagon_side_length(inner_configs, outer_center, outer_angle):
    """
    Given inner hexagon configs (cx, cy, angle) and outer hexagon center/angle,
    find minimum side length of outer hexagon that contains all inner hexagons.
    """
    ocx, ocy = outer_center
    # For each inner hexagon, find the maximum distance from outer center
    # in the direction of each of the 6 outer hexagon normals
    # The outer hexagon side length s must satisfy: for each inner vertex, 
    # the projection onto each outer normal <= s * sqrt(3)/2
    
    outer_normals = []
    outer_angle_rad = np.radians(outer_angle)
    for k in range(6):
        theta = outer_angle_rad + np.radians(60 * k + 30)
        outer_normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    max_proj = 0
    for (cx, cy, angle_deg) in inner_configs:
        verts = hexagon_vertices(cx, cy, 1.0, angle_deg)
        for normal in outer_normals:
            projections = verts @ normal
            max_proj = max(max_proj, np.max(projections))
            # Also need to consider the negative direction (the other side)
            # Actually, we need to check both sides
            min_proj = np.min(projections)
            max_proj = max(max_proj, -min_proj)
    
    # The side length s satisfies: s * sqrt(3)/2 >= max_proj
    # So s >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3)

def objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, ocx, ocy, oangle]
    Total: 3*11 + 3 = 36 parameters
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    ocx = params[3*N]
    ocy = params[3*N+1]
    oangle = params[3*N+2]
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if sat_overlap(v1, v2):
                return 1e6  # Penalty for overlap
    
    # Compute outer side length
    side = outer_hexagon_side_length(inner, (ocx, ocy), oangle)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initial configuration: hexagonal cluster
    # Center at origin, 6 around at distance 2, then 4 more
    # For hexagons with side 1, the distance between centers of adjacent hexagons is 2
    initial_inner = []
    # Center
    initial_inner.append((0.0, 0.0, 0.0))
    # 6 around
    for k in range(6):
        angle = np.radians(60 * k)
        initial_inner.append((2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0))
    # 4 more: place them in the gaps
    # Gaps are at angles 30, 90, 150, 210, 270, 330 at distance 2*sqrt(3)
    # Let's place 4 of them
    for k in [1, 3, 5, 0]:  # Pick 4 gap positions
        angle = np.radians(60 * k + 30)
        dist = 2.0 * np.sqrt(3)
        initial_inner.append((dist * np.cos(angle), dist * np.sin(angle), 0.0))
    
    # Initial outer
    ocx, ocy = 0.0, 0.0
    oangle = 0.0
    
    # Initial params
    x0 = []
    for (x, y, a) in initial_inner:
        x0.extend([x, y, a])
    x0.extend([ocx, ocy, oangle])
    x0 = np.array(x0)
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Try multiple random restarts
    np.random.seed(42)
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random perturbation
        if r == 0:
            params = x0.copy()
        else:
            params = x0 + np.random.randn(len(x0)) * 0.5
        
        try:
            result = minimize(objective, params, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Check if the best result is valid (not overlapping)
    # If best_score is high, it means overlaps were present.
    # Fall back to the initial valid configuration if so.
    if best_score > 1e5:
        inner = [[x, y, a] for (x, y, a) in initial_inner]
        ocx, ocy = 0.0, 0.0
        oangle = 0.0
    else:
        inner = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            a = best_params[3*i+2]
            inner.append([x, y, a])
        ocx = best_params[3*N]
        ocy = best_params[3*N+1]
        oangle = best_params[3*N+2]
        
    # Recalculate side length with a small safety margin to ensure containment
    # This handles floating point precision issues that might cause "not contained" errors
    final_side = outer_hexagon_side_length(inner, (ocx, ocy), oangle)
    final_side += 1e-5
    
    return inner, (ocx, ocy), final_side, oangle
