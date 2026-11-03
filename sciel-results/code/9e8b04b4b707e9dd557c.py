import numpy as np
import time
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (flat-top when angle=0)."""
    angle = np.radians(angle_deg)
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = -np.radians(angle_deg)
    lx = dx * np.cos(angle) - dy * np.sin(angle)
    ly = dx * np.sin(angle) + dy * np.cos(angle)
    # Flat-top hexagon: |ly| <= side*sqrt(3)/2 and |lx| <= side - |ly|/sqrt(3)
    return abs(ly) <= side * np.sqrt(3) / 2 + 1e-9 and abs(lx) <= side - abs(ly) / np.sqrt(3) + 1e-9

def hexagons_overlap(v1, v2):
    """Check if two convex polygons (arrays of 6 vertices) overlap using separating axis theorem."""
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            # Edge normal
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Normal
            nx = -ey
            ny = ex
            # Project both polygons
            proj1 = np.dot(v1, [nx, ny])
            proj2 = np.dot(v2, [nx, ny])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    return True

def check_inside_outer(v_inner, outer_center, outer_side, outer_angle):
    """Check all vertices of inner hex are inside outer hex."""
    for i in range(6):
        px, py = v_inner[i]
        if not point_in_hex(px, py, outer_center[0], outer_center[1], outer_side, outer_angle):
            return False
    return True

def penalty(params, outer_side):
    """Penalty function: 0 if valid, positive if violations."""
    inner = []
    for i in range(N):
        x = params[3 * i]
        y = params[3 * i + 1]
        ang = params[3 * i + 2]
        inner.append((x, y, ang))
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    pen = 0.0
    
    # Check each inner hex is inside outer
    for i in range(N):
        x, y, ang = inner[i]
        v = hex_vertices(x, y, 1.0, ang)
        if not check_inside_outer(v, outer_center, outer_side, outer_angle):
            # Compute max violation
            for j in range(6):
                px, py = v[j]
                dx = px - outer_center[0]
                dy = py - outer_center[1]
                angle = -np.radians(outer_angle)
                lx = dx * np.cos(angle) - dy * np.sin(angle)
                ly = dx * np.sin(angle) + dy * np.cos(angle)
                # Violation: how much outside
                viol_y = max(0, abs(ly) - outer_side * np.sqrt(3) / 2)
                viol_x = max(0, abs(lx) - (outer_side - abs(ly) / np.sqrt(3)))
                pen += (viol_y + viol_x) ** 2
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            v1 = hex_vertices(x1, y1, 1.0, a1)
            v2 = hex_vertices(x2, y2, 1.0, a2)
            if hexagons_overlap(v1, v2):
                # Estimate penetration depth
                pen += 0.5
    
    return pen

def objective(params):
    """Minimize outer_side + penalty."""
    # outer_side is encoded as last element
    outer_side = params[-1]
    if outer_side < 1.0:
        return 1e6
    pen = penalty(params, outer_side)
    return outer_side + pen * 10.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_params = None
    best_score = float('inf')
    
    # Initial guess: hexagonal arrangement
    # 12 hexagons: center + 1 ring of 6 + partial second ring
    # Try a structured initial guess
    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        params = np.zeros(3 * N + 1)
        # Place in a rough hexagonal pattern
        positions = []
        # Center
        positions.append((0, 0))
        # Ring 1: 6 hexagons
        for k in range(6):
            ang = k * 60 * np.pi / 180
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        # Ring 2: 5 more hexagons
        for k in range(5):
            ang = (k * 72 + 30) * np.pi / 180
            positions.append((3.5 * np.cos(ang), 3.5 * np.sin(ang)))
        
        for i in range(N):
            if i < len(positions):
                params[3 * i] = positions[i][0] + rng.normal(0, 0.5)
                params[3 * i + 1] = positions[i][1] + rng.normal(0, 0.5)
            else:
                params[3 * i] = rng.uniform(-3, 3)
                params[3 * i + 1] = rng.uniform(-3, 3)
            params[3 * i + 2] = rng.uniform(0, 60)  # angle
        params[-1] = 4.0  # initial outer side
        return params
    
    # Try multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        params0 = make_initial_guess(restart)
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or res.fun < best_score:
                score = res.fun
                # Extract actual outer side (without penalty)
                actual_side = res.x[-1]
                pen = penalty(res.x, actual_side)
                if pen < 1e-6:
                    best_score = actual_side
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # If no valid solution found, try a simpler approach
    if best_params is None:
        # Fallback: use a known valid configuration
        # 12 hexagons in a hexagonal pattern with outer side ~4.5
        params = np.zeros(3 * N + 1)
        positions = []
        positions.append((0, 0))
        for k in range(6):
            ang = k * 60 * np.pi / 180
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for k in range(5):
            ang = (k * 72 + 30) * np.pi / 180
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        
        for i in range(N):
            params[3 * i] = positions[i][0]
            params[3 * i + 1] = positions[i][1]
            params[3 * i + 2] = 0.0
        params[-1] = 4.5
        
        # Verify and adjust
        outer_side = 4.5
        while penalty(params, outer_side) > 1e-6 and outer_side < 10:
            outer_side += 0.01
        params[-1] = outer_side
        best_params = params
        best_score = outer_side
    
    # Extract result
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        ang = best_params[3 * i + 2]
        inner.append([float(x), float(y), float(ang)])
    
    outer_side = float(best_params[-1])
    return inner, (0.0, 0.0), outer_side, 0.0

