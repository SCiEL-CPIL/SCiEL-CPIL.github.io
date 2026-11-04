import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side s, rotated by angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3.0
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def hexagon_contains_point(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, angle angle_deg."""
    a = np.radians(angle_deg)
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    # Rotate by -a
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local frame, hexagon has vertices at angles 0, 60, ..., 300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the apothem is s * sqrt(3)/2
    # The distance from center to each edge (apothem) is s * sqrt(3)/2
    apothem = s * np.sqrt(3) / 2.0
    # The 6 edge normals point at angles 30, 90, 150, 210, 270, 330 degrees
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        nx, ny = np.cos(theta), np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-9:
            return False
    return True

def min_distance_to_edge(px, py, cx, cy, s, angle_deg):
    """Return the minimum signed distance from point to hexagon edges (positive if inside)."""
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    apothem = s * np.sqrt(3) / 2.0
    min_dist = np.inf
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        nx, ny = np.cos(theta), np.sin(theta)
        dist = lx * nx + ly * ny
        d = apothem - dist  # positive if inside
        if d < min_dist:
            min_dist = d
    return min_dist

def check_validity(inner, center, side, angle):
    """Check if all inner hexagons are fully inside the outer hexagon and don't overlap."""
    ox, oy = center
    # Check each inner hexagon is inside outer
    for (ix, iy, ia) in inner:
        # Check all 6 vertices of inner hexagon are inside outer
        iv = hex_vertices(ix, iy, 1.0, ia)
        for (vx, vy) in iv:
            if not hexagon_contains_point(vx, vy, ox, oy, side, angle):
                return False
    # Check no overlap between inner hexagons
    for i in range(N):
        for j in range(i + 1, N):
            ix1, iy1, ia1 = inner[i]
            ix2, iy2, ia2 = inner[j]
            # Two convex hexagons overlap if one vertex of one is inside the other
            # or edges intersect. Simple check: vertices inside.
            v1 = hex_vertices(ix1, iy1, 1.0, ia1)
            v2 = hex_vertices(ix2, iy2, 1.0, ia2)
            for (vx, vy) in v1:
                if hexagon_contains_point(vx, vy, ix2, iy2, 1.0, ia2):
                    return False
            for (vx, vy) in v2:
                if hexagon_contains_point(vx, vy, ix1, iy1, 1.0, ia1):
                    return False
    return True

def objective(params):
    """Minimize outer side length. params = [ox, oy, side, angle_deg, x0,y0,a0, x1,y1,a1, ...]"""
    ox, oy = params[0], params[1]
    side = params[2]
    angle = params[3]
    if side <= 0:
        return 1e6
    # Check inner hexagons inside outer
    penalty = 0.0
    for i in range(N):
        ix = params[4 + 3*i]
        iy = params[5 + 3*i]
        ia = params[6 + 3*i]
        # Vertices of inner hexagon
        iv = hex_vertices(ix, iy, 1.0, ia)
        for (vx, vy) in iv:
            d = min_distance_to_edge(vx, vy, ox, oy, side, angle)
            if d < 0:
                penalty += d * d  # squared penalty for violation
    # Check non-overlap: for each pair, ensure minimum distance between boundaries > 0
    # Approximate: check that vertices of each are outside the other
    for i in range(N):
        for j in range(i+1, N):
            ix1, iy1, ia1 = params[4+3*i], params[5+3*i], params[6+3*i]
            ix2, iy2, ia2 = params[4+3*j], params[5+3*j], params[6+3*j]
            v1 = hex_vertices(ix1, iy1, 1.0, ia1)
            v2 = hex_vertices(ix2, iy2, 1.0, ia2)
            for (vx, vy) in v1:
                d = min_distance_to_edge(vx, vy, ix2, iy2, 1.0, ia2)
                if d < 0:
                    penalty += d * d
            for (vx, vy) in v2:
                d = min_distance_to_edge(vx, vy, ix1, iy1, 1.0, ia1)
                if d < 0:
                    penalty += d * d
    return side + 100.0 * penalty

def construct():
    start_time = time.time()
    
    # Initial guess: try a few symmetric arrangements
    # Best known: 12 hexagons in a hexagon. Try starting from a grid-like arrangement.
    
    best_side = np.inf
    best_params = None
    
    # Try multiple random restarts
    rng = np.random.RandomState(42)
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate initial guess
        # Place hexagons in a rough hexagonal pattern
        if restart == 0:
            # Symmetric initial: 1 center + rings
            # For 12: 1 + 5 + 6? Or 2 rows of 6?
            # Try: 6 in a row, 6 in another row offset
            init = []
            # Row 1: 6 hexagons
            for k in range(6):
                x = (k - 2.5) * 2.0  # spacing ~2 for non-overlap
                y = 0.0
                init.append([x, y, 0.0])
            # Row 2: 6 hexagons offset
            for k in range(6):
                x = (k - 2.5) * 2.0 + 1.0
                y = np.sqrt(3)
                init.append([x, y, 0.0])
            ox, oy = 0.0, np.sqrt(3)/2
            side = 8.0
            angle = 0.0
        else:
            # Perturbed initial
            ox = rng.uniform(-2, 2)
            oy = rng.uniform(-2, 2)
            side = rng.uniform(5, 10)
            angle = rng.uniform(0, 60)
            init = []
            for i in range(N):
                x = rng.uniform(-4, 4)
                y = rng.uniform(-4, 4)
                a = rng.uniform(0, 60)
                init.append([x, y, a])
        
        params = np.array([ox, oy, side, angle] + [v for row in init for v in row])
        
        # Optimize
        res = minimize(objective, params, method='SLSQP', 
                      options={'maxiter': 200, 'ftol': 1e-8})
        
        if time.time() - start_time > DEADLINE:
            break
        
        if res.success or res.fun < best_side:
            p = res.x
            s = p[2]
            if s < best_side:
                best_side = s
                best_params = p.copy()
    
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    p = best_params
    ox, oy = p[0], p[1]
    side = p[2]
    angle = p[3]
    inner = []
    for i in range(N):
        x = p[4 + 3*i]
        y = p[5 + 3*i]
        a = p[6 + 3*i]
        inner.append([float(x), float(y), float(a)])
    
    return inner, (float(ox), float(oy)), float(side), float(angle)
