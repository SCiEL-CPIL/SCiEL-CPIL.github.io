import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Vertices of unit hex at 0 degrees (pointy-top? flat-top?)
    # Let's use flat-top: vertices at angles 0, 60, 120, 180, 240, 300
    # Actually for a unit hexagon, the distance from center to vertex is 1.
    # Vertices at angles 0, 60, ... 300 degrees.
    angles = np.radians(np.arange(0, 360, 60)) + a
    vx = cx + np.cos(angles)
    vy = cy + np.sin(angles)
    return np.column_stack([vx, vy])

def _outer_hex_vertices(center, side, angle_deg):
    """Return 6 vertices of the outer regular hexagon."""
    a = np.radians(angle_deg)
    angles = np.radians(np.arange(0, 360, 60)) + a
    vx = center[0] + side * np.cos(angles)
    vy = center[1] + side * np.sin(angles)
    return np.column_stack([vx, vy])

def _point_in_hexagon(px, py, center, side, angle_deg):
    """Check if point (px, py) is inside the outer hexagon."""
    # Transform to hexagon's local frame
    dx = px - center[0]
    dy = py - center[1]
    a = -np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # In local frame, hexagon is centered at origin, side length s, flat-top (vertices at 0,60,...)
    # The hexagon can be defined by 6 half-planes.
    # For a regular hexagon with vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The edges are:
    # 1. Right edge: x = s (from angle -30 to 30? No, vertices at 0 and 60. Edge from (s,0) to (s/2, s*sqrt3/2))
    #    Actually, let's use the support function.
    #    The hexagon is the intersection of 6 half-planes.
    #    Normal directions are at angles 30, 90, 150, 210, 270, 330 degrees.
    #    The distance from center to each edge (apothem) is s * sqrt(3) / 2.
    #    So for each normal direction theta_k = 30 + 60*k degrees, the constraint is:
    #    lx * cos(theta_k) + ly * sin(theta_k) <= s * sqrt(3) / 2
    
    thetas = np.radians(30 + 60 * np.arange(6))
    proj = lx * np.cos(thetas) + ly * np.sin(thetas)
    max_proj = np.max(proj)
    apothem = side * SQRT3 / 2
    return max_proj <= apothem + 1e-9, max_proj - apothem

def _penalty_for_overlap(p1, p2, r1, r2):
    """Penalty for two unit hexagons overlapping. Simple distance-based penalty."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[2]
    dist = np.sqrt(dx*dx + dy*dy)
    # Minimum distance between centers of two non-overlapping unit hexagons is 2.
    # (They touch when centers are distance 2 apart, since inradius is sqrt(3)/2 ~ 0.866, 
    #  but actually for two unit hexagons to be disjoint, center distance must be >= 2)
    # Wait, two unit hexagons (side 1) touching flat-to-flat have center distance 2*sqrt(3)/2 = sqrt(3) ~ 1.732?
    # No, inradius of unit hex is sqrt(3)/2. Two hexagons touching flat-to-flat have center distance = 2 * inradius = sqrt(3).
    # But they could also touch vertex-to-vertex, distance = 2.
    # To be safe, require center distance >= 2.
    if dist < 2.0:
        return (2.0 - dist) ** 2
    return 0.0

def _fitness(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, side, container_angle]
    Total: 12*3 + 4 = 40 parameters.
    """
    # Unpack
    inner_params = params[:N*3]
    cx = params[N*3]
    cy = params[N*3+1]
    side = params[N*3+2]
    cont_angle = params[N*3+3]
    
    # Ensure side is positive
    side = max(side, 1.0)
    
    # Calculate overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        xi, yi, ai = inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2]
        for j in range(i+1, N):
            xj, yj, aj = inner_params[j*3], inner_params[j*3+1], inner_params[j*3+2]
            overlap_pen += _penalty_for_overlap((xi, yi), (xj, yj), 1, 1)
    
    # Calculate container constraint
    # For each inner hexagon, check all 6 vertices
    container_pen = 0.0
    for i in range(N):
        xi, yi, ai = inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2]
        verts = _hex_vertices(xi, yi, ai)
        for k in range(6):
            px, py = verts[k]
            inside, violation = _point_in_hexagon(px, py, (cx, cy), side, cont_angle)
            if violation > 0:
                container_pen += violation ** 2
    
    # Objective: minimize side + penalties
    # If all constraints are satisfied, we want to minimize side.
    # If not, we want to reduce penalties.
    # Use a large weight for penalties to ensure feasibility first.
    obj = side + 100.0 * (overlap_pen + container_pen)
    return obj

def _initial_guess():
    """Generate an initial guess using a hexagonal grid pattern."""
    params = np.zeros(N*3 + 4)
    # 3 rows of 4
    for row in range(3):
        y = (row - 1) * SQRT3
        for col in range(4):
            x = (col - 1.5) * 2.0
            idx = row * 4 + col
            params[idx*3] = x
            params[idx*3+1] = y
            params[idx*3+2] = 0.0
    
    # Container
    params[N*3] = 0.0   # cx
    params[N*3+1] = 0.0 # cy
    params[N*3+2] = 5.0 # side
    params[N*3+3] = 0.0 # container angle
    return params

def construct():
    # Try a few different container angles to see if rotation helps
    best_result = None
    best_obj = float('inf')
    
    # Differential evolution is robust but slow. Let's try Nelder-Mead with multiple starts.
    # Or use differential_evolution for the full 40D problem? That might be too slow.
    # Let's try Nelder-Mead with a good initial guess.
    
    x0 = _initial_guess()
    
    # Try optimizing with fixed container angle = 0 first
    try:
        res = minimize(_fitness, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # Try with container angle = 30
    x0_30 = _initial_guess()
    x0_30[N*3+3] = 30.0
    try:
        res = minimize(_fitness, x0_30, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # If optimization fails or doesn't improve much, fall back to a known good configuration
    # Let's try a simpler, more compact arrangement manually.
    # 12 hexagons: 2 rows of 6? No, 3x4 is better.
    # Let's try to fit in a smaller hexagon.
    
    if best_result is None or best_obj > 10.0:
        # Fallback: use the 3x4 grid with side=5
        inner = []
        for row in range(3):
            y = (row - 1) * SQRT3
            for col in range(4):
                x = (col - 1.5) * 2.0
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 5.0, 0.0
    
    # Extract results
    params = best_result
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append([x, y, a])
    
    cx = params[N*3]
    cy = params[N*3+1]
    side = params[N*3+2]
    cont_angle = params[N*3+3]
    
    return inner, (cx, cy), side, cont_angle
