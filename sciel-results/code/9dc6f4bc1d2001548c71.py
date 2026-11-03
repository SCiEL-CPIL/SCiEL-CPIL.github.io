import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angles = np.deg2rad(angle_deg) + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    theta = -np.deg2rad(angle_deg)
    x = dx * np.cos(theta) - dy * np.sin(theta)
    y = dx * np.sin(theta) + dy * np.cos(theta)
    # For a regular hexagon with side s, the inscribed circle radius is s*sqrt(3)/2
    # A point is inside if its distance from center in the hex metric <= s*sqrt(3)/2
    # Use the 6 half-plane constraints
    r = side * np.sqrt(3) / 2
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6  # angles of normals to edges
        if x * np.cos(a) + y * np.sin(a) > r:
            return False
    return True

def hexagon_overlap(v1, v2, gap=1.001):
    """Check if two hexagons (as vertex arrays) overlap with a small gap factor."""
    # Use separating axis theorem for convex polygons
    def project(points, axis):
        return np.dot(points, axis)
    
    def get_axes(vertices):
        axes = []
        for i in range(len(vertices)):
            edge = vertices[(i+1) % len(vertices)] - vertices[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    for axis in axes:
        p1 = project(v1, axis)
        p2 = project(v2, axis)
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        # Check for separation with gap
        if max1 + gap < min2 or max2 + gap < min1:
            return False
    return True

def min_outer_side(params, n_inner=11):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    # params: 3*n_inner + 3 = [x1,y1,a1, x2,y2,a2, ..., cx, cy, outer_angle]
    positions = params[:3*n_inner].reshape(n_inner, 3)
    outer_cx = params[3*n_inner]
    outer_cy = params[3*n_inner+1]
    outer_angle = params[3*n_inner+2]
    
    # Get inner hexagon vertices
    inner_hexes = []
    for i in range(n_inner):
        x, y, ang = positions[i]
        verts = hex_vertices(x, y, 1.0, ang)
        inner_hexes.append(verts)
    
    # Check disjointness
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            if hexagon_overlap(inner_hexes[i], inner_hexes[j]):
                return 1e6  # penalty for overlap
    
    # Find minimum outer side length
    # For each inner hexagon vertex, find the required outer side
    # The outer hexagon must contain all vertices
    # For a regular hexagon centered at (cx,cy) with side S and angle theta,
    # a point (px,py) is inside if for all 6 edge normals n_i:
    #   dot((px-cx, py-cy), n_i) <= S * sqrt(3) / 2
    
    best_S = 0.0
    for verts in inner_hexes:
        for v in verts:
            dx = v[0] - outer_cx
            dy = v[1] - outer_cy
            theta = -np.deg2rad(outer_angle)
            x = dx * np.cos(theta) - dy * np.sin(theta)
            y = dx * np.sin(theta) + dy * np.cos(theta)
            for i in range(6):
                a = np.pi / 3 * i + np.pi / 6
                val = x * np.cos(a) + y * np.sin(a)
                required_S = val / (np.sqrt(3) / 2)
                if required_S > best_S:
                    best_S = required_S
    
    return max(best_S, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Initial guess: hexagonal cluster pattern
    # Center + 6 around + 4 in gaps
    # For unit hexagons touching, center-to-center distance is 2*sqrt(3)
    d = 2.0 * np.sqrt(3)
    
    # 11 hexagons in a compact arrangement
    # Row pattern: 1, 2, 3, 2, 1 or use hexagonal lattice
    # Let's use a 3-4-3 or similar compact pattern
    
    # Hexagonal lattice positions for 11 hexagons:
    # Center at origin, then 6 at distance d, then 4 more
    positions = np.zeros((N, 3))
    positions[0] = [0, 0, 0]  # center
    
    # 6 surrounding
    for i in range(6):
        angle = np.pi / 3 * i
        positions[i+1] = [d * np.cos(angle), d * np.sin(angle), 0]
    
    # 4 more in the gaps (at distance 2*d from center, in directions between)
    for i in range(4):
        angle = np.pi / 3 * (2*i + 1) + np.pi/6  # offset
        # Actually let's place them at the "corner" positions
        # For a hexagonal arrangement of 19, we'd have 1+6+12. 
        # For 11, let's try: center, 6 ring, then 4 at specific positions
        # Better: use a 2-3-2-1-1-1-1 or just optimize from a reasonable start
    
    # Let me try a different initial: linear-ish but bent
    # Actually, let's just start with a rough hexagonal cluster and let optimization fix it
    
    # Better initial: place in a hexagonal grid pattern
    # Hex grid with axial coordinates
    hex_coords = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,-1), (-2,1)]
    # Convert axial to pixel for pointy-top hexagons
    for i, (q, r) in enumerate(hex_coords):
        x = d * (q + r/2)
        y = d * np.sqrt(3)/2 * r
        positions[i] = [x, y, 0]
    
    # Initial outer parameters
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    best_params = np.concatenate([positions.flatten(), [outer_cx, outer_cy, outer_angle]])
    best_score = min_outer_side(best_params)
    
    # Use differential evolution for global optimization
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-5, 5))  # outer cx
    bounds.append((-5, 5))  # outer cy
    bounds.append((-180, 180))  # outer angle
    
    # Try differential evolution with time limit
    def objective(params):
        return min_outer_side(params)
    
    # First try a quick local optimization from the initial guess
    try:
        result = minimize(objective, best_params, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except:
        pass
    
    if time.time() > deadline:
        return _make_result(best_params)
    
    # Try differential evolution with limited iterations
    try:
        de_result = differential_evolution(objective, bounds, maxiter=50, popsize=10,
                                           tol=1e-4, seed=42, polish=True,
                                           init='sobol')
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_params = de_result.x
    except:
        pass
    
    if time.time() > deadline:
        return _make_result(best_params)
    
    # Polish with Nelder-Mead
    try:
        result = minimize(objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except:
        pass
    
    return _make_result(best_params)

def _make_result(params):
    n_inner = 11
    positions = params[:3*n_inner].reshape(n_inner, 3)
    outer_cx = params[3*n_inner]
    outer_cy = params[3*n_inner+1]
    outer_angle = params[3*n_inner+2]
    
    inner = [[float(x), float(y), float(a)] for x, y, a in positions]
    side = float(min_outer_side(params))
    
    return inner, (float(outer_cx), float(outer_cy)), side, float(outer_angle)
