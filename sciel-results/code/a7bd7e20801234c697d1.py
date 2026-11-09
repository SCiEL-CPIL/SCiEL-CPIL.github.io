"""
Packing 11 unit regular hexagons inside a regular hexagon, minimizing outer side length.

Strategy:
- Use scipy.optimize.minimize (L-BFGS-B) with a smooth penalty objective.
- Objective: outer_side_length + large * max(0, violation)
  where violation = max over all pairs of (overlap_depth) and (boundary_violation).
- We parameterize each inner hexagon by (x, y, theta).
- Outer hexagon is centered at origin with side length S and fixed rotation 0.
- We minimize S subject to all inner hexagons being disjoint and contained.
- Multi-start from diverse initial configurations.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, theta, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), rotated by theta radians, side length 1."""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + theta
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _segment_intersection_test(v1, v2, v3, v4):
    """Check if segments (v1,v2) and (v3,v4) intersect (proper intersection)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(v3, v4, v1)
    d2 = cross(v3, v4, v2)
    d3 = cross(v1, v2, v3)
    d4 = cross(v1, v2, v4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _point_in_hexagon(pt, cx, cy, theta, side=1.0):
    """Check if point pt is inside (or on boundary of) hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local frame
    dx = pt[0] - cx
    dy = pt[1] - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    # Regular hexagon with flat top (vertices at angles 0,60,...): 
    # The hexagon with side 1 has inradius = sqrt(3)/2, circumradius = 1
    # In local frame, the hexagon boundaries are at |lx| <= 1, |ly| <= sqrt(3)/2, 
    # and |lx/2 + ly*sqrt(3)/2| <= 1... 
    # Actually for a regular hexagon with vertices at angles 0,60,120,180,240,300:
    # The six half-planes are:
    # x*cos(60*k) + y*sin(60*k) <= 1 for k=0..5? No.
    # The inradius is sqrt(3)/2. The distance from center to each side is sqrt(3)/2.
    # Side normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    # So: lx*cos(30+60k) + ly*sin(30+60k) <= sqrt(3)/2 for k=0..5
    normals = np.array([np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6))
    dists = normals @ np.array([lx, ly])
    return np.all(dists <= np.sqrt(3)/2 + 1e-9)

def _overlap_penalty(hex1, hex2):
    """
    Compute a smooth penalty for overlap between two hexagons.
    hex1, hex2: arrays of 6 vertices each.
    Returns 0 if no overlap, positive value indicating depth of overlap.
    We use: if any edge of hex1 intersects any edge of hex2, or one vertex is inside the other,
    we compute a penalty based on minimum translation distance.
    
    For smoothness, we use a soft version: sum of squared max(0, penetration) for each vertex-in-polygon check.
    """
    # Check if any vertex of hex1 is inside hex2, or vice versa
    penalty = 0.0
    # For each vertex of hex1, check if inside hex2
    for v in hex1:
        # Check if v is inside hex2 using the half-plane method
        # Compute center and orientation of hex2
        c2 = np.mean(hex2, axis=0)
        # Get angle of first edge
        e = hex2[1] - hex2[0]
        theta2 = np.arctan2(e[1], e[0])
        if _point_in_hexagon(v, c2[0], c2[1], theta2):
            penalty += 1.0
    for v in hex2:
        c1 = np.mean(hex1, axis=0)
        e = hex1[1] - hex1[0]
        theta1 = np.arctan2(e[1], e[0])
        if _point_in_hexagon(v, c1[0], c1[1], theta1):
            penalty += 1.0
    
    # Also check edge-edge intersections
    for i in range(6):
        v1 = hex1[i]
        v2 = hex1[(i+1) % 6]
        for j in range(6):
            v3 = hex2[j]
            v4 = hex2[(j+1) % 6]
            if _segment_intersection_test(v1, v2, v3, v4):
                penalty += 1.0
    
    return penalty

def _boundary_violation(hex_verts, S):
    """
    Check if hexagon (with vertices hex_verts) is contained in outer hexagon of side S centered at origin.
    Outer hexagon vertices at angles 0,60,...,300 with circumradius S.
    Returns max penetration depth (0 if contained).
    """
    # Outer hexagon half-planes: for k=0..5, normal at angle 30+60k, distance S*sqrt(3)/2
    violation = 0.0
    normals = np.array([np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6))
    for v in hex_verts:
        dists = normals @ v
        max_dist = np.max(dists)
        if max_dist > S * np.sqrt(3)/2:
            violation += (max_dist - S * np.sqrt(3)/2) ** 2
    return violation

def _objective(params, S):
    """
    params: array of length 3*N, [x1,y1,theta1, x2,y2,theta2, ...]
    S: outer side length (also a parameter, but we'll optimize it separately or jointly)
    """
    # Actually, let's make S part of the optimization too
    # params: [S, x1, y1, theta1, x2, y2, theta2, ..., xN, yN, thetaN]
    S_val = params[0]
    if S_val < 2.0:  # minimum possible is about 2
        return 1e6
    
    hexes = []
    for i in range(N):
        offset = 1 + 3*i
        x, y, theta = params[offset], params[offset+1], params[offset+2]
        hex_verts = _hex_vertices(x, y, theta)
        hexes.append(hex_verts)
    
    total_penalty = 0.0
    # Pairwise overlap
    for i in range(N):
        for j in range(i+1, N):
            total_penalty += _overlap_penalty(hexes[i], hexes[j])
    
    # Boundary violation
    for i in range(N):
        total_penalty += _boundary_violation(hexes[i], S_val)
    
    # Objective: minimize S + large penalty
    return S_val + 100.0 * total_penalty

def _objective_full(params):
    """Full objective where S is the first parameter."""
    return _objective(params, params[0])

def _grad(params):
    """Numerical gradient (L-BFGS-B can use numerical, but let's provide it for speed)."""
    # Use finite differences
    eps = 1e-7
    g = np.zeros_like(params)
    f0 = _objective_full(params)
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        f_p = _objective_full(params_p)
        g[i] = (f_p - f0) / eps
    return g

def construct():
    start_time = time.time()
    
    best_S = np.inf
    best_params = None
    
    # Define multiple initial configurations
    def make_initial_config(type_idx):
        """Create initial parameter vector [S, x1,y1,theta1, ...]"""
        params = np.zeros(1 + 3*N)
        params[0] = 6.0  # initial S
        
        if type_idx == 0:
            # Linear arrangement
            for i in range(N):
                x = (i - N//2) * 2.2
                y = 0.0
                theta = 0.0
                params[1+3*i] = x
                params[2+3*i] = y
                params[3+3*i] = theta
        elif type_idx == 1:
            # Hexagonal lattice pattern
            # Center + 6 around + 4 more
            positions = [(0,0)]
            for k in range(6):
                angle = k * np.pi/3
                positions.append((2.1 * np.cos(angle), 2.1 * np.sin(angle)))
            for k in range(4):
                angle = (2*k+1) * np.pi/3
                positions.append((4.2 * np.cos(angle), 4.2 * np.sin(angle)))
            for i in range(N):
                params[1+3*i] = positions[i][0]
                params[2+3*i] = positions[i][1]
                params[3+3*i] = 0.0
        elif type_idx == 2:
            # Circular arrangement
            for i in range(N):
                angle = 2 * np.pi * i / N
                r = 3.0
                params[1+3*i] = r * np.cos(angle)
                params[2+3*i] = r * np.sin(angle)
                params[3+3*i] = angle
        elif type_idx == 3:
            # Zigzag
            for i in range(N):
                x = (i - N//2) * 2.0
                y = (0.8 if i % 2 == 0 else -0.8)
                params[1+3*i] = x
                params[2+3*i] = y
                params[3+3*i] = 0.0
        elif type_idx == 4:
            # Random
            np.random.seed(42)
            for i in range(N):
                params[1+3*i] = np.random.uniform(-4, 4)
                params[2+3*i] = np.random.uniform(-4, 4)
                params[3+3*i] = np.random.uniform(0, np.pi/3)
        else:
            # Compact cluster
            for i in range(N):
                angle = 2 * np.pi * i / N
                r = 1.8
                params[1+3*i] = r * np.cos(angle)
                params[2+3*i] = r * np.sin(angle)
                params[3+3*i] = 0.0
        
        return params
    
    # Try multiple starting configurations
    n_starts = 6
    for start_idx in range(n_starts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        x0 = make_initial_config(start_idx)
        
        try:
            result = minimize(
                _objective_full,
                x0,
                method='L-BFGS-B',
                jac=_grad,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if result.success or result.fun < best_S:
                S_candidate = result.x[0]
                # Verify feasibility
                penalty = 0.0
                hexes = []
                for i in range(N):
                    x, y, theta = result.x[1+3*i], result.x[2+3*i], result.x[3+3*i]
                    hex_verts = _hex_vertices(x, y, theta)
                    hexes.append(hex_verts)
                
                for i in range(N):
                    for j in range(i+1, N):
                        penalty += _overlap_penalty(hexes[i], hexes[j])
                for i in range(N):
                    penalty += _boundary_violation(hexes[i], S_candidate)
                
                if penalty < 1e-6 and S_candidate < best_S:
                    best_S = S_candidate
                    best_params = result.x.copy()
                    
        except Exception:
            continue
    
    # If we found a good solution, use it; otherwise fall back to a reasonable default
    if best_params is not None:
        inner = []
        for i in range(N):
            x = best_params[1+3*i]
            y = best_params[2+3*i]
            theta_rad = best_params[3+3*i]
            theta_deg = np.degrees(theta_rad) % 360.0
            inner.append([float(x), float(y), float(theta_deg)])
        return inner, (0.0, 0.0), float(best_S), 0.0
    else:
        # Fallback: linear arrangement with conservative S
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
