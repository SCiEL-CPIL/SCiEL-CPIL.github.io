"""Packing 11 unit hexagons into minimal outer hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees + rotation
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local frame
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # For a regular hexagon with 'flat top' orientation (vertex at angle 0),
    # the half-width is side, half-height is side*sqrt(3)/2
    # Actually for pointy-top (vertex at 90 deg), different.
    # Let's use the standard: vertices at 0,60,...,300 from center at distance=side
    # The hexagon is the intersection of 6 half-planes
    # For vertex at angle theta, the edge is perpendicular at distance side*sqrt(3)/2
    # Actually, apothem = side * sqrt(3)/2
    apothem = side * np.sqrt(3) / 2.0
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 (perpendicular to edges)
    # Wait, if vertices are at 0,60,120,..., the edges are between them.
    # Edge between vertex at 0 and vertex at 60: midpoint at 30 deg, distance from center = side*sqrt(3)/2
    # Normal direction is 30 deg
    edge_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    for ea in edge_angles:
        nx = np.cos(ea)
        ny = np.sin(ea)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-10:
            return False
    return True

def _hex_overlap_penalty(h1, h2, outer):
    """Penalty if hex1 and hex2 overlap, or if hex1 is not inside outer.
    h1, h2 are (cx, cy, angle_deg) with side=1.
    outer is (cx, cy, side, angle_deg).
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    ocx, ocy, oside, oangle = outer
    
    # Check if hex1 is inside outer: all 6 vertices of hex1 must be inside outer
    vx, vy = _hex_vertices(cx1, cy1, 1.0, a1)
    for i in range(6):
        if not _point_in_hex(vx[i], vy[i], ocx, ocy, oside, oangle):
            # Compute how far outside
            # Distance from vertex to the hexagon boundary
            # Approximate penalty
            dx = vx[i] - ocx
            dy = vy[i] - ocy
            angle = np.radians(oangle)
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            apothem = oside * np.sqrt(3) / 2.0
            edge_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
            max_violation = 0
            for ea in edge_angles:
                nx = np.cos(ea)
                ny = np.sin(ea)
                dist = lx * nx + ly * ny
                violation = dist - apothem
                if violation > max_violation:
                    max_violation = violation
            if max_violation > 0:
                return max_violation ** 2
    
    # Check overlap between hex1 and hex2
    # Two hexagons overlap if any vertex of one is inside the other
    # or if edges intersect. For simplicity, check vertices.
    vx1, vy1 = _hex_vertices(cx1, cy1, 1.0, a1)
    vx2, vy2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    for i in range(6):
        if _point_in_hex(vx1[i], vy1[i], cx2, cy2, 1.0, a2):
            return 1.0  # heavy penalty
        if _point_in_hex(vx2[i], vy2[i], cx1, cy1, 1.0, a1):
            return 1.0
    
    return 0.0

def _total_penalty(params):
    """Compute total penalty for a given parameter vector.
    params: [cx1, cy1, a1, cx2, cy2, a2, ..., ocx, ocy, oside, oangle]
    """
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        a = params[3*i+2]
        inner.append((cx, cy, a))
    
    ocx = params[3*n]
    ocy = params[3*n+1]
    oside = params[3*n+2]
    oangle = params[3*n+3]
    
    if oside <= 0:
        return 1e10
    
    outer = (ocx, ocy, oside, oangle)
    
    penalty = 0.0
    for i in range(n):
        penalty += _hex_overlap_penalty(inner[i], inner[i], outer)
        for j in range(i+1, n):
            penalty += _hex_overlap_penalty(inner[i], inner[j], outer)
    
    return penalty

def _objective(params):
    """Objective: outer side length + penalty for violations."""
    n = N
    oside = params[3*n+2]
    if oside <= 0.5:
        return 1e10
    penalty = _total_penalty(params)
    return oside + 100.0 * penalty

def _initial_guess():
    """Create initial guess: compact hexagonal cluster."""
    params = []
    # Honeycomb lattice: centers at triangular lattice points
    # Distance between adjacent centers = sqrt(3)
    d = np.sqrt(3)
    
    # Central hexagon at origin
    centers = [(0.0, 0.0)]
    # 6 neighbors at distance d, angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = k * np.pi / 3.0
        centers.append((d * np.cos(angle), d * np.sin(angle)))
    # 4 more from the next ring (distance 2*d at angles 0, 60, 120, 180)
    for k in range(4):
        angle = k * np.pi / 3.0
        centers.append((2*d * np.cos(angle), 2*d * np.sin(angle)))
    
    for cx, cy in centers:
        params.append(cx)
        params.append(cy)
        params.append(0.0)  # angle
    
    # Outer hexagon: large enough to contain all
    # The cluster extends roughly to distance 2*d + 1 from center
    max_r = max(np.sqrt(cx**2 + cy**2) for cx, cy in centers) + 2.0
    # Side length of outer hexagon: need apothem >= max_r
    # apothem = side * sqrt(3)/2, so side >= max_r * 2/sqrt(3)
    oside = max_r * 2.0 / np.sqrt(3) + 1.0
    params.append(0.0)  # ocx
    params.append(0.0)  # ocy
    params.append(oside)
    params.append(0.0)  # oangle
    
    return np.array(params)

def construct():
    deadline = time.time() + 140.0  # safety margin
    
    params0 = _initial_guess()
    
    best_params = params0.copy()
    best_obj = _objective(params0)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(_objective, params0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    if time.time() > deadline:
        # Return best found so far
        pass
    else:
        # Try a few more local optimizations from perturbed starts
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() > deadline:
                break
            perturbation = rng.normal(0, 0.5, size=len(params0))
            params_pert = params0 + perturbation
            try:
                result = minimize(_objective, params_pert, method='Nelder-Mead',
                                  options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                           'adaptive': True})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
    
    # Extract solution
    n = N
    inner = []
    for i in range(n):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(cx), float(cy), float(a)])
    
    ocx = float(best_params[3*n])
    ocy = float(best_params[3*n+1])
    oside = float(best_params[3*n+2])
    oangle = float(best_params[3*n+3])
    
    return inner, (ocx, ocy), oside, oangle

