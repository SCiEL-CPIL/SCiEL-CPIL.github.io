"""
Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
Uses scipy.optimize.least_squares with SDF-based residuals for collision and containment.
"""
import numpy as np
import time
import math
from scipy.optimize import least_squares, minimize

N = 12

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - SAFETY_MARGIN - _elapsed()

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _hex_normals(angle_deg):
    """Return the 6 outward unit normals of a regular hexagon with given rotation."""
    angle_rad = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0 + math.pi / 2.0
        normals.append((math.cos(theta), math.sin(theta)))
    return normals

def _sdf_hexagon(px, py, cx, cy, side, angle_deg):
    """
    Signed distance from point (px,py) to a regular hexagon centered at (cx,cy)
    with side length and rotation. Negative means inside.
    For a regular hexagon with side s, the apothem is s*sqrt(3)/2.
    The hexagon is the intersection of 6 half-planes defined by the normals.
    """
    apothem = side * math.sqrt(3) / 2.0
    normals = _hex_normals(angle_deg)
    dx = px - cx
    dy = py - cy
    
    # Find the maximum penetration (most positive dot product)
    max_dot = -float('inf')
    for nx, ny in normals:
        dot = dx * nx + dy * ny
        if dot > max_dot:
            max_dot = dot
    
    # SDF: max(0, max_dot - apothem) if outside, otherwise -min distance to edges
    # Actually for a convex polygon, SDF = max over all half-planes of (dot - apothem)
    # If all dots < apothem, point is inside, SDF = -(apothem - max_dot) ... no that's not right.
    # The SDF for a convex polygon defined by half-planes n_i . x <= d_i is:
    # SDF = max_i (n_i . x - d_i)
    # If this is negative, point is inside.
    
    # For a regular hexagon with side s, the half-plane constraint is:
    # n_i . (x - center) <= apothem
    # So SDF = max_i (n_i . (x - center) - apothem)
    
    return max_dot - apothem

def _min_distance_hex_hex(cx1, cy1, a1, cx2, cy2, a2, side1=1.0, side2=1.0):
    """
    Compute the minimum separation distance between two hexagons.
    Returns a positive value if separated, negative if overlapping.
    Uses SAT-like approach: check all 12 axes (6 from each hex).
    """
    verts1 = _hex_vertices(cx1, cy1, side1, a1)
    verts2 = _hex_vertices(cx2, cy2, side2, a2)
    
    # Get all axes to check
    axes = []
    for verts in (verts1, verts2):
        for i in range(6):
            j = (i + 1) % 6
            dx = verts[j][0] - verts[i][0]
            dy = verts[j][1] - verts[i][1]
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx/norm, ny/norm))
    
    min_gap = float('inf')
    for axis in axes:
        dots1 = [v[0]*axis[0] + v[1]*axis[1] for v in verts1]
        dots2 = [v[0]*axis[0] + v[1]*axis[1] for v in verts2]
        gap = min(dots2) - max(dots1)
        if gap > min_gap:
            min_gap = gap
    
    # If min_gap > 0, they are separated by min_gap
    # If min_gap < 0, they overlap by |min_gap|
    # But this is an approximation; the true SDF might be different
    # For our purposes, use this as a proxy
    return -min_gap  # Negative if overlapping, positive if separated

def _residuals(params, outer_angle):
    """
    Compute residuals for least_squares.
    params: flat array of length 3*N: [x0,y0,a0, x1,y1,a1, ...]
    Returns array of residuals.
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    residuals = []
    
    # Pairwise collision residuals
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            # Use SDF-based distance
            dist = _min_distance_hex_hex(x1, y1, a1, x2, y2, a2)
            # We want dist >= 0 (no overlap)
            # Residual is max(0, -dist) but we use a smooth approximation
            # Use softplus-like: -dist + small if dist < 0
            if dist < 0:
                residuals.append(-dist)
            else:
                residuals.append(0.0)
    
    # Containment residuals: each inner vertex must be inside outer hexagon
    outer_center = (0.0, 0.0)
    # We'll optimize the outer side implicitly; for now, use a fixed large outer
    # and add residuals for how much each inner vertex exceeds a target apothem
    # Actually, let's add a global residual for the outer side needed
    
    # For each inner hexagon, compute the maximum distance from outer center to any vertex
    # in the 6 normal directions of the outer hexagon
    outer_apothem_target = 0.0
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        normals = _hex_normals(outer_angle)
        for (px, py) in verts:
            for nx, ny in normals:
                d = (px - 0.0) * nx + (py - 0.0) * ny
                if d > outer_apothem_target:
                    outer_apothem_target = d
    
    # Add a residual proportional to the outer apothem needed
    # This encourages smaller outer hexagon
    residuals.append(outer_apothem_target * 10.0)
    
    return np.array(residuals)

def _outer_side_needed(inner, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    max_dist = 0.0
    normals = _hex_normals(outer_angle)
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for (px, py) in verts:
            for nx, ny in normals:
                d = (px - outer_center[0]) * nx + (py - outer_center[1]) * ny
                if d > max_dist:
                    max_dist = d
    if max_dist <= 0:
        return 0.0
    return max_dist / (math.sqrt(3) / 2.0)

def _is_valid(inner, outer_center, outer_side, outer_angle):
    """Check if configuration is valid: no overlaps and all contained."""
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            dist = _min_distance_hex_hex(x1, y1, a1, x2, y2, a2)
            if dist < -1e-6:
                return False
    # Check containment
    side_needed = _outer_side_needed(inner, outer_center, outer_angle)
    if side_needed > outer_side + 1e-6:
        return False
    return True

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Config 1: Ring
    ring = []
    for i in range(12):
        angle = i * 30.0
        r = 3.0
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        ring.append((x, y, 0.0))
    configs.append(ring)
    
    # Config 2: Two rows
    row = []
    for i in range(6):
        x = (i - 2.5) * 2.2
        y = -1.1
        row.append((x, y, 0.0))
    for i in range(6):
        x = (i - 2.5) * 2.2
        y = 1.1
        row.append((x, y, 0.0))
    configs.append(row)
    
    # Config 3: Hexagonal cluster
    hcp = []
    positions = [(0,0), (2.2,0), (-2.2,0), (1.1, 1.9), (-1.1, 1.9),
                 (1.1, -1.9), (-1.1, -1.9), (3.3, 0.95), (-3.3, 0.95),
                 (3.3, -0.95), (-3.3, -0.95), (0, 3.8)]
    for (x, y) in positions:
        hcp.append((x, y, 0.0))
    configs.append(hcp)
    
    # Config 4: Random with seed
    np.random.seed(42)
    rand = []
    for i in range(12):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        rand.append((x, y, a))
    configs.append(rand)
    
    # Config 5: Tighter ring
    ring2 = []
    for i in range(12):
        angle = i * 30.0
        r = 2.5
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        ring2.append((x, y, 0.0))
    configs.append(ring2)
    
    return configs

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_params = None
    best_score = float('inf')
    best_inner = None
    
    outer_center = (0.0, 0.0)
    outer_angles = [0.0, 30.0]
    
    configs = _generate_initial_configs()
    
    for oa in outer_angles:
        if _time_left() < 15.0:
            break
        
        for config in configs:
            if _time_left() < 10.0:
                break
            
            params = np.array([v for item in config for v in item])
            
            # Use least_squares
            try:
                result = least_squares(
                    _residuals,
                    params,
                    args=(oa,),
                    method='trf',
                    max_nfev=500,
                    xtol=1e-8,
                    ftol=1e-8,
                    gtol=1e-8
                )
                
                if result.success or result.cost < best_score:
                    inner = []
                    for i in range(N):
                        x = result.x[3*i]
                        y = result.x[3*i+1]
                        a = result.x[3*i+2]
                        inner.append([x, y, a])
                    
                    side = _outer_side_needed(inner, outer_center, oa)
                    
                    # Check validity
                    if _is_valid(inner, outer_center, side, oa):
                        if side < best_score:
                            best_score = side
                            best_params = result.x.copy()
                            best_inner = inner
            except Exception:
                pass
            
            # Also try L-BFGS-B with a smooth objective
            if _time_left() > 20.0:
                try:
                    def obj_smooth(p):
                        r = _residuals(p, oa)
                        return np.sum(r**2)
                    
                    result2 = minimize(
                        obj_smooth,
                        params,
                        method='L-BFGS-B',
                        maxiter=200,
                        options={'ftol': 1e-10, 'maxls': 50}
                    )
                    
                    if result2.success:
                        inner2 = []
                        for i in range(N):
                            x = result2.x[3*i]
                            y = result2.x[3*i+1]
                            a = result2.x[3*i+2]
                            inner2.append([x, y, a])
                        
                        side2 = _outer_side_needed(inner2, outer_center, oa)
                        
                        if _is_valid(inner2, outer_center, side2, oa):
                            if side2 < best_score:
                                best_score = side2
                                best_params = result2.x.copy()
                                best_inner = inner2
                except Exception:
                    pass
    
    # If no valid result, use fallback
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final polish: try to optimize outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa in [0.0, 15.0, 30.0, 45.0]:
        side = _outer_side_needed(best_inner, (0.0, 0.0), oa)
        if side < best_side:
            best_side = side
            best_oa = oa
    
    # Verify
    if not _is_valid(best_inner, (0.0, 0.0), best_side, best_oa):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, (0.0, 0.0), best_side, best_oa
