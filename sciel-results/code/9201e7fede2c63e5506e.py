"""
Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
Uses SAT-based collision detection with diverse initial configurations and Nelder-Mead optimization.
"""
import numpy as np
import time
import math

N = 12

# Deadline guard
_START_TIME = time.time()
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to spare

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

def _project_on_axis(verts, axis):
    """Project vertices onto axis, return (min, max)."""
    dots = [v[0]*axis[0] + v[1]*axis[1] for v in verts]
    return min(dots), max(dots)

def _sat_overlap(verts1, verts2):
    """
    Separating Axis Theorem: check if two convex polygons overlap.
    Returns True if they overlap (i.e., no separating axis found).
    """
    edges = []
    for verts in (verts1, verts2):
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            dx = verts[j][0] - verts[i][0]
            dy = verts[j][1] - verts[i][1]
            # Normal (perpendicular)
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            edges.append((nx/norm, ny/norm))
    
    for axis in edges:
        min1, max1 = _project_on_axis(verts1, axis)
        min2, max2 = _project_on_axis(verts2, axis)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False  # Separating axis found, no overlap
    return True  # Overlap

def _hexagon_overlap(verts1, verts2):
    """Check if two hexagons overlap using SAT."""
    return _sat_overlap(verts1, verts2)

def _max_support(verts, center):
    """Find the maximum distance from center to any vertex."""
    return max(math.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2) for v in verts)

def _bounding_side(verts, outer_center, outer_side, outer_angle):
    """
    Check if all inner hexagon vertices are inside the outer hexagon.
    Returns the minimum outer side length needed (if outer is centered at outer_center with given angle).
    Actually, we compute: for a given outer hexagon (center, side, angle), check containment.
    If not contained, return the required side length to contain.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    # For each inner vertex, find the minimum scaling of outer hex needed to contain it
    # A point (px, py) is inside a regular hexagon centered at (cx, cy) with side s and angle a
    # if its distance in each of the 6 normal directions is <= s * cos(pi/6) = s * sqrt(3)/2
    # Actually for a regular hexagon with side s, the apothem is s*sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes.
    
    cx, cy = outer_center
    angle_rad = math.radians(outer_angle)
    apothem = outer_side * math.sqrt(3) / 2.0
    
    # The 6 outward normals of the outer hexagon
    max_dist = 0.0
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0 + math.pi / 2.0  # normals point outward
        nx, ny = math.cos(theta), math.sin(theta)
        # For each inner vertex, compute signed distance along this normal
        for (px, py) in verts:
            d = (px - cx) * nx + (py - cy) * ny
            if d > max_dist:
                max_dist = d
    
    # Required apothem is max_dist, so required side = max_dist / (sqrt(3)/2)
    if max_dist <= 0:
        return 0.0
    return max_dist / (math.sqrt(3) / 2.0)

def _outer_side_needed(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum outer side length such that all inner hexagons fit inside.
    inner_hexes: list of (cx, cy, angle_deg)
    outer_center: (cx, cy)
    outer_angle: rotation of outer hexagon in degrees
    """
    max_side = 0.0
    for (cx, cy, ang) in inner_hexes:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        side = _bounding_side(verts, outer_center, 1.0, outer_angle)
        if side > max_side:
            max_side = side
    return max_side

def _objective(params, outer_center, outer_angle):
    """
    Objective: minimize outer side length, with penalty for overlaps.
    params: flat array of length 3*N: [x0,y0,a0, x1,y1,a1, ...]
    Returns (outer_side, overlap_penalty)
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    # Check overlaps
    overlap_count = 0
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _hexagon_overlap(v1, v2):
                overlap_count += 1
    
    # Compute outer side needed
    side = _outer_side_needed(inner, outer_center, outer_angle)
    
    # Penalty: large if overlaps exist
    penalty = 100.0 * overlap_count
    return side + penalty

def _nelder_mead(f, x0, max_iter=200, tol=1e-8, time_check=None):
    """
    Nelder-Mead simplex optimization.
    f: objective function
    x0: initial point (numpy array)
    Returns best point and value.
    """
    n = len(x0)
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 if xi[i] == 0 else 0.05 * xi[i]
        simplex.append(xi)
    
    vals = [f(simplex[i]) for i in range(n+1)]
    best_x = simplex[0].copy()
    best_val = vals[0]
    
    for i in range(n+1):
        if vals[i] < best_val:
            best_val = vals[i]
            best_x = simplex[i].copy()
    
    alpha, gamma, rho = 1.0, 2.0, 0.5
    iter_count = 0
    
    while iter_count < max_iter:
        if time_check is not None and time_check():
            break
        if best_val < tol:
            break
        
        # Sort simplex
        order = np.argsort(vals)
        simplex = [simplex[i].copy() for i in order]
        vals = [vals[i] for i in order]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < vals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            # Contraction
            if fr < vals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        vals[i] = f(simplex[i])
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
        
        # Update best
        for i in range(n+1):
            if vals[i] < best_val:
                best_val = vals[i]
                best_x = simplex[i].copy()
        
        iter_count += 1
    
    return best_x, best_val

def _generate_initial_configs():
    """Generate diverse initial configurations for the 12 inner hexagons."""
    configs = []
    
    # Config 1: Ring arrangement
    ring = []
    for i in range(12):
        angle = i * 30.0
        r = 3.5
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        ring.append((x, y, 0.0))
    configs.append(ring)
    
    # Config 2: Two rows of 6
    row1 = []
    for i in range(6):
        x = (i - 2.5) * 2.1
        y = -1.2
        row1.append((x, y, 0.0))
    row2 = []
    for i in range(6):
        x = (i - 2.5) * 2.1
        y = 1.2
        row2.append((x, y, 0.0))
    configs.append(row1 + row2)
    
    # Config 3: Rotated grid (3x4 with offsets)
    grid = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 2.1
            y = (row - 1.0) * 1.85
            if row % 2 == 1:
                x += 1.05
            grid.append((x, y, 0.0))
    configs.append(grid)
    
    # Config 4: Hexagonal close-packed cluster
    hcp = []
    positions = [(0,0), (2.1,0), (-2.1,0), (1.05, 1.82), (-1.05, 1.82),
                 (1.05, -1.82), (-1.05, -1.82), (3.15, 0.91), (-3.15, 0.91),
                 (3.15, -0.91), (-3.15, -0.91), (0, 3.64)]
    for (x, y) in positions:
        hcp.append((x, y, 0.0))
    configs.append(hcp)
    
    # Config 5: Random
    np.random.seed(42)
    rand = []
    for i in range(12):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        a = np.random.uniform(0, 60)
        rand.append((x, y, a))
    configs.append(rand)
    
    # Config 6: Honeycomb-like with slight perturbation
    honey = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 2.1 + 0.1 * np.random.randn()
            y = (row - 1.0) * 1.85 + 0.1 * np.random.randn()
            if row % 2 == 1:
                x += 1.05
            honey.append((x, y, 0.0))
    configs.append(honey)
    
    return configs

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_params = None
    best_score = float('inf')
    best_inner = None
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0  # Try a few outer angles
    
    configs = _generate_initial_configs()
    
    # Try different outer angles
    outer_angles = [0.0, 30.0]
    
    for oa in outer_angles:
        if _time_left() < 10.0:
            break
        
        for config in configs:
            if _time_left() < 10.0:
                break
            
            # Flatten config to params
            params = np.array([v for item in config for v in item])
            
            def obj(p):
                return _objective(p, outer_center, oa)
            
            # Run Nelder-Mead
            result_params, result_val = _nelder_mead(
                obj, params, max_iter=150, tol=1e-7,
                time_check=lambda: _time_left() < 5.0
            )
            
            if result_val < best_score:
                best_score = result_val
                best_params = result_params.copy()
            
            # Also try with slightly perturbed initial points
            if _time_left() > 20.0:
                for trial in range(2):
                    if _time_left() < 5.0:
                        break
                    perturbed = params + np.random.randn(len(params)) * 0.5
                    result_params2, result_val2 = _nelder_mead(
                        obj, perturbed, max_iter=100, tol=1e-7,
                        time_check=lambda: _time_left() < 5.0
                    )
                    if result_val2 < best_score:
                        best_score = result_val2
                        best_params = result_params2.copy()
    
    # If no valid result found, use a safe default
    if best_params is None:
        # Fallback: spread them out
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract inner hexagons from best params
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    # Compute the actual outer side needed
    # Find the outer angle that minimizes side
    best_side = float('inf')
    best_oa = 0.0
    for oa in [0.0, 30.0, 15.0, 45.0]:
        side = _outer_side_needed(inner, (0.0, 0.0), oa)
        if side < best_side:
            best_side = side
            best_oa = oa
    
    # Verify no overlaps
    has_overlap = False
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _hexagon_overlap(v1, v2):
                has_overlap = True
    
    if has_overlap:
        # If overlaps remain, use a conservative fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (0.0, 0.0), best_side, best_oa
