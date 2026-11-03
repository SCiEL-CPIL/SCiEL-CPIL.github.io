"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (pointy-top orientation at 0 deg)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _project_range(verts, axis):
    """Min and max projection of polygon vertices onto a unit axis."""
    dots = verts @ axis
    return dots.min(), dots.max()

def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons overlap (interior intersection)."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = _project_range(v1, normal)
            r2min, r2max = _project_range(v2, normal)
            if r1max < r2min + tol or r2max < r1min + tol:
                return False
    return True

def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True

def _inner_fits_outer(inner_verts, outer_verts, tol=1e-9):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_convex(v[0], v[1], outer_verts, tol):
            return False
    return True

def _feasible(outer_side, placements, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Check if given placements fit inside outer hexagon and don't overlap each other."""
    ox, oy = outer_center
    outer_verts = _outer_vertices(ox, oy, outer_side, outer_angle)
    
    inner_verts_list = []
    for (ix, iy, iangle) in placements:
        inner_verts_list.append(_hex_vertices(ix, iy, iangle, 1.0))
    
    for iv in inner_verts_list:
        if not _inner_fits_outer(iv, outer_verts, tol=1e-9):
            return False
    
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], tol=1e-9):
                return False
    
    return True

def _distance_to_boundary(point, outer_verts):
    """Compute the minimum distance from point to the boundary of outer hexagon."""
    min_dist = float('inf')
    n = len(outer_verts)
    for i in range(n):
        j = (i + 1) % n
        ax, ay = outer_verts[i]
        bx, by = outer_verts[j]
        # Distance from point to line segment
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            dist = math.sqrt((point[0] - ax)**2 + (point[1] - ay)**2)
        else:
            t = ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_sq
            t = max(0, min(1, t))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dist = math.sqrt((point[0] - proj_x)**2 + (point[1] - proj_y)**2)
        min_dist = min(min_dist, dist)
    return min_dist

def _packing_objective(x, outer_side, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Objective function for optimization.
    x = [x1, y1, theta1, x2, y2, theta2, ..., x12, y12, theta12]
    Returns a scalar to be minimized. We use a penalty-based approach:
    - Penalties for overlaps between inner hexagons
    - Penalties for inner hexagons extending outside the outer hexagon
    """
    n_hex = len(x) // 3
    penalty = 0.0
    
    outer_verts = _outer_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    inner_verts_list = []
    for i in range(n_hex):
        cx = x[3*i]
        cy = x[3*i + 1]
        angle = x[3*i + 2]
        verts = _hex_vertices(cx, cy, angle, 1.0)
        inner_verts_list.append(verts)
    
    # Check if each inner hexagon fits inside outer
    for iv in inner_verts_list:
        for v in iv:
            if not _point_in_convex(v[0], v[1], outer_verts, tol=1e-9):
                # Compute how far outside
                dist = _distance_to_boundary((v[0], v[1]), outer_verts)
                # If point is outside, distance to boundary is the violation
                # Actually, we need to check if it's outside first
                # Let's use a simpler penalty: if not inside, add penalty
                penalty += 100.0
    
    # Check pairwise overlaps
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], tol=1e-9):
                # Compute overlap depth (approximate)
                penalty += 50.0
    
    return penalty

def _initial_guess(outer_side):
    """Generate an initial guess for the optimizer."""
    # Start with a hexagonal packing pattern
    # For 12 hexagons, we can use a 3-4-3 or 2-4-4-2 arrangement
    # Let's use a simple grid-like arrangement
    sqrt3 = math.sqrt(3)
    
    # Hexagonal close packing: each hexagon has 6 neighbors
    # For 12 hexagons, a good arrangement is:
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons (offset)
    # Row 3: 4 hexagons
    # But that's 12, so let's do 3 rows of 4
    
    x = np.zeros(3 * N)
    
    # Hexagonal packing: horizontal spacing = 2*side, vertical spacing = sqrt(3)*side
    # For flat-top hexagons, the horizontal distance between centers is 2*side
    # and vertical is sqrt(3)*side, with alternating rows offset by side
    
    # Let's use a more compact arrangement
    # 12 hexagons in a hexagonal cluster: center + 6 + 5 (partial ring)
    # Or just a 3x4 grid with hexagonal offsets
    
    # Simple approach: 3 rows of 4
    row_spacing = sqrt3 * 1.0  # vertical distance between rows
    col_spacing = 2.0 * 1.0    # horizontal distance between columns
    
    # Center the arrangement
    total_width = 3 * col_spacing
    total_height = 2 * row_spacing
    x_offset = -total_width / 2
    y_offset = -total_height / 2
    
    idx = 0
    for row in range(3):
        for col in range(4):
            cx = x_offset + col * col_spacing
            cy = y_offset + row * row_spacing
            # Offset alternating rows
            if row % 2 == 1:
                cx += col_spacing / 2
            x[3*idx] = cx
            x[3*idx + 1] = cy
            x[3*idx + 2] = 0.0  # No rotation initially
            idx += 1
    
    return x

def _optimize_packing(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, max_iter=500, time_budget=30.0):
    """
    Use SLSQP to optimize the packing.
    Returns the optimized positions or None if it fails.
    """
    x0 = _initial_guess(outer_side)
    
    # Bounds: positions should be within the outer hexagon
    # For a hexagon of side S, the extent is roughly -S to S in x and -S*sqrt(3)/2 to S*sqrt(3)/2 in y
    # But with rotation, it's more complex. Use generous bounds.
    max_coord = outer_side * 1.5
    bounds = []
    for i in range(N):
        bounds.append((-max_coord, max_coord))  # x
        bounds.append((-max_coord, max_coord))  # y
        bounds.append((0, 2*math.pi))           # angle
    
    # Use SLSQP with the penalty objective
    result = minimize(
        _packing_objective,
        x0,
        args=(outer_side, outer_center, outer_angle),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-8, 'disp': False}
    )
    
    if result.success or result.fun < 1e-6:
        # Extract placements
        placements = []
        for i in range(N):
            placements.append((result.x[3*i], result.x[3*i + 1], math.degrees(result.x[3*i + 2]) % 360.0))
        return placements
    
    return None

def _grid_search(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, time_budget=20.0):
    """
    Try to find 12 non-overlapping unit hexagons inside a hexagon of given side.
    Uses a combination of grid search and local optimization.
    """
    start_time = time.time()
    
    # First, try the initial guess and see if it's feasible
    x0 = _initial_guess(outer_side)
    placements = []
    for i in range(N):
        placements.append((x0[3*i], x0[3*i + 1], math.degrees(x0[3*i + 2]) % 360.0))
    
    if _feasible(outer_side, placements, outer_center, outer_angle):
        return placements
    
    # Try local optimization from the initial guess
    if time.time() - start_time < time_budget:
        result = _optimize_packing(outer_side, outer_center, outer_angle, 
                                   max_iter=200, time_budget=time_budget - (time.time() - start_time))
        if result is not None and _feasible(outer_side, result, outer_center, outer_angle):
            return result
    
    # Try a few random initial guesses
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Generate random initial positions
        x0 = np.zeros(3 * N)
        for i in range(N):
            # Random position within the hexagon
            r = outer_side * math.sqrt(np.random.random())
            theta = 2 * math.pi * np.random.random()
            x0[3*i] = r * math.cos(theta)
            x0[3*i + 1] = r * math.sin(theta)
            x0[3*i + 2] = 2 * math.pi * np.random.random()
        
        # Optimize from this random start
        max_iter = 100
        result = minimize(
            _packing_objective,
            x0,
            args=(outer_side, outer_center, outer_angle),
            method='SLSQP',
            bounds=[(-outer_side*1.5, outer_side*1.5), (-outer_side*1.5, outer_side*1.5), (0, 2*math.pi)] * N,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.fun < 1e-6:
            placements = []
            for i in range(N):
                placements.append((result.x[3*i], result.x[3*i + 1], math.degrees(result.x[3*i + 2]) % 360.0))
            if _feasible(outer_side, placements, outer_center, outer_angle):
                return placements
    
    return None

def _bisection_search(lo, hi, outer_center=(0.0, 0.0), outer_angle=0.0, time_budget=100.0):
    """
    Bisection search for the minimum outer side length.
    """
    start_time = time.time()
    best_placements = None
    best_side = hi
    
    # First, verify that hi is feasible
    result = _grid_search(hi, outer_center, outer_angle, time_budget=10.0)
    if result is not None:
        best_placements = result
        best_side = hi
    else:
        # If hi is not feasible, we need to find a feasible upper bound
        test_sides = [5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
        for ts in test_sides:
            if time.time() - start_time > time_budget - 20:
                break
            result = _grid_search(ts, outer_center, outer_angle, time_budget=10.0)
            if result is not None:
                best_placements = result
                best_side = ts
                hi = ts
                break
    
    if best_placements is None:
        # Fallback: place in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Bisection
    for _ in range(15):
        if time.time() - start_time > time_budget - 10:
            break
        mid = (lo + hi) / 2.0
        result = _grid_search(mid, outer_center, outer_angle, time_budget=8.0)
        if result is not None:
            hi = mid
            best_side = mid
            best_placements = result
        else:
            lo = mid
    
    return best_placements, (0.0, 0.0), best_side, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Bisection search on outer side length
    # Lower bound: must fit 12 unit hexagons
    # Area of outer hex = (3*sqrt(3)/2) * S^2
    # Area of 12 unit hexagons = 12 * (3*sqrt(3)/2) * 1^2 = 6*sqrt(3)
    # So S^2 >= 6, S >= sqrt(6) ≈ 2.449
    # But packing efficiency is less than 100%, so S is larger.
    lo = 2.5
    hi = 15.0
    
    # Try different outer angles to see if rotation helps
    best_result = None
    best_side = float('inf')
    
    for outer_angle in [0.0, 30.0, 60.0]:
        if time.time() > deadline - 20.0:
            break
        
        inner, center, side, angle = _bisection_search(lo, hi, (0.0, 0.0), outer_angle, 
                                                       time_budget=deadline - time.time() - 5.0)
        if side < best_side:
            best_side = side
            best_result = (inner, center, side, angle)
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: place in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END

