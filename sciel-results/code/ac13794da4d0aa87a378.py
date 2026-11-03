"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 115.0  # Safety margin before 120s limit

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_polygon(point, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagons_disjoint(v1, v2, eps=1e-9):
    """Check if two convex polygons are disjoint (no overlap).
    Returns True if they are disjoint, False if they overlap."""
    # For convex polygons, check if any edge of one intersects any edge of the other
    # Or if one vertex of one is inside the other
    # Simpler: check if any vertex of v1 is inside v2 or vice versa
    for i in range(6):
        if point_in_polygon(v1[i], v2):
            return False
    for i in range(6):
        if point_in_polygon(v2[i], v1):
            return False
    # Also check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def hexagons_inside_outer(inner_verts, outer_verts):
    """Check if all vertices of inner hexagon are inside or on the boundary of outer hexagon."""
    for v in inner_verts:
        if not point_in_polygon(v, outer_verts):
            return False
    return True

def evaluate_packing(inner_positions, outer_center, outer_side, outer_angle):
    """Evaluate a packing configuration.
    inner_positions: list of (x, y, angle_deg) for N hexagons
    Returns: outer_side if valid, None if invalid"""
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    inner_verts_list = []
    for pos in inner_positions:
        x, y, ang = pos
        verts = hexagon_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
    
    # Check all inner hexagons are inside outer
    for verts in inner_verts_list:
        if not hexagons_inside_outer(verts, outer_verts):
            return None
    
    # Check all pairs of inner hexagons are disjoint
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts_list[i], inner_verts_list[j]):
                return None
    
    return outer_side

def construct():
    start_time = time.time()
    
    # Try several strategies with time budget
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Symmetric arrangement - hexagonal lattice pattern
    # Place 12 hexagons in a compact hexagonal cluster
    # A hexagonal number pattern: center + 6 around + 5 more
    # For 12 hexagons, one good pattern is a "flower" with some extensions
    
    # Let's try a known good configuration:
    # Hexagonal lattice with spacing 2 (since unit hexagons need distance 2 between centers for non-overlap in same orientation)
    # Actually for unit hexagons, minimum center distance depends on relative orientation
    
    # Start with a grid-like arrangement
    # For 12 hexagons, try 4x3 grid with hexagonal spacing
    # Hexagonal lattice vectors: (2, 0) and (1, sqrt(3))
    
    sqrt3 = np.sqrt(3)
    
    # Initial guess: hexagonal packing
    positions = []
    # 4 columns, 3 rows in hexagonal lattice
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * sqrt3
            positions.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Estimate outer side length needed
    # Find the bounding hexagon
    def compute_min_enclosing_hexagon(positions):
        """Find minimum side length regular hexagon that encloses all unit hexagons at given positions."""
        # Try different orientations
        best_side = float('inf')
        for outer_angle in range(0, 60, 15):
            # For a given outer hexagon orientation, find minimum side length
            # such that all inner hexagons fit inside
            
            # Binary search on side length
            lo = 0.0
            hi = 20.0
            
            for _ in range(30):
                mid = (lo + hi) / 2
                outer_verts = hexagon_vertices(0, 0, mid, outer_angle)
                
                valid = True
                for pos in positions:
                    x, y, ang = pos
                    verts = hexagon_vertices(x, y, 1.0, ang)
                    if not hexagons_inside_outer(verts, outer_verts):
                        valid = False
                        break
                
                if valid:
                    hi = mid
                else:
                    lo = mid
            
            if hi < best_side:
                best_side = hi
        
        return best_side
    
    # Check if initial arrangement is valid (disjoint inner hexagons)
    inner_verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in positions]
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts_list[i], inner_verts_list[j]):
                valid = False
                break
        if not valid:
            break
    
    if valid:
        side = compute_min_enclosing_hexagon(positions)
        if side < best_score:
            best_score = side
            best_result = (positions, (0.0, 0.0), side, 0.0)
    
    # Strategy 2: Use differential evolution to optimize
    if time.time() - start_time < DEADLINE:
        def objective(params):
            """Objective: minimize outer side length.
            params: 3N + 3 = 39 parameters
            First 3N: x, y, angle for each inner hexagon
            Last 3: outer center x, y, outer angle
            Outer side is determined by constraint satisfaction."""
            elapsed = time.time() - start_time
            if elapsed > DEADLINE:
                return 100.0  # Penalty
            
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            
            positions = []
            for i in range(N):
                x = inner_params[i*3]
                y = inner_params[i*3+1]
                ang = inner_params[i*3+2]
                positions.append([x, y, ang])
            
            outer_cx = outer_params[0]
            outer_cy = outer_params[1]
            outer_angle = outer_params[2]
            
            # Translate so outer center is at origin for symmetry
            # Actually, let's just evaluate directly
            
            # Estimate required outer side
            # Use a soft constraint approach
            outer_verts = hexagon_vertices(outer_cx, outer_cy, 15.0, outer_angle)  # Large hex for checking
            
            # Check if all inner hexagons fit in a hexagon of side S
            # We'll compute the minimum S needed
            
            # For each inner hexagon, compute the minimum outer hexagon side that contains it
            # given the outer hexagon center and angle
            
            min_side = 0.0
            for pos in positions:
                x, y, ang = pos
                verts = hexagon_vertices(x, y, 1.0, ang)
                
                # For each vertex, compute the minimum hexagon side that contains it
                # given center (outer_cx, outer_cy) and angle outer_angle
                for vx, vy in verts:
                    # Transform to hexagon frame
                    dx = vx - outer_cx
                    dy = vy - outer_cy
                    # Rotate by -outer_angle
                    cos_a = np.cos(-np.deg2rad(outer_angle))
                    sin_a = np.sin(-np.deg2rad(outer_angle))
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    
                    # In the hexagon frame, the hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2, etc.
                    # Actually, a regular hexagon with side s centered at origin with one vertex at angle 0:
                    # The distance from center to each edge is s * sqrt(3) / 2
                    # The distance from center to each vertex is s
                    
                    # For a point (rx, ry), the minimum s such that it's inside the hexagon:
                    # The hexagon has 6 edges. For each edge, the constraint is:
                    # n_i · p <= s * sqrt(3) / 2 where n_i is the outward normal of edge i
                    
                    # Edge normals for hexagon with vertex at angle 0:
                    # The edges are perpendicular to directions at angles 30, 90, 150, 210, 270, 330 degrees
                    # Wait, let me think again.
                    
                    # A regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees
                    # has edges with outward normals at angles 30, 90, 150, 210, 270, 330 degrees
                    
                    max_constraint = 0.0
                    for k in range(6):
                        normal_angle = np.deg2rad(30 + k * 60)
                        nx = np.cos(normal_angle)
                        ny = np.sin(normal_angle)
                        dist = abs(rx * nx + ry * ny)
                        # The distance from center to the edge along the normal is s * sqrt(3) / 2
                        # So we need s * sqrt(3) / 2 >= dist
                        # s >= dist * 2 / sqrt(3)
                        required_s = dist * 2.0 / sqrt3
                        if required_s > max_constraint:
                            max_constraint = required_s
                    
                    if max_constraint > min_side:
                        min_side = max_constraint
            
            # Add penalty for overlapping inner hexagons
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    vi = hexagon_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
                    vj = hexagon_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                    if not hexagons_disjoint(vi, vj):
                        # Compute overlap penalty based on center distance
                        di = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
                        overlap_penalty += max(0, 2.0 - di) ** 2
            
            # Also penalize if inner hexagons are too far from center
            spread_penalty = 0.0
            for pos in positions:
                d = np.sqrt((pos[0] - outer_cx)**2 + (pos[1] - outer_cy)**2)
                spread_penalty += d ** 2 * 0.01
            
            return min_side + overlap_penalty * 10.0 + spread_penalty
        
        # Set up bounds
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((0, 60))  # angle
        bounds.append((-1, 1))  # outer cx
        bounds.append((-1, 1))  # outer cy
        bounds.append((0, 60))  # outer angle
        
        # Run differential evolution with limited iterations
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            
            if result.success or result.fun < best_score:
                params = result.x
                inner_params = params[:3*N]
                outer_params = params[3*N:]
                
                positions_de = []
                for i in range(N):
                    x = inner_params[i*3]
                    y = inner_params[i*3+1]
                    ang = inner_params[i*3+2]
                    positions_de.append([x, y, ang])
                
                outer_cx = outer_params[0]
                outer_cy = outer_params[1]
                outer_angle = outer_params[2]
                
                # Compute actual minimum side length
                side = compute_min_enclosing_hexagon(positions_de)
                
                # Verify validity
                inner_verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in positions_de]
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if not hexagons_disjoint(inner_verts_list[i], inner_verts_list[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and side < best_score:
                    best_score = side
                    best_result = (positions_de, (outer_cx, outer_cy), side, outer_angle)
                    
        except Exception as e:
            pass
    
    # Strategy 3: Try known good configurations
    if time.time() - start_time < DEADLINE:
        # Configuration: 12 hexagons in a compact arrangement
        # Try a "hexagonal flower" with 7 in center and 5 around
        configs = []
        
        # Config 1: 4x3 hexagonal grid (already tried)
        # Config 2: Compact cluster
        # Center at (0,0), 6 around at distance 2, then 5 more
        
        # Let's try a specific known good pattern
        # For 12 unit hexagons, one good arrangement is:
        # Row 0: 4 hexagons
        # Row 1: 4 hexagons (offset)
        # Row 2: 4 hexagons
        
        # Actually, let's try different offsets
        for offset_factor in [1.0, 1.1, 0.9]:
            positions_test = []
            for row in range(3):
                for col in range(4):
                    x = col * 2.0 * offset_factor + (row % 2) * 1.0 * offset_factor
                    y = row * sqrt3 * offset_factor
                    positions_test.append([x, y, 0.0])
            
            # Center
            xs = [p[0] for p in positions_test]
            ys = [p[1] for p in positions_test]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            for p in positions_test:
                p[0] -= cx
                p[1] -= cy
            
            inner_verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in positions_test]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if not hexagons_disjoint(inner_verts_list[i], inner_verts_list[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side = compute_min_enclosing_hexagon(positions_test)
                if side < best_score:
                    best_score = side
                    best_result = (positions_test, (0.0, 0.0), side, 0.0)
        
        # Config 3: Rotated arrangements
        for base_angle in [0, 15, 30, 45]:
            positions_test = []
            for row in range(3):
                for col in range(4):
                    x = col * 2.0 + (row % 2) * 1.0
                    y = row * sqrt3
                    # Rotate
                    rad = np.deg2rad(base_angle)
                    xr = x * np.cos(rad) - y * np.sin(rad)
                    yr = x * np.sin(rad) + y * np.cos(rad)
                    positions_test.append([xr, yr, base_angle])
            
            xs = [p[0] for p in positions_test]
            ys = [p[1] for p in positions_test]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            for p in positions_test:
                p[0] -= cx
                p[1] -= cy
            
            inner_verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in positions_test]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if not hexagon_disjoint(inner_verts_list[i], inner_verts_list[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side = compute_min_enclosing_hexagon(positions_test)
                if side < best_score:
                    best_score = side
                    best_result = (positions_test, (0.0, 0.0), side, 0.0)
    
    # Strategy 4: Local optimization from best found
    if time.time() - start_time < DEADLINE and best_result is not None:
        positions_opt = [p[:] for p in best_result[0]]
        outer_cx_opt = best_result[1][0]
        outer_cy_opt = best_result[1][1]
        outer_angle_opt = best_result[1][2]
        
        def local_objective(params):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE:
                return 100.0
            
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            
            positions_local = []
            for i in range(N):
                x = inner_params[i*3]
                y = inner_params[i*3+1]
                ang = inner_params[i*3+2]
                positions_local.append([x, y, ang])
            
            ocx = outer_params[0]
            ocy = outer_params[1]
            oang = outer_params[2]
            
            # Compute minimum side
            min_side = 0.0
            for pos in positions_local:
                x, y, ang = pos
                verts = hexagon_vertices(x, y, 1.0, ang)
                for vx, vy in verts:
                    dx = vx - ocx
                    dy = vy - ocy
                    cos_a = np.cos(-np.deg2rad(oang))
                    sin_a = np.sin(-np.deg2rad(oang))
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    
                    max_constraint = 0.0
                    for k in range(6):
                        normal_angle = np.deg2rad(30 + k * 60)
                        nx = np.cos(normal_angle)
                        ny = np.sin(normal_angle)
                        dist = abs(rx * nx + ry * ny)
                        required_s = dist * 2.0 / sqrt3
                        if required_s > max_constraint:
                            max_constraint = required_s
                    
                    if max_constraint > min_side:
                        min_side = max_constraint
            
            # Overlap penalty
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    vi = hexagon_vertices(positions_local[i][0], positions_local[i][1], 1.0, positions_local[i][2])
                    vj = hexagon_vertices(positions_local[j][0], positions_local[j][1], 1.0, positions_local[j][2])
                    if not hexagons_disjoint(vi, vj):
                        di = np.sqrt((positions_local[i][0] - positions_local[j][0])**2 + (positions_local[i][1] - positions_local[j][1])**2)
                        overlap_penalty += max(0, 2.0 - di) ** 2
            
            return min_side + overlap_penalty * 10.0
        
        # Initial params
        init_params = []
        for p in positions_opt:
            init_params.extend([p[0], p[1], p[2]])
        init_params.extend([outer_cx_opt, outer_cy_opt, outer_angle_opt])
        
        try:
            result = minimize(
                local_objective, init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if result.fun < best_score:
                params = result.x
                positions_nm = []
                for i in range(N):
                    x = params[i*3]
                    y = params[i*3+1]
                    ang = params[i*3+2]
                    positions_nm.append([x, y, ang])
                
                ocx = params[3*N]
                ocy = params[3*N+1]
                oang = params[3*N+2]
                
                side = compute_min_enclosing_hexagon(positions_nm)
                
                inner_verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in positions_nm]
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if not hexagons_disjoint(inner_verts_list[i], inner_verts_list[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and side < best_score:
                    best_score = side
                    best_result = (positions_nm, (ocx, ocy), side, oang)
                    
        except Exception as e:
            pass
    
    # Return best result found
    if best_result is not None:
        return best_result
    else:
        # Fallback: return a valid but not optimal configuration
        positions_fb = []
        for row in range(3):
            for col in range(4):
                x = col * 2.0 + (row % 2) * 1.0
                y = row * sqrt3
                positions_fb.append([x, y, 0.0])
        
        xs = [p[0] for p in positions_fb]
        ys = [p[1] for p in positions_fb]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in positions_fb:
            p[0] -= cx
            p[1] -= cy
        
        side_fb = compute_min_enclosing_hexagon(positions_fb)
        return positions_fb, (0.0, 0.0), side_fb, 0.0
# EVOLVE-BLOCK-END
