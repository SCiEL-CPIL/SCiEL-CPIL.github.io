"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N=11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Each inner hexagon has side length 1, can translate and rotate.
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (x, y) of outer hexagon center
      side: side length of outer hexagon
      angle: rotation angle of outer hexagon (degrees)
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with side length and rotation."""
        # Regular hexagon vertices at angles 0, 60, 120, ... from center
        # The "flat" orientation: vertices at angles 0, 60, 120, 180, 240, 300
        a = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = a + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)
    
    def polygon_contains_point(hex_verts, px, py):
        """Check if point (px,py) is inside polygon defined by hex_verts (6 vertices)."""
        # Use cross product method
        n = len(hex_verts)
        sign = 0
        for i in range(n):
            x1, y1 = hex_verts[i]
            x2, y2 = hex_verts[(i + 1) % n]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross > 1e-10:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-10:
                if sign > 0:
                    return False
                sign = -1
        return True
    
    def polygons_disjoint(hex1_verts, hex2_verts, tol=1e-7):
        """
        Check if two convex hexagons are disjoint (no interior overlap).
        Two convex polygons are disjoint if one can be separated from the other by a line.
        Use separating axis theorem (SAT).
        """
        axes = []
        for verts in [hex1_verts, hex2_verts]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                # Normal to edge
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(hex1_verts, axis)
            proj2 = np.dot(hex2_verts, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If gap exists (with tolerance for touching)
            if max1 < min2 - tol or max2 < min1 - tol:
                return True  # Separated
        
        return False  # Overlapping
    
    def check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-7):
        """
        Check:
        1. Inner hexagons are mutually disjoint.
        2. All inner hexagons are inside the outer hexagon.
        """
        # Compute outer hexagon vertices
        outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        
        # Compute inner hexagon vertices
        inner_verts_list = []
        for i in range(N):
            x, y, ang = inner_positions[i]
            verts = hex_vertices(x, y, 1.0, ang)
            inner_verts_list.append(verts)
        
        # Check mutual disjointness
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint(inner_verts_list[i], inner_verts_list[j], tol):
                    return False
        
        # Check all inner hexagons inside outer
        for i in range(N):
            for v in inner_verts_list[i]:
                if not polygon_contains_point(outer_verts, v[0], v[1]):
                    return False
        
        return True
    
    def compute_min_outer_side(inner_positions):
        """
        Given inner positions, find minimum outer hexagon side length that contains all.
        We search over outer center and rotation.
        Use optimization to minimize side length subject to containment.
        """
        # Initial guess: center at centroid, side based on bounding box
        cx = np.mean([p[0] for p in inner_positions])
        cy = np.mean([p[1] for p in inner_positions])
        
        # Estimate side from max distance
        max_dist = 0
        for p in inner_positions:
            d = math.hypot(p[0] - cx, p[1] - cy)
            if d > max_dist:
                max_dist = d
        side_est = max_dist + 2.0
        
        best_side = side_est + 10.0
        best_params = (cx, cy, 0.0, best_side)
        
        # Try several rotations and optimize center and side
        for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
            def objective(params):
                ocx, ocy, side = params
                outer_verts = hex_vertices(ocx, ocy, side, outer_angle_deg)
                # Check containment
                max_violation = 0.0
                for i in range(N):
                    x, y, ang = inner_positions[i]
                    verts = hex_vertices(x, y, 1.0, ang)
                    for v in verts:
                        if not polygon_contains_point(outer_verts, v[0], v[1]):
                            # Estimate distance outside
                            # Find closest point on outer hexagon
                            min_dist = float('inf')
                            for ei in range(6):
                                p1 = outer_verts[ei]
                                p2 = outer_verts[(ei + 1) % 6]
                                # Project point onto segment
                                dx = p2[0] - p1[0]
                                dy = p2[1] - p1[1]
                                len2 = dx * dx + dy * dy
                                if len2 > 1e-12:
                                    t = max(0, min(1, ((v[0] - p1[0]) * dx + (v[1] - p1[1]) * dy) / len2))
                                    px = p1[0] + t * dx
                                    py = p1[1] + t * dy
                                    d = math.hypot(v[0] - px, v[1] - py)
                                    if d < min_dist:
                                        min_dist = d
                            # Check if point is outside
                            # Actually we need to determine if it's outside
                            # Use the cross product test
                            outside = False
                            n_pts = len(outer_verts)
                            sign = 0
                            for ei in range(n_pts):
                                x1, y1 = outer_verts[ei]
                                x2, y2 = outer_verts[(ei + 1) % n_pts]
                                cross = (x2 - x1) * (v[1] - y1) - (y2 - y1) * (v[0] - x1)
                                if cross > 1e-10:
                                    if sign < 0:
                                        outside = True
                                        break
                                    sign = 1
                                elif cross < -1e-10:
                                    if sign > 0:
                                        outside = True
                                        break
                                    sign = -1
                            if outside:
                                if min_dist > max_violation:
                                    max_violation = min_dist
                return side + 100.0 * max_violation ** 2
            
            # Bounded optimization
            bounds = [(cx - side_est, cx + side_est),
                      (cy - side_est, cy + side_est),
                      (0.1, side_est + 5)]
            
            try:
                result = minimize(objective, [cx, cy, side_est], method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 100, 'ftol': 1e-10})
                if result.success:
                    ocx, ocy, side = result.x
                    if side < best_side:
                        # Verify validity
                        if check_validity(inner_positions, (ocx, ocy), side, outer_angle_deg):
                            best_side = side
                            best_params = (ocx, ocy, outer_angle_deg, side)
            except:
                pass
            
            if time.time() > deadline - 20:
                break
        
        return best_params, best_side
    
    # Initial placement: hexagonal lattice arrangement
    def initial_hex_layout():
        """Create a hexagonal cluster of N unit hexagons."""
        positions = []
        # Central hexagon
        positions.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons around center
        # For flat-top hexagons, neighbors are at distance 2 in 6 directions
        for i in range(6):
            angle = i * 60
            x = 2.0 * math.cos(math.radians(angle))
            y = 2.0 * math.sin(math.radians(angle))
            positions.append([x, y, 0.0])
        # Ring 2: need 4 more to reach 11
        # Place at some positions in ring 2
        # Ring 2 positions at distance 4 or 2*sqrt(3)
        # For hex lattice, ring 2 has 12 positions
        # Let's pick 4 that are well-spread
        for i in [0, 2, 4, 6]:
            angle = i * 30
            x = 4.0 * math.cos(math.radians(angle))
            y = 4.0 * math.sin(math.radians(angle))
            positions.append([x, y, 0.0])
        
        return positions
    
    def initial_square_layout():
        """Alternative: roughly square cluster."""
        positions = []
        # 3x3 grid minus corners, plus center adjustments
        coords = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),  (0, 0),  (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ]
        for i, (ix, iy) in enumerate(coords[:N]):
            x = ix * 2.1
            y = iy * 1.85
            positions.append([x, y, 0.0])
        return positions
    
    # Try multiple initial layouts and optimize
    best_result = None
    best_side = float('inf')
    
    layouts = [initial_hex_layout(), initial_square_layout()]
    
    # Also try random perturbations
    np.random.seed(42)
    for _ in range(3):
        base = initial_hex_layout()
        perturbed = []
        for p in base:
            x = p[0] + np.random.uniform(-0.5, 0.5)
            y = p[1] + np.random.uniform(-0.5, 0.5)
            ang = p[2] + np.random.uniform(-30, 30)
            perturbed.append([x, y, ang])
        layouts.append(perturbed)
    
    for layout in layouts:
        if time.time() > deadline - 10:
            break
        
        # Optimize inner positions
        def inner_objective(params):
            positions = []
            for i in range(N):
                x = params[3*i]
                y = params[3*i+1]
                ang = params[3*i+2]
                positions.append([x, y, ang])
            
            # Check pairwise overlaps
            penalty = 0.0
            inner_verts_list = []
            for i in range(N):
                verts = hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
                inner_verts_list.append(verts)
            
            for i in range(N):
                for j in range(i+1, N):
                    # SAT check
                    axes = []
                    for verts in [inner_verts_list[i], inner_verts_list[j]]:
                        for k in range(6):
                            p1 = verts[k]
                            p2 = verts[(k+1) % 6]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm > 1e-12:
                                normal = normal / norm
                                axes.append(normal)
                    for axis in axes:
                        proj1 = np.dot(inner_verts_list[i], axis)
                        proj2 = np.dot(inner_verts_list[j], axis)
                        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                        if overlap > 0:
                            penalty += overlap ** 2
                    break  # Only check first pair for speed in inner opt? No, need all.
                    # Actually this is too slow. Let's use a simpler approximation.
            
            # Simplified: use minimum distance penalty
            for i in range(N):
                for j in range(i+1, N):
                    # Approximate: distance between centers minus 2 (min separation)
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = math.hypot(dx, dy)
                    # Minimum center-to-center distance for non-overlapping unit hexagons is 2
                    if dist < 2.0:
                        penalty += (2.0 - dist) ** 2
                        # Add rotation-dependent correction
                        # Actually the minimum distance depends on relative angle
                        # For same angle, min distance is 2
                        # For 30 deg rotation, it might be different
                        # Use a conservative bound
                        # Let's add extra penalty based on angle difference
                        ang_diff = abs(positions[i][2] - positions[j][2]) % 60
                        ang_diff = min(ang_diff, 60 - ang_diff)
                        # At 0 deg, min dist = 2; at 30 deg, min dist can be smaller
                        # Use 2*cos(ang_diff) as a rough estimate? Not quite.
                        # Keep simple: penalty for dist < 2
            return penalty
        
        # Optimize inner positions to minimize overlaps
        x0 = []
        for p in layout:
            x0.extend(p)
        
        try:
            result = minimize(inner_objective, x0, method='L-BFGS-B', 
                              bounds=[(v - 5, v + 5) for v in x0],
                              options={'maxiter': 200, 'ftol': 1e-12})
            if result.success:
                opt_positions = []
                for i in range(N):
                    opt_positions.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
            else:
                opt_positions = layout
        except:
            opt_positions = layout
        
        # Now optimize outer hexagon
        params, side = compute_min_outer_side(opt_positions)
        
        if side < best_side:
            # Verify
            if check_validity(opt_positions, (params[0], params[1]), params[3], params[2]):
                best_side = params[3]
                best_result = (opt_positions, (params[0], params[1]), params[3], params[2])
            else:
                # Try with slightly larger side
                for delta in [0.01, 0.05, 0.1, 0.5, 1.0]:
                    test_side = params[3] + delta
                    if check_validity(opt_positions, (params[0], params[1]), test_side, params[2]):
                        best_side = test_side
                        best_result = (opt_positions, (params[0], params[1]), test_side, params[2])
                        break
    
    if best_result is None:
        # Fallback: use initial layout with large enough outer
        positions = initial_hex_layout()
        cx, cy = 0.0, 0.0
        # Estimate side
        max_d = max(math.hypot(p[0] - cx, p[1] - cy) for p in positions)
        side = max_d + 2.0
        # Make valid
        while not check_validity(positions, (cx, cy), side, 0.0):
            side += 0.1
        best_result = (positions, (cx, cy), side, 0.0)
    
    return best_result
# EVOLVE-BLOCK-END
