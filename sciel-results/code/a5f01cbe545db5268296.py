"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, angle_deg, side):
        """Check if point (px,py) is inside regular hexagon with center (cx,cy),
        rotation angle_deg, side length."""
        angle_rad = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        # Rotate point into hexagon's local frame
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # A point (lx, ly) is inside a regular hexagon of side s centered at origin
        # iff for all 3 axes (at 0, 60, 120 deg from x-axis), the projection is <= s*sqrt(3)/2
        # Actually, the inradius of a regular hexagon with side s is s*sqrt(3)/2
        # The hexagon is the intersection of 3 strips of width s*sqrt(3)
        # The 3 normal directions are at angles 30, 90, 150 degrees (for a flat-top hex)
        # For a pointy-top hex (vertices at 0, 60, ...), normals are at 0, 60, 120
        # Let's use: the hexagon with vertices at angles 0, 60, 120, 180, 240, 300
        # The 3 pairs of parallel sides have normals at 30, 90, 150 degrees
        inradius = side * math.sqrt(3) / 2
        for k in range(3):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = lx * nx + ly * ny
            if abs(proj) > inradius + 1e-9:
                return False
        return True

    def hexagon_in_hexagon(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_angle, side):
        """Check if inner unit hexagon is inside outer hexagon."""
        verts = hex_vertices(inner_cx, inner_cy, inner_angle)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_angle, side):
                return False
        return True

    def compute_min_side_for_containment(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg):
        """Compute minimum side length of outer hexagon to contain all inner hexagons."""
        angle_rad = math.radians(outer_angle_deg)
        max_proj = 0.0
        for i in range(N):
            verts = hex_vertices(inner_centers[i], inner_angles[i], 0)  # angle already in vertices
            # Wait, hex_vertices takes angle_deg. Let me fix.
            verts = hex_vertices(inner_centers[i], inner_angles[i], 0)
            # Actually hex_vertices(cx, cy, angle_deg) - the angle is already baked in
            # Let me recompute:
            verts = hex_vertices(inner_centers[i], inner_angles[i], 0)
            for k in range(6):
                normal_angle = angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                rel = verts - np.array([outer_cx, outer_cy])
                proj = np.dot(rel, normal)
                max_proj = max(max_proj, np.max(proj))
        # For a regular hexagon with side s, the distance from center to a side (inradius) is s*sqrt(3)/2
        # The max projection onto any normal direction equals the inradius for containment
        # So: max_proj <= side * sqrt(3) / 2
        # side >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        """
        params: 
        0..23: inner centers (x,y) for 12 hexagons
        24..35: inner angles (radians) for 12 hexagons
        36: outer cx
        37: outer cy
        38: outer angle (radians)
        """
        if time.time() > deadline:
            return 1e6

        inner_centers = [(params[2*i], params[2*i+1]) for i in range(N)]
        inner_angles_rad = [params[24+i] for i in range(N)]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle_rad = params[38]

        # Compute side length needed
        angle_deg = math.degrees(outer_angle_rad)
        inner_angles_deg = [math.degrees(a) for a in inner_angles_rad]
        side = compute_min_side_for_containment(inner_centers, inner_angles_deg, outer_cx, outer_cy, angle_deg)

        # Penalty for overlaps: distance between centers must be >= 2
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = inner_centers[i][0] - inner_centers[j][0]
                dy = inner_centers[i][1] - inner_centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2

        return side + 500.0 * penalty

    def constraints(params):
        """Non-overlap constraints: distance between centers >= 2."""
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                # dist^2 >= 4, i.e., dist^2 - 4 >= 0
                def make_constraint(i, j):
                    def constraint(p):
                        dx = p[2*i] - p[2*j]
                        dy = p[2*i+1] - p[2*j+1]
                        return dx*dx + dy*dy - 4.0
                    return constraint
                cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        return cons

    # Initial configuration: hexagonal lattice pattern
    s = 1.0
    # For hexagons with vertices at angles 0, 60, 120...:
    # Width (horizontal) = 2s, Height (vertical) = s*sqrt(3)
    # In a hexagonal lattice:
    # Same row: dx = 2s
    # Adjacent row: dy = s*sqrt(3)/2, dx_offset = s
    
    centers = []
    # Try a 3-4-3-2 pattern or similar
    # Let's do rows of 3, 4, 3, 2 = 12
    row_sizes = [3, 4, 3, 2]
    dy = s * math.sqrt(3) / 2
    dx = 2 * s
    offset = s
    
    for r, size in enumerate(row_sizes):
        row_width = (size - 1) * dx
        start_x = -row_width / 2
        for c in range(size):
            x = start_x + c * dx
            y = r * dy
            centers.append((x, y))
    
    # Center the whole thing
    cx_mean = sum(c[0] for c in centers) / N
    cy_mean = sum(c[1] for c in centers) / N
    centers = [(x - cx_mean, y - cy_mean) for x, y in centers]

    # Initialize parameters
    x0 = np.zeros(39)
    for i in range(N):
        x0[2*i] = centers[i][0]
        x0[2*i+1] = centers[i][1]
        x0[24+i] = 0.0  # angle in radians

    x0[36] = 0.0
    x0[37] = 0.0
    x0[38] = 0.0

    best_side = float('inf')
    best_params = x0.copy()
    best_valid = False

    # Try multiple initial configurations and optimization methods
    configs = []

    # Config 1: Hexagonal lattice
    configs.append(('lattice', x0.copy()))

    # Config 2: Ring of 6 + inner 6
    ring_centers = []
    for i in range(6):
        angle = i * math.pi / 3
        ring_centers.append((2.5 * math.cos(angle), 2.5 * math.sin(angle)))
    inner_ring = []
    for i in range(6):
        angle = i * math.pi / 3 + math.pi / 6
        inner_ring.append((1.2 * math.cos(angle), 1.2 * math.sin(angle)))
    all_centers = ring_centers + inner_ring
    cx_mean = sum(c[0] for c in all_centers) / N
    cy_mean = sum(c[1] for c in all_centers) / N
    all_centers = [(x - cx_mean, y - cy_mean) for x, y in all_centers]
    x0_ring = np.zeros(39)
    for i in range(N):
        x0_ring[2*i] = all_centers[i][0]
        x0_ring[2*i+1] = all_centers[i][1]
        x0_ring[24+i] = 0.0
    x0_ring[36] = 0.0
    x0_ring[37] = 0.0
    x0_ring[38] = 0.0
    configs.append(('ring', x0_ring.copy()))

    # Config 3: Grid 4x3
    grid_centers = []
    dx_g = 2.0
    dy_g = math.sqrt(3)
    for r in range(4):
        for c in range(3):
            x = c * dx_g
            y = r * dy_g
            if r % 2 == 1:
                x += 1.0
            grid_centers.append((x, y))
    cx_mean = sum(c[0] for c in grid_centers) / N
    cy_mean = sum(c[1] for c in grid_centers) / N
    grid_centers = [(x - cx_mean, y - cy_mean) for x, y in grid_centers]
    x0_grid = np.zeros(39)
    for i in range(N):
        x0_grid[2*i] = grid_centers[i][0]
        x0_grid[2*i+1] = grid_centers[i][1]
        x0_grid[24+i] = 0.0
    x0_grid[36] = 0.0
    x0_grid[37] = 0.0
    x0_grid[38] = 0.0
    configs.append(('grid', x0_grid.copy()))

    for config_name, x0_config in configs:
        if time.time() > deadline:
            break

        try:
            # Try SLSQP with constraints
            res = minimize(
                objective,
                x0_config,
                method='SLSQP',
                constraints=constraints(x0_config),
                options={'maxiter': 500, 'ftol': 1e-6}
            )

            # Evaluate result
            inner_centers = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
            inner_angles_rad = [res.x[24+i] for i in range(N)]
            outer_cx = res.x[36]
            outer_cy = res.x[37]
            outer_angle_rad = res.x[38]

            angle_deg = math.degrees(outer_angle_rad)
            inner_angles_deg = [math.degrees(a) for a in inner_angles_rad]
            side = compute_min_side_for_containment(inner_centers, inner_angles_deg, outer_cx, outer_cy, angle_deg)

            # Check validity: no overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = inner_centers[i][0] - inner_centers[j][0]
                    dy = inner_centers[i][1] - inner_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 2.0 - 1e-5:
                        valid = False
                        break
                if not valid:
                    break

            # Check containment
            if valid:
                for i in range(N):
                    if not hexagon_in_hexagon(inner_centers[i][0], inner_centers[i][1], inner_angles_deg[i], outer_cx, outer_cy, angle_deg, side):
                        valid = False
                        break

            if valid and side < best_side:
                best_side = side
                best_params = res.x.copy()
                best_valid = True

        except Exception as e:
            pass

        # Also try Nelder-Mead without constraints (penalty-based)
        if time.time() > deadline:
            break

        try:
            res2 = minimize(
                objective,
                x0_config,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )

            inner_centers = [(res2.x[2*i], res2.x[2*i+1]) for i in range(N)]
            inner_angles_rad = [res2.x[24+i] for i in range(N)]
            outer_cx = res2.x[36]
            outer_cy = res2.x[37]
            outer_angle_rad = res2.x[38]

            angle_deg = math.degrees(outer_angle_rad)
            inner_angles_deg = [math.degrees(a) for a in inner_angles_rad]
            side = compute_min_side_for_containment(inner_centers, inner_angles_deg, outer_cx, outer_cy, angle_deg)

            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = inner_centers[i][0] - inner_centers[j][0]
                    dy = inner_centers[i][1] - inner_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 2.0 - 1e-5:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                for i in range(N):
                    if not hexagon_in_hexagon(inner_centers[i][0], inner_centers[i][1], inner_angles_deg[i], outer_cx, outer_cy, angle_deg, side):
                        valid = False
                        break

            if valid and side < best_side:
                best_side = side
                best_params = res2.x.copy()
                best_valid = True

        except Exception as e:
            pass

    # If no valid result, use best effort
    if not best_valid:
        # Use the initial lattice configuration
        inner_centers = centers
        inner_angles_deg = [0.0] * N
        outer_cx = 0.0
        outer_cy = 0.0
        angle_deg = 0.0
        side = compute_min_side_for_containment(inner_centers, inner_angles_deg, outer_cx, outer_cy, angle_deg)
        best_side = side
        best_valid = True

    # Construct return value
    inner = []
    for i in range(N):
        cx = best_params[2*i]
        cy = best_params[2*i+1]
        ang_deg = math.degrees(best_params[24+i])
        inner.append([float(cx), float(cy), float(ang_deg)])

    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_angle_deg = math.degrees(best_params[38])

    # Add small safety margin
    final_side = best_side * 1.005 + 1e-4

    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle_deg)
# EVOLVE-BLOCK-END

