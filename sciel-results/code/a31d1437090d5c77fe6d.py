"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hex(px, py, poly, eps=1e-7):
        """Check if point (px, py) is inside convex hexagon poly."""
        # poly is list of (x, y) in CCW order
        inside = True
        for k in range(6):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % 6]
            # Vector edge
            ex, ey = x2 - x1, y2 - y1
            # Vector to point
            px1, py1 = px - x1, py - y1
            cross = ex * py1 - ey * px1
            if cross < -eps:  # If strictly to the right (outside)
                inside = False
                break
        return inside

    def check_overlap(centers, angles, eps=1e-6):
        """Check if any two inner hexagons overlap."""
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                # Max distance between any two points in two unit hexagons is 4 (vertex to vertex if aligned)
                # Min distance to avoid overlap is 2.0 (side to side if aligned)
                # If dist > 4.0, definitely no overlap.
                if dist_sq > 16.0:
                    continue

                vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])

                # If dist < 1.9, likely overlap (conservative)
                if dist_sq < (1.9) ** 2:
                    return False

                # Detailed check for close distances
                # Check if any vertex of i is inside polygon j
                for vx, vy in vi:
                    if point_in_hex(vx, vy, vj):
                        return False
                # Check if any vertex of j is inside polygon i
                for vx, vy in vj:
                    if point_in_hex(vx, vy, vi):
                        return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """
        Compute the minimal side length of a regular hexagon (centered at origin, 
        with given rotation) that contains all inner unit hexagons.
        """
        # 6 outward normal directions for outer hexagon with rotation outer_angle
        # The sides of the outer hexagon are perpendicular to these normals.
        # The distance from center to side is the apothem.
        # Side length L = 2 * apothem / sqrt(3)
        
        normals = []
        for k in range(6):
            # Normals point to the midpoints of the sides.
            # If outer hexagon is rotated by outer_angle, its sides are at angles outer_angle + k*60 + 30?
            # Let's define the normals as pointing to the vertices of the dual hexagon.
            # A regular hexagon with side L has apothem a = L*sqrt(3)/2.
            # The condition is that for all vertices v of inner hexagons,
            # max_k (v . n_k) <= a
            # where n_k are unit normals to the sides.
            # The normals are at angles: outer_angle + 30 + k*60 degrees.
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])
            
            for nk in range(6):
                nx, ny = normals[nk]
                n_angle = math.atan2(ny, nx)
                
                # Max dot product of any vertex of inner hex i with normal n_k
                # Vertex j of inner hex i is at angle: inner_angle_rad + j*pi/3
                # Dot product is cos(vertex_angle - n_angle)
                # We need max over j of cos(inner_angle_rad + j*pi/3 - n_angle)
                
                # Since cos is periodic and symmetric, we can just check the 6 vertices.
                max_dot = -2.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val
                
                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent
        
        # max_extent is the apothem of the outer hexagon
        return 2.0 * max_extent / sqrt3

    # Initialize with a valid hexagonal lattice configuration
    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3 / 2 * r
        y = 1.5 * r
        return (x, y)

    init_grid_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]

    centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
    angles = [0.0] * N

    # Verify this initial config is valid
    if not check_overlap(centers, angles):
        # Fallback to a very sparse config
        centers = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0

    # Define the objective function for scipy
    def objective(params):
        # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_angle]
        # Total 12*3 + 1 = 37 parameters
        centers_opt = []
        angles_opt = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            centers_opt.append((x, y))
            angles_opt.append(a)
        outer_angle = params[36]
        
        # Check overlap with a penalty
        # If overlap, return a large value
        if not check_overlap(centers_opt, angles_opt):
            return 1e6 + enclosing_side_length(centers_opt, angles_opt, outer_angle)
        
        return enclosing_side_length(centers_opt, angles_opt, outer_angle)

    # Initial parameters
    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(0.0)
    
    # Bounds
    bounds = [(-10, 10), (-10, 10), (-180, 180)] * N + [(-30, 30)]

    # Use SLSQP or Nelder-Mead
    # SLSQP might be slow or get stuck. Nelder-Mead is derivative-free and robust.
    # Let's try Nelder-Mead with a reasonable maxiter.
    
    # Since the problem is non-convex and has discontinuities (overlap penalty),
    # a global-ish approach or multiple restarts might be better.
    # But we have a good local start.
    
    # Let's do a few iterations of local search with Nelder-Mead, restarting if stuck or improved.
    
    current_params = init_params[:]
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Run minimization
        try:
            res = minimize(
                objective,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
        except Exception:
            break
            
        if res.success or res.fun < objective(current_params):
            current_params = res.x
            # Update best
            centers_opt = []
            angles_opt = []
            for i in range(N):
                centers_opt.append((current_params[i * 3], current_params[i * 3 + 1]))
                angles_opt.append(current_params[i * 3 + 2])
            outer_angle = current_params[36]
            
            if check_overlap(centers_opt, angles_opt):
                L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
                if L < best_L:
                    best_L = L
                    best_centers = centers_opt
                    best_angles = angles_opt
                    best_outer_angle = outer_angle
        else:
            # If no improvement, maybe perturb and try again?
            # Or break.
            pass
            
    # Final check and return
    if not check_overlap(best_centers, best_angles):
        # If best is invalid, use the initial valid one
        best_centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
        best_angles = [0.0] * N
        best_L = enclosing_side_length(best_centers, best_angles, 0.0)
        best_outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])

    center = (0.0, 0.0)
    side = best_L
    angle = best_outer_angle

    return inner, center, side, angle
# EVOLVE-BLOCK-END
