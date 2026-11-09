import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg):
        """Return 6 vertices of unit hexagon centered at (cx,cy) rotated by angle_deg."""
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        # Unit hexagon vertices at angles 0,60,120,180,240,300
        verts = []
        for k in range(6):
            theta = a + math.radians(60*k)
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return verts

    def hex_dist_sq(c1, c2, r1, r2):
        """Squared center-to-center distance between two hexagons."""
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return dx*dx + dy*dy

    def polygons_disjoint_fast(v1, v2, min_dist_sq=0.49):
        """Fast check: if centers too far, definitely disjoint. Otherwise detailed check."""
        # Quick bounding box check
        min_x1 = min(p[0] for p in v1)
        max_x1 = max(p[0] for p in v1)
        min_y1 = min(p[1] for p in v1)
        max_y1 = max(p[1] for p in v1)
        min_x2 = min(p[0] for p in v2)
        max_x2 = max(p[0] for p in v2)
        min_y2 = min(p[1] for p in v2)
        max_y2 = max(p[1] for p in v2)
        
        # Bounding box separation
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return True
        
        # Check if any vertex of one is inside the other
        def point_in_hex(pt, poly):
            x, y = pt
            n = len(poly)
            inside = False
            p1x, p1y = poly[0]
            for i in range(1, n + 1):
                p2x, p2y = poly[i % n]
                if (y > p1y) != (y > p2y):
                    xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                    if x < xint:
                        inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        
        for v in v1:
            if point_in_hex(v, v2):
                return False
        for v in v2:
            if point_in_hex(v, v1):
                return False
        
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))
        
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def get_min_L(vertices_list):
        """Compute minimal regular hexagon side length that contains all vertices."""
        # Precompute cos/sin for the 6 normal directions
        cos_norms = [math.cos(math.radians(30 + 60*k)) for k in range(6)]
        sin_norms = [math.sin(math.radians(30 + 60*k)) for k in range(6)]
        
        max_proj = 0.0
        for v in vertices_list:
            x, y = v
            for k in range(6):
                proj = x*cos_norms[k] + y*sin_norms[k]
                if proj > max_proj:
                    max_proj = proj
        return max_proj / (sqrt3 / 2.0)

    def is_valid_config(positions):
        """Check if configuration is valid (no overlaps)."""
        verts_list = [get_hex_vertices(p[0], p[1], p[2]) for p in positions]
        for i in range(N):
            for j in range(i+1, N):
                if not polygons_disjoint_fast(verts_list[i], verts_list[j]):
                    return False
        return True

    # Discrete search space
    LATTICE_SPACING = 0.25
    ROTATIONS = list(range(0, 151, 15))  # 0, 15, 30, ..., 150

    # Generate lattice points within a reasonable range
    lattice_points = []
    max_coord = 3.5
    for x in np.arange(-max_coord, max_coord + LATTICE_SPACING, LATTICE_SPACING):
        for y in np.arange(-max_coord, max_coord + LATTICE_SPACING, LATTICE_SPACING):
            lattice_points.append((x, y))

    # Initial best: hexagonal grid
    grid_centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                grid_centers.append((x, y))

    dists = [(math.hypot(x, y), i) for i, (x, y) in enumerate(grid_centers)]
    dists.sort()
    init_grid_idx = [d[1] for d in dists[:12]]

    best_positions = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        best_positions.append([x, y, 0.0])

    best_L = get_min_L([v for p in best_positions for v in get_hex_vertices(p[0], p[1], p[2])])

    # Discrete hill-climbing search
    def evaluate_config(positions):
        """Return (L, is_valid) for a configuration."""
        all_vertices = []
        for p in positions:
            all_vertices.extend(get_hex_vertices(p[0], p[1], p[2]))
        L = get_min_L(all_vertices)
        valid = is_valid_config(positions)
        return L, valid

    # Local search: try moving each hexagon to nearby lattice points and different rotations
    def local_search(positions, max_iterations=3000):
        """Hill-climbing local search with early termination."""
        nonlocal best_L, best_positions
        current_positions = [p[:] for p in positions]
        current_L, current_valid = evaluate_config(current_positions)
        if not current_valid:
            return current_positions, float('inf')
        current_L = current_L if current_valid else float('inf')

        # Precompute nearby lattice offsets for efficiency
        offsets_1 = [(dx * LATTICE_SPACING, dy * LATTICE_SPACING) 
                     for dx in range(-2, 3) for dy in range(-2, 3)
                     if not (dx == 0 and dy == 0)]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 15.0:
                break

            improved = False
            # Try to improve by moving one hexagon at a time
            for i in range(N):
                if time.time() > deadline - 15.0:
                    break

                old_x, old_y, old_rot = current_positions[i]

                # Try nearby lattice points
                for ox, oy in offsets_1:
                    if time.time() > deadline - 15.0:
                        break
                    nx = old_x + ox
                    ny = old_y + oy

                    # Try a few key rotations (0, 30, 60, 90, 120, 150)
                    for rot in [0, 30, 60, 90, 120, 150]:
                        if time.time() > deadline - 15.0:
                            break

                        new_pos = [p[:] for p in current_positions]
                        new_pos[i] = [nx, ny, float(rot)]

                        new_L, new_valid = evaluate_config(new_pos)
                        if new_valid and new_L < current_L - 1e-10:
                            current_positions = new_pos
                            current_L = new_L
                            improved = True
                            if current_L < best_L:
                                best_L = current_L
                                best_positions = [p[:] for p in current_positions]
                            break  # Move to this position, break rotation loop
                    if improved:
                        break  # Break nearby points loop

            if not improved:
                break

        return current_positions, current_L

    # Phase 1: Start from multiple initial configurations
    initial_configs = []

    # 1. Grid with 0° rotation
    config1 = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        config1.append([x, y, 0.0])
    initial_configs.append(config1)

    # 2. Grid with 30° rotation
    config2 = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        config2.append([x, y, 30.0])
    initial_configs.append(config2)

    # 3. Alternating 0°/30°
    config3 = []
    for idx, i in enumerate(init_grid_idx):
        x, y = grid_centers[i]
        rot = 0.0 if idx % 2 == 0 else 30.0
        config3.append([x, y, rot])
    initial_configs.append(config3)

    # 4. Ring configuration (6 outer + 6 inner)
    config4 = []
    for i in range(6):
        angle = 2 * math.pi * i / 6
        r_outer = 2.5
        x = r_outer * math.cos(angle)
        y = r_outer * math.sin(angle)
        x_snap = round(x / LATTICE_SPACING) * LATTICE_SPACING
        y_snap = round(y / LATTICE_SPACING) * LATTICE_SPACING
        config4.append([x_snap, y_snap, 0.0])
    for i in range(6):
        angle = 2 * math.pi * i / 6 + math.pi/6
        r_inner = 1.2
        x = r_inner * math.cos(angle)
        y = r_inner * math.sin(angle)
        x_snap = round(x / LATTICE_SPACING) * LATTICE_SPACING
        y_snap = round(y / LATTICE_SPACING) * LATTICE_SPACING
        config4.append([x_snap, y_snap, 0.0])
    initial_configs.append(config4)

    # 5. Two rows
    config5 = []
    for row in range(2):
        y = (-1.0 if row == 0 else 1.0) * sqrt3 * 0.5
        y_snap = round(y / LATTICE_SPACING) * LATTICE_SPACING
        for col in range(6):
            x = -2.25 + col * 1.5
            if row == 1:
                x += 0.75
            x_snap = round(x / LATTICE_SPACING) * LATTICE_SPACING
            config5.append([x_snap, y_snap, 0.0])
    initial_configs.append(config5)

    # 6. Compact cluster (3x4 grid pattern)
    config6 = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 1.5
            y = (row - 1.0) * sqrt3
            x_snap = round(x / LATTICE_SPACING) * LATTICE_SPACING
            y_snap = round(y / LATTICE_SPACING) * LATTICE_SPACING
            config6.append([x_snap, y_snap, 0.0])
    initial_configs.append(config6)

    # Run local search from each initial config
    for init_config in initial_configs:
        if time.time() > deadline - 40.0:
            break

        # First check if initial config is valid
        L_init, valid_init = evaluate_config(init_config)
        if valid_init:
            if L_init < best_L:
                best_L = L_init
                best_positions = [p[:] for p in init_config]

        # Run local search
        positions, L = local_search(init_config, max_iterations=2500)
        if L < best_L:
            best_L = L
            best_positions = [p[:] for p in positions]

    # Phase 2: Fine-tuning with continuous optimization on best discrete solution
    if time.time() < deadline - 25.0:
        # Convert best discrete positions to continuous parameters
        params = []
        for p in best_positions:
            params.extend([p[0], p[1], p[2]])

        def objective_continuous(params):
            if time.time() > deadline:
                return 1e10
            positions = []
            for i in range(N):
                positions.append([params[3*i], params[3*i+1], params[3*i+2]])

            # Check validity
            if not is_valid_config(positions):
                return 1e10

            all_vertices = []
            for p in positions:
                all_vertices.extend(get_hex_vertices(p[0], p[1], p[2]))
            return get_min_L(all_vertices)

        # Try multiple Nelder-Mead runs with different initial perturbations
        best_cont_L = best_L
        best_cont_positions = [p[:] for p in best_positions]
        
        for trial in range(3):
            if time.time() > deadline - 25.0:
                break
            
            if trial == 0:
                trial_params = params[:]
            else:
                # Add small random perturbation
                import random
                trial_params = []
                for i in range(N):
                    dx = random.uniform(-0.05, 0.05)
                    dy = random.uniform(-0.05, 0.05)
                    trial_params.extend([params[3*i] + dx, params[3*i+1] + dy, params[3*i+2]])

            try:
                res = minimize(objective_continuous, trial_params, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
                if res.fun < best_cont_L and res.fun < 1e9:
                    # Verify the result is valid
                    final_positions = []
                    for i in range(N):
                        final_positions.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
                    if is_valid_config(final_positions):
                        all_vertices = []
                        for p in final_positions:
                            all_vertices.extend(get_hex_vertices(p[0], p[1], p[2]))
                        L_final = get_min_L(all_vertices)
                        if L_final < best_cont_L:
                            best_cont_L = L_final
                            best_cont_positions = final_positions
            except Exception:
                pass

        if best_cont_L < best_L:
            best_L = best_cont_L
            best_positions = best_cont_positions

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        params = []
        for p in best_positions:
            params.extend([p[0], p[1], p[2]])

        def objective_polish(params):
            if time.time() > deadline:
                return 1e10
            positions = []
            for i in range(N):
                positions.append([params[3*i], params[3*i+1], params[3*i+2]])

            if not is_valid_config(positions):
                return 1e10

            all_vertices = []
            for p in positions:
                all_vertices.extend(get_hex_vertices(p[0], p[1], p[2]))
            return get_min_L(all_vertices)

        try:
            res = minimize(objective_polish, params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_L and res.fun < 1e9:
                final_positions = []
                for i in range(N):
                    final_positions.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
                if is_valid_config(final_positions):
                    all_vertices = []
                    for p in final_positions:
                        all_vertices.extend(get_hex_vertices(p[0], p[1], p[2]))
                    L_final = get_min_L(all_vertices)
                    if L_final < best_L:
                        best_L = L_final
                        best_positions = final_positions
        except Exception:
            pass

    # Final verification
    if not is_valid_config(best_positions):
        # Fallback to grid
        best_positions = []
        for i in init_grid_idx:
            x, y = grid_centers[i]
            best_positions.append([x * 1.000001, y * 1.000001, 0.0])

    all_vertices = []
    for p in best_positions:
        all_vertices.extend(get_hex_vertices(p[0], p[1], p[2]))
    L_final = get_min_L(all_vertices)

    return best_positions, (0.0, 0.0), L_final, 0.0
