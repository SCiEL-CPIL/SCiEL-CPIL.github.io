"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return vertices of a regular hexagon with center (cx,cy), side length 'side',
        rotated by angle_deg degrees. Flat-top orientation at angle=0."""
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        verts = []
        for i in range(6):
            base_angle = math.radians(60 * i)
            vx = side * math.cos(base_angle)
            vy = side * math.sin(base_angle)
            # Rotate
            rx = vx * cos_a - vy * sin_a
            ry = vx * sin_a + vy * cos_a
            verts.append((cx + rx, cy + ry))
        return verts

    def point_in_hex(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside regular hexagon centered at (cx,cy)
        with side length 'side' and rotation angle_deg."""
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        # Translate point to hex center and rotate
        dx = px - cx
        dy = py - cy
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        # Check against 6 sides of hexagon
        # For a regular hexagon with flat-top orientation, the sides are at angles 0, 60, ..., 300
        # The distance from center to each side (apothem) is side * sqrt(3)/2
        apothem = side * sqrt3 / 2.0
        for i in range(6):
            angle = math.radians(60 * i)
            nx = math.cos(angle)
            ny = math.sin(angle)
            if rx * nx + ry * ny > apothem:
                return False
        return True

    def hex_contains_point(px, py, cx, cy, side, angle_deg):
        return point_in_hex(px, py, cx, cy, side, angle_deg)

    def check_inside_hexagon(vertices, cx, cy, L, outer_angle):
        """Check if all points are inside the outer hexagon."""
        for vx, vy in vertices:
            if not point_in_hex(vx, vy, cx, cy, L, outer_angle):
                return False
        return True

    def overlap_check(config, tolerance=1e-6):
        """Check if any two hexagons overlap. Returns True if no overlap."""
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                xi, yi, ai = config[i]
                xj, yj, aj = config[j]
                # Check if any vertex of hex i is inside hex j
                vi = hex_vertices(xi, yi, ai)
                for vx, vy in vi:
                    if point_in_hex(vx, vy, xj, yj, 1.0, aj):
                        return False
                # Check if any vertex of hex j is inside hex i
                vj = hex_vertices(xj, yj, aj)
                for vx, vy in vj:
                    if point_in_hex(vx, vy, xi, yi, 1.0, ai):
                        return False
        return True

    def min_L_for_config(config, max_outer_rotations=36):
        """Find minimum L such that all hexagons fit in outer hexagon."""
        # Collect all vertices
        all_verts = []
        for x, y, a in config:
            all_verts.extend(hex_vertices(x, y, a))

        # Find bounding box
        xs = [v[0] for v in all_verts]
        ys = [v[1] for v in all_verts]
        cx_avg = (min(xs) + max(xs)) / 2.0
        cy_avg = (min(ys) + max(ys)) / 2.0

        # Binary search on L
        def can_fit(L, outer_angle):
            return check_inside_hexagon(all_verts, cx_avg, cy_avg, L, outer_angle)

        lo = 0.0
        hi = 100.0

        # Find a valid upper bound
        while not can_fit(hi, 0.0):
            hi *= 1.5
            if hi > 1000:
                break

        best_L = hi
        best_angle = 0.0

        for outer_rot in range(0, 360, 10):
            lo2 = 0.0
            hi2 = hi
            for _ in range(20):
                mid = (lo2 + hi2) / 2.0
                if can_fit(mid, outer_rot):
                    hi2 = mid
                else:
                    lo2 = mid
            if hi2 < best_L:
                best_L = hi2
                best_angle = outer_rot

        # Refine angle around best
        for outer_rot in range(int(best_angle) - 10, int(best_angle) + 11, 2):
            lo2 = 0.0
            hi2 = hi
            for _ in range(20):
                mid = (lo2 + hi2) / 2.0
                if can_fit(mid, outer_rot):
                    hi2 = mid
                else:
                    lo2 = mid
            if hi2 < best_L:
                best_L = hi2
                best_angle = outer_rot

        return best_L, best_angle

    # Generate candidate configurations
    # Strategy: Place hexagons in a hexagonal grid pattern
    # For side-1 hexagons, the distance between adjacent centers is sqrt(3) * side = sqrt(3)

    best_config = None
    best_L = float('inf')
    best_outer_center = (0, 0)
    best_outer_angle = 0.0

    # Try different arrangements
    # 1) 3x4 grid in hexagonal pattern
    # 2) Center + ring
    # 3) Various other patterns

    # Hex grid placement: axial coordinates
    def axial_to_cartesian(q, r, size=1.0):
        x = size * 1.5 * q
        y = size * sqrt3 * (r + q / 2.0)
        return x, y

    # Pattern 1: 4x3 rectangular-ish hex grid
    configs = []

    # Pattern: 3 rows of 4, offset
    for offset_x in [0.0, 0.75, 1.5]:
        for offset_y in [0.0, sqrt3/2]:
            cfg = []
            for row in range(3):
                for col in range(4):
                    q = col - 1.5
                    r = row - 1.0
                    x, y = axial_to_cartesian(q, r)
                    x += offset_x
                    y += offset_y
                    cfg.append([x, y, 0.0])
            configs.append(cfg)

    # Pattern: centered arrangement
    # Center hex + 6 around + 5 more
    # Actually, let's try a more systematic approach
    # Place hexagons at positions that minimize the bounding hexagon

    # Try random-ish arrangements with local optimization
    import random
    random.seed(42)

    # Start with a reasonable configuration
    base_config = []
    # 4 hexagons in a row
    for k in range(4):
        base_config.append([k * sqrt3, 0.0, 0.0])
    # 4 hexagons in another row, offset
    for k in range(4):
        base_config.append([k * sqrt3 + sqrt3/2, sqrt3 * 1.5, 0.0])
    # 4 hexagons in third row
    for k in range(4):
        base_config.append([k * sqrt3, sqrt3 * 3.0, 0.0])

    configs.append(base_config)

    # Try different rotations of inner hexagons
    for rot_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
        rotated_config = []
        for x, y, a in base_config:
            rotated_config.append([x, y, a + rot_angle])
        configs.append(rotated_config)

    # Try more compact arrangements
    # Hexagonal cluster: center + 6 neighbors + 5 more
    cluster_configs = []
    # Standard hexagonal packing of 12 hexagons
    # Axial coordinates for 12 hexagons in a compact cluster
    hex_positions = [
        (0, 0),    # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # 6 neighbors
        (2, 0), (-2, 0), (1, 1), (-1, -1), (2, -1)  # 5 more
    ]

    cfg_cluster = []
    for q, r in hex_positions:
        x, y = axial_to_cartesian(q, r)
        cfg_cluster.append([x, y, 0.0])
    configs.append(cfg_cluster)

    # Try rotating the cluster
    for rot in [0.0, 30.0, 60.0]:
        cfg_rot = []
        for x, y, a in cfg_cluster:
            # Rotate position around center
            angle_rad = math.radians(rot)
            cos_r, sin_r = math.cos(angle_rad), math.sin(angle_rad)
            nx = x * cos_r - y * sin_r
            ny = x * sin_r + y * cos_r
            cfg_rot.append([nx, ny, a + rot])
        configs.append(cfg_rot)

    # Evaluate all configurations
    for cfg in configs:
        if time.time() > deadline:
            break

        # Check for overlaps
        if not overlap_check(cfg):
            continue

        L, outer_angle = min_L_for_config(cfg)
        if L < best_L:
            best_L = L
            best_config = cfg
            best_outer_angle = outer_angle

    # Local optimization: perturb each hexagon's position and angle
    if best_config is not None and time.time() < deadline:
        current_config = [list(c) for c in best_config]
        current_L, current_outer_angle = min_L_for_config(current_config)
        best_local_L = current_L
        best_local_config = [list(c) for c in current_config]
        best_local_outer_angle = current_outer_angle

        step = 0.1
        for iteration in range(200):
            if time.time() > deadline:
                break

            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        for da in [-10, 0, 10]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            new_config = [list(c) for c in current_config]
                            new_config[i] = [current_config[i][0] + dx,
                                             current_config[i][1] + dy,
                                             current_config[i][2] + da]

                            if not overlap_check(new_config):
                                continue

                            L_test, angle_test = min_L_for_config(new_config)
                            if L_test < best_local_L - 1e-9:
                                best_local_L = L_test
                                best_local_config = new_config
                                best_local_outer_angle = angle_test
                                current_config = new_config
                                improved = True

            if not improved:
                step *= 0.7
                if step < 0.01:
                    break

        if best_local_L < best_L:
            best_L = best_local_L
            best_config = best_local_config
            best_outer_angle = best_local_outer_angle

    # Final: compute outer center as centroid of all vertices
    if best_config is None:
        # Fallback
        inner = [[k * sqrt3, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 20.0, 0.0

    all_verts = []
    for x, y, a in best_config:
        all_verts.extend(hex_vertices(x, y, a))

    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0

    # Verify and refine
    L_final, angle_final = min_L_for_config(best_config)

    inner = [[c[0], c[1], c[2]] for c in best_config]
    return inner, (cx, cy), L_final, angle_final

# EVOLVE-BLOCK-END
