"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Precompute cos/sin for common angles
    ANGLES = [0, 30, 60, 90, 120, 150]
    ANGLE_RAD = [math.radians(a) for a in ANGLES]

    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def point_in_polygon(p, polygon):
        """Ray casting point in polygon test."""
        x, y = p
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

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        def point_on_segment(a, b, p):
            cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if abs(cr) > 1e-9:
                return False
            dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
            length_sq = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
            return dot >= -1e-9 and dot <= length_sq + 1e-9

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        if d1 == 0 and point_on_segment(p3, p4, p1):
            return True
        if d2 == 0 and point_on_segment(p3, p4, p2):
            return True
        if d3 == 0 and point_on_segment(p1, p2, p3):
            return True
        if d4 == 0 and point_on_segment(p1, p2, p4):
            return True
        return False

    def hexagons_disjoint(verts_a, verts_b):
        """Check if two hexagons (given as vertex lists) are disjoint."""
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            if segments_intersect(verts_a[i], verts_a[(i+1) % 6], verts_b[i], verts_b[(i+1) % 6]):
                return False
        return True

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        """Check if all inner hexagons are inside outer and mutually disjoint."""
        outer_verts = hex_vertices(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = hex_vertices(ix, iy, 1.0, iangle)
            inner_verts.append(iv)
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """Find minimum S such that all inner hexagons fit in outer hexagon."""
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(hex_vertices(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def make_hexagonal_grid_config():
        """Create a hexagonal grid arrangement of N hexagons."""
        inner = []
        # Hexagonal packing: centers at distance sqrt(3) apart for non-overlapping unit hexagons
        d = math.sqrt(3) + 0.01  # slight gap
        # Generate hexagonal grid points
        points = []
        # Center
        points.append((0, 0))
        # First ring: 6 hexagons
        for i in range(6):
            angle = math.radians(60 * i)
            points.append((d * math.cos(angle), d * math.sin(angle)))
        # Second ring: 12 hexagons
        for i in range(6):
            angle = math.radians(60 * i)
            # 2 hexagons in each direction
            for j in [1, 2]:
                r = d * j
                points.append((r * math.cos(angle), r * math.sin(angle)))
            # 2 hexagons between directions
            angle2 = math.radians(60 * i + 30)
            r = d * math.sqrt(3)
            points.append((r * math.cos(angle2), r * math.sin(angle2)))
        
        # Take first N points
        points = points[:N]
        for (px, py) in points:
            inner.append([px, py, 0.0])
        return inner

    def make_linear_config():
        """Create a linear arrangement."""
        inner = []
        d = math.sqrt(3) + 0.01
        for k in range(-(N // 2), N - N // 2):
            inner.append([d * k, 0.0, 0.0])
        return inner

    # Generate initial configurations
    initial_configs = []
    
    # Linear config
    initial_configs.append(make_linear_config())
    
    # Hexagonal grid config
    initial_configs.append(make_hexagonal_grid_config())
    
    # Random configs
    rng_init = random.Random(42)
    for _ in range(5):
        inner = []
        for _ in range(N):
            x = rng_init.uniform(-4, 4)
            y = rng_init.uniform(-4, 4)
            ang = rng_init.choice(ANGLES)
            inner.append([x, y, ang])
        initial_configs.append(inner)

    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Evaluate initial configs
    for config in initial_configs:
        S = compute_outer_hex(config, 0.0, 0.0, 0.0)
        if S < best_S and all_hexagons_fit(config, 0.0, 0.0, S, 0.0):
            best_S = S
            best_inner = [list(x) for x in config]
            best_center = (0.0, 0.0)
            best_angle = 0.0

    # Local search
    def local_search(initial_inner, seed):
        nonlocal best_S, best_inner, best_center, best_angle
        rng = random.Random(seed)
        cur_inner = [list(x) for x in initial_inner]
        cur_S = compute_outer_hex(cur_inner, 0.0, 0.0, 0.0)
        
        if not all_hexagons_fit(cur_inner, 0.0, 0.0, cur_S, 0.0):
            return
        
        for it in range(3000):
            if time.time() > deadline:
                return
            
            # Perturb a random inner hex
            idx = rng.randrange(N)
            old_x = cur_inner[idx][0]
            old_y = cur_inner[idx][1]
            old_ang = cur_inner[idx][2]
            
            step = max(0.05, 0.3 * (1 - it / 5000.0))
            cur_inner[idx][0] += rng.gauss(0, step)
            cur_inner[idx][1] += rng.gauss(0, step)
            if rng.random() < 0.15:
                cur_inner[idx][2] = rng.choice(ANGLES)
            
            S_new = compute_outer_hex(cur_inner, 0.0, 0.0, 0.0)
            
            if S_new < cur_S and all_hexagons_fit(cur_inner, 0.0, 0.0, S_new, 0.0):
                cur_S = S_new
                if S_new < best_S:
                    best_S = S_new
                    best_inner = [list(x) for x in cur_inner]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                # Revert
                cur_inner[idx][0] = old_x
                cur_inner[idx][1] = old_y
                cur_inner[idx][2] = old_ang

    # Run local search on each initial config
    seed = 0
    for config in initial_configs:
        if time.time() > deadline:
            break
        local_search(config, seed)
        seed += 1

    # Additional random restarts
    while time.time() < deadline:
        rng_restart = random.Random(seed)
        inner = []
        for _ in range(N):
            x = rng_restart.uniform(-5, 5)
            y = rng_restart.uniform(-5, 5)
            ang = rng_restart.choice(ANGLES)
            inner.append([x, y, ang])
        local_search(inner, seed)
        seed += 1

    if best_inner is None:
        # Fallback to linear config
        inner = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        return inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
