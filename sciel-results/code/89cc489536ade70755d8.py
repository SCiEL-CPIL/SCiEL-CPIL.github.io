"""Combined hexagon packing with rotation optimization and scipy-based refinement."""
import numpy as np
import math
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a

        apothem = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(30 + k * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4 (proper intersection only)."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hexagons_overlap(i1, i2, tol=1e-7):
        """Check if two unit hexagons overlap (return True if they do)."""
        x1, y1, a1 = i1
        x2, y2, a2 = i2
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        # Quick reject: max distance between any two points in two unit hexagons
        # is at most 2 (center-to-center) + 1 + 1 = 4, but for overlap need closer
        if dist > 2.0 + 1e-6:
            return False

        v1 = hexagon_vertices(x1, y1, 1.0, a1)
        v2 = hexagon_vertices(x2, y2, 1.0, a2)

        # Check if any vertex of one is inside the other
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, x2, y2, 1.0, a2):
                return True
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, x1, y1, 1.0, a1):
                return True

        # Check edge intersections
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return True
        return False

    def all_disjoint(inner_list, tol=1e-7):
        """Check all pairs are disjoint."""
        n = len(inner_list)
        for i in range(n):
            for j in range(i+1, n):
                if hexagons_overlap(inner_list[i], inner_list[j], tol):
                    return False
        return True

    def min_outer_side(inner_list, outer_center, outer_angle):
        """Binary search for minimum outer side containing all inner hexagons."""
        cx, cy = outer_center
        lo = 1.0
        hi = 20.0

        def fits(side):
            for ix, iy, iang in inner_list:
                verts = hexagon_vertices(ix, iy, 1.0, iang)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, cx, cy, side, outer_angle):
                        return False
            return True

        if not fits(hi):
            return None
        if fits(lo):
            return lo

        for _ in range(35):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi + 1e-10

    def evaluate_packing(inner_list, outer_center, outer_angle):
        """Evaluate packing. Returns (side, valid)."""
        if not all_disjoint(inner_list):
            return None, False
        side = min_outer_side(inner_list, outer_center, outer_angle)
        if side is None:
            return None, False
        return side, True

    # --- Generate hexagonal cluster initial positions ---
    def axial_to_cartesian(q, r):
        x = 2.0 * q + r
        y = sqrt3 * r
        return x, y

    def hex_ring_positions(ring):
        if ring == 0:
            return [(0, 0)]
        positions = []
        q, r = ring, 0
        for side_idx in range(6):
            for step in range(ring):
                positions.append((q, r))
                dq = [1, 0, -1, -1, 0, 1][side_idx]
                dr = [0, -1, -1, 0, 1, 1][side_idx]
                q += dq
                r += dr
        return positions

    all_positions = [(0, 0)]
    ring = 1
    while len(all_positions) < N:
        all_positions.extend(hex_ring_positions(ring))
        ring += 1
    selected = all_positions[:N]

    # Build initial inner list with centered positions
    initial_inner = []
    for q, r in selected:
        x, y = axial_to_cartesian(q, r)
        initial_inner.append([x, y, 0.0])

    # Center the cluster
    cx_mean = sum(p[0] for p in initial_inner) / len(initial_inner)
    cy_mean = sum(p[1] for p in initial_inner) / len(initial_inner)
    initial_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in initial_inner]

    best_side = float('inf')
    best_result = None

    # --- Strategy 1: Evaluate initial lattice with various outer angles ---
    for outer_angle in [0.0, 30.0, 15.0, 45.0, 75.0, 60.0, 20.0, 50.0]:
        if time.time() > deadline - 20:
            break
        side, valid = evaluate_packing(initial_inner, (0.0, 0.0), outer_angle)
        if valid and side < best_side:
            best_side = side
            best_result = (initial_inner, (0.0, 0.0), side, outer_angle)

    # --- Strategy 2: Optimize positions and angles using scipy ---
    # Variables: 2*N positions + N angles = 3*N variables
    def objective(params):
        """Objective: outer side length + penalty for overlaps."""
        if time.time() > deadline - 10:
            return 1e10
        inner = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i + 1]
            a = params[2*N + i]
            inner.append([x, y, a])

        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(inner[i], inner[j]):
                    return 1e6

        # Center the cluster
        cx_m = sum(p[0] for p in inner) / N
        cy_m = sum(p[1] for p in inner) / N
        centered = [[p[0] - cx_m, p[1] - cy_m, p[2]] for p in inner]

        # Try a few outer angles
        best_s = float('inf')
        for oa in [0.0, 30.0]:
            side = min_outer_side(centered, (0.0, 0.0), oa)
            if side is not None and side < best_s:
                best_s = side

        if best_s == float('inf'):
            return 1e6
        return best_s

    if time.time() < deadline - 40:
        # Initial params: positions + angles (all 0)
        x0 = []
        for p in initial_inner:
            x0.append(p[0])
            x0.append(p[1])
        x0.extend([0.0] * N)

        # Use Nelder-Mead first (faster, no gradient needed)
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                inner_opt = []
                for i in range(N):
                    inner_opt.append([res.x[2*i], res.x[2*i+1], res.x[2*N+i]])
                cx_m = sum(p[0] for p in inner_opt) / N
                cy_m = sum(p[1] for p in inner_opt) / N
                inner_opt = [[p[0]-cx_m, p[1]-cy_m, p[2]] for p in inner_opt]
                for oa in [0.0, 30.0, 15.0, 45.0]:
                    side, valid = evaluate_packing(inner_opt, (0.0, 0.0), oa)
                    if valid and side < best_side:
                        best_side = side
                        best_result = (inner_opt, (0.0, 0.0), side, oa)
        except Exception:
            pass

    # --- Strategy 3: Differential evolution for global optimization ---
    if time.time() < deadline - 30:
        # Bounds for positions: [-5, 5], angles: [0, 60]
        bounds = []
        for i in range(N):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
        for i in range(N):
            bounds.append((0.0, 60.0))

        try:
            res = differential_evolution(objective, bounds, seed=42,
                                         maxiter=50, tol=1e-5,
                                         popsize=8, polish=True,
                                         init='sobol')
            if res.fun < best_side:
                inner_opt = []
                for i in range(N):
                    inner_opt.append([res.x[2*i], res.x[2*i+1], res.x[2*N+i]])
                cx_m = sum(p[0] for p in inner_opt) / N
                cy_m = sum(p[1] for p in inner_opt) / N
                inner_opt = [[p[0]-cx_m, p[1]-cy_m, p[2]] for p in inner_opt]
                for oa in [0.0, 30.0, 15.0, 45.0, 60.0]:
                    side, valid = evaluate_packing(inner_opt, (0.0, 0.0), oa)
                    if valid and side < best_side:
                        best_side = side
                        best_result = (inner_opt, (0.0, 0.0), side, oa)
        except Exception:
            pass

    # --- Strategy 4: Local refinement around best found ---
    if best_result is not None and time.time() < deadline - 15:
        base_inner = [list(p) for p in best_result[0]]
        base_angle = best_result[3]

        # Try small perturbations of positions and angles
        rng = np.random.default_rng(42)
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            perturbed = []
            for p in base_inner:
                dx = rng.normal(0, 0.05)
                dy = rng.normal(0, 0.05)
                da = rng.normal(0, 5.0)
                perturbed.append([p[0]+dx, p[1]+dy, p[2]+da])

            cx_m = sum(p[0] for p in perturbed) / N
            cy_m = sum(p[1] for p in perturbed) / N
            centered = [[p[0]-cx_m, p[1]-cy_m, p[2]] for p in perturbed]

            side, valid = evaluate_packing(centered, (0.0, 0.0), base_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered, (0.0, 0.0), side, base_angle)

    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_result
# EVOLVE-BLOCK-END
