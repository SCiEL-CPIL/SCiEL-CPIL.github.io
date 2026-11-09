"""Combined hexagon packing: exact geometry + smooth edge-distance force optimization."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150 s limit

    sqrt3 = math.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
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

    def seg_seg_min_dist(p1, p2, p3, p4):
        """Minimum distance between segments p1-p2 and p3-p4 (also 0 if they intersect)."""
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        d2 = (p4[0] - p3[0], p4[1] - p3[1])
        r = (p1[0] - p3[0], p1[1] - p3[1])
        a = d1[0] * d1[0] + d1[1] * d1[1]
        e = d2[0] * d2[0] + d2[1] * d2[1]
        f = d2[0] * r[0] + d2[1] * r[1]
        EPS = 1e-12
        if a <= EPS and e <= EPS:
            return math.hypot(r[0], r[1])
        if a <= EPS:
            s = 0.0
            t = f / e
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            px = p1[0]
            py = p1[1]
            qx = p3[0] + t * d2[0] - px
            qy = p3[1] + t * d2[1] - py
            return math.hypot(qx, qy)
        c = d1[0] * r[0] + d1[1] * r[1]
        if e <= EPS:
            t = 0.0
            s = c / a
            s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
            px = p1[0] + s * d1[0]
            py = p1[1] + s * d1[1]
            qx = p3[0] - px
            qy = p3[1] - py
            return math.hypot(qx, qy)
        b = d1[0] * d2[0] + d1[1] * d2[1]
        denom = a * e - b * b
        s = (b * f - c * e) / denom if denom != 0 else 0.0
        s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        t = (b * s + f) / e
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        px = p1[0] + s * d1[0]
        py = p1[1] + s * d1[1]
        qx = p3[0] + t * d2[0] - px
        qy = p3[1] + t * d2[1] - py
        return math.hypot(qx, qy)

    def hex_edge_min_dist(h1, h2):
        """Minimum distance between the boundaries of two hexagons (0 if they touch/cross)."""
        best = float('inf')
        for i in range(6):
            a = h1[i]
            b = h1[(i + 1) % 6]
            for j in range(6):
                c = h2[j]
                d = h2[(j + 1) % 6]
                dd = seg_seg_min_dist(a, b, c, d)
                if dd < best:
                    best = dd
                    if best < 1e-9:
                        return 0.0
        return best

    def edge_collision(i1, i2):
        """Smooth signed clearance: >0 means the two unit hexagons overlap by that amount."""
        v1 = hexagon_vertices(i1[0], i1[1], 1.0, i1[2])
        v2 = hexagon_vertices(i2[0], i2[1], 1.0, i2[2])
        d = hex_edge_min_dist(v1, v2)
        return -d  # overlap if d < 0

    def hexagons_disjoint(i1, i2, threshold=1e-6):
        x1, y1, a1 = i1
        x2, y2, a2 = i2
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > 2.0 + threshold:
            return True
        if abs(a1 - a2) < 1e-6:
            return dist >= 2.0 - threshold
        v1 = hexagon_vertices(x1, y1, 1.0, a1)
        v2 = hexagon_vertices(x2, y2, 1.0, a2)
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, x2, y2, 1.0, a2):
                return False
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, x1, y1, 1.0, a1):
                return False
        return True

    def all_disjoint(inner_list, threshold=1e-6):
        n = len(inner_list)
        for i in range(n):
            for j in range(i + 1, n):
                if not hexagons_disjoint(inner_list[i], inner_list[j], threshold):
                    return False
        return True

    def max_radial(inner_list, cx, cy, outer_angle):
        """Largest value of max_k n_k . p over all inner vertices, in the outer frame."""
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        norms = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            norms.append((math.cos(theta), math.sin(theta)))
        best = 0.0
        for ix, iy, iang in inner_list:
            verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in verts:
                rx = (vx - cx) * cos_a - (vy - cy) * sin_a
                ry = (vx - cx) * sin_a + (vy - cy) * cos_a
                for nx, ny in norms:
                    val = rx * nx + ry * ny
                    if val > best:
                        best = val
        return best

    def min_outer_side(inner_list, outer_center, outer_angle):
        cx, cy = outer_center
        rmax = max_radial(inner_list, cx, cy, outer_angle)
        return rmax / (sqrt3 / 2) + 1e-10

    def evaluate_packing(inner_list, outer_center, outer_angle):
        if not all_disjoint(inner_list):
            return None, False
        side = min_outer_side(inner_list, outer_center, outer_angle)
        return side, True

    # --- hexagonal lattice generation ---
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

    def make_lattice():
        inner = []
        for q, r in selected:
            x, y = axial_to_cartesian(q, r)
            inner.append([x, y, 0.0])
        cx_mean = sum(p[0] for p in inner) / len(inner)
        cy_mean = sum(p[1] for p in inner) / len(inner)
        return [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]

    best_side = float('inf')
    best_result = None

    def consider(inner_list, outer_angle):
        nonlocal best_side, best_result
        side, valid = evaluate_packing(inner_list, (0.0, 0.0), outer_angle)
        if valid and side < best_side:
            best_side = side
            best_result = (inner_list, (0.0, 0.0), side, outer_angle)

    # Strategy 1: lattice, several outer angles
    lattice = make_lattice()
    for outer_angle in [0.0, 30.0, 15.0, 45.0, 75.0]:
        if time.time() > deadline - 40:
            break
        consider(lattice, outer_angle)

    # --- force-directed optimization with smooth edge-distance repulsion ---
    def force_directed_optimize(initial_positions, max_iters=3000):
        n = len(initial_positions)
        angles = [p[2] for p in initial_positions]
        pos = np.array([[p[0], p[1]] for p in initial_positions], dtype=float)
        lr = 0.08
        decay = 0.9995
        min_dist = 2.0
        prev_max = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            forces = np.zeros_like(pos)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pos[i] - pos[j]
                    dist = float(np.linalg.norm(diff))
                    if dist < 1e-6:
                        diff = np.array([0.01, 0.0])
                        dist = 0.01
                    # smooth edge-based clearance (primary)
                    overlap = edge_collision([pos[i][0], pos[i][1], angles[i]],
                                             [pos[j][0], pos[j][1], angles[j]])
                    if overlap > 1e-9:
                        forces[i] += (overlap * 4.0) * diff / dist
                        forces[j] -= (overlap * 4.0) * diff / dist
                    # center-distance safety net
                    if dist < min_dist - 0.3:
                        f = (min_dist - dist) * 1.5
                        forces[i] += f * diff / dist
                        forces[j] -= f * diff / dist
            # compacting inward attraction
            for i in range(n):
                to_center = -pos[i]
                d = float(np.linalg.norm(to_center))
                if d > 1e-6:
                    forces[i] += 0.15 * to_center / d
            step = lr * forces
            maxstep = float(np.max(np.abs(step))) if step.size else 0.0
            pos += step
            lr *= decay
            if maxstep < 1e-4:
                break
        inner = [[pos[i][0], pos[i][1], angles[i]] for i in range(n)]
        cx_mean = float(np.mean([p[0] for p in inner]))
        cy_mean = float(np.mean([p[1] for p in inner]))
        return [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]

    if time.time() < deadline - 30:
        for outer_angle in [0.0, 30.0]:
            if time.time() > deadline - 20:
                break
            opt = force_directed_optimize(lattice, max_iters=2500)
            consider(opt, outer_angle)

    # --- local Nelder-Mead polish over centered positions ---
    def nm_objective(centered_positions, outer_angle):
        side = min_outer_side(centered_positions, (0.0, 0.0), outer_angle)
        pen = 0.0
        n = len(centered_positions)
        for i in range(n):
            for j in range(i + 1, n):
                ov = edge_collision(centered_positions[i], centered_positions[j])
                if ov > 0:
                    pen += ov * ov
        return side + 50.0 * pen

    def nelder_mead(start_pos, outer_angle, max_iters=1500):
        n = len(start_pos)
        pts = [list(p) for p in start_pos]
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        eps = 1e-6
        f0 = nm_objective(pts, outer_angle)
        simplex = [list(pts)]
        for d in range(n):
            v = list(pts)
            v[d][0] += 0.05
            simplex.append(v)
        vals = [nm_objective(p, outer_angle) for p in simplex]
        for it in range(max_iters):
            if time.time() > deadline - 12:
                break
            order = sorted(range(len(simplex)), key=lambda i: vals[i])
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]
            if abs(vals[0] - vals[-1]) < eps:
                break
            centroid = [[sum(simplex[i][d][c] for i in range(len(simplex) - 1)) / (len(simplex) - 1)
                         for c in range(2)] for d in range(n)]
            worst = simplex[-1]
            xr = [centroid[d][c] + alpha * (centroid[d][c] - worst[d][c]) for d in range(n) for c in range(2)]
            xr = [[xr[d * 2], xr[d * 2 + 1]] for d in range(n)]
            fr = nm_objective(xr, outer_angle)
            if fr < vals[0]:
                xe = [centroid[d][c] + gamma * (centroid[d][c] - worst[d][c]) for d in range(n) for c in range(2)]
                xe = [[xe[d * 2], xe[d * 2 + 1]] for d in range(n)]
                fe = nm_objective(xe, outer_angle)
                if fe < fr:
                    simplex[-1], vals[-1] = xe, fe
                else:
                    simplex[-1], vals[-1] = xr, fr
            elif fr < vals[-2]:
                simplex[-1], vals[-1] = xr, fr
            else:
                xc = [centroid[d][c] + rho * (worst[d][c] - centroid[d][c]) for d in range(n) for c in range(2)]
                xc = [[xc[d * 2], xc[d * 2 + 1]] for d in range(n)]
                fc = nm_objective(xc, outer_angle)
                if fc < vals[-1]:
                    simplex[-1], vals[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = [[simplex[0][d][c] + sigma * (simplex[i][d][c] - simplex[0][d][c])
                                       for c in range(2)] for d in range(n)]
                        vals[i] = nm_objective(simplex[i], outer_angle)
        best_i = min(range(len(simplex)), key=lambda i: vals[i])
        return simplex[best_i], vals[best_i]

    if time.time() < deadline - 15:
        if best_result is not None:
            base = best_result
        else:
            base = (lattice, (0.0, 0.0), float('inf'), 0.0)
        base_inner = [list(p) for p in base[0]]
        base_angle = base[3]
        polished, _ = nelder_mead(base_inner, base_angle, max_iters=1200)
        consider(polished, base_angle)

    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_result
# EVOLVE-BLOCK-END
