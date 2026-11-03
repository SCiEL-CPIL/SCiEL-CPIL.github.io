"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    def hex_support(angle, rot_deg):
        """Support function of a unit regular hexagon (side=1) rotated by rot_deg,
        in the direction given by angle (radians)."""
        # Unit hexagon vertices at angle 0 (pointy-right):
        # vertices at angles 0, 60, 120, 180, 240, 300 degrees
        rot = math.radians(rot_deg)
        max_proj = 0.0
        for k in range(6):
            va = math.radians(60 * k) + rot
            vx = math.cos(va)
            vy = math.sin(va)
            proj = vx * math.cos(angle) + vy * math.sin(angle)
            if proj > max_proj:
                max_proj = proj
        return max_proj

    def hex_hex_min_dist(rot_a_deg, rot_b_deg):
        """Minimum distance between two unit hexagons (side=1) with rotations rot_a, rot_b.
        This is the minimum over all directions of (support_a + support_b)."""
        # The minimum separation distance between two convex polygons is the
        # minimum over directions d of (h_A(d) + h_B(-d))
        # For regular hexagons, we can sample the 12 critical directions.
        rot_a = math.radians(rot_a_deg)
        rot_b = math.radians(rot_b_deg)
        min_sep = 1e10
        # Critical directions: normals of edges of both hexagons
        angles = set()
        for k in range(6):
            angles.add(math.radians(30 + 60 * k) + rot_a)
            angles.add(math.radians(30 + 60 * k) + rot_b)
            angles.add(math.radians(30 + 60 * k) + rot_a + math.pi)
            angles.add(math.radians(30 + 60 * k) + rot_b + math.pi)
        for ang in angles:
            sa = hex_support(ang, rot_a_deg)
            sb = hex_support(ang + math.pi, rot_b_deg)
            sep = sa + sb
            if sep < min_sep:
                min_sep = sep
        return min_sep

    def calc_outer_side(positions, rotations, theta_deg):
        """Compute the minimum side length S of an outer regular hexagon
        centered at origin, rotated by theta_deg, that contains all inner hexagons."""
        theta_rad = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + 60 * k) + theta_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)

            for i in range(N):
                x, y = positions[i]
                rot_i = rotations[i]
                h = hex_support(normal_angle, rot_i)
                proj = x * nx + y * ny + h
                if proj > max_proj:
                    max_proj = proj
        return max_proj * 2.0 / sqrt3

    def make_objective(positions_init, rotations_init, theta_init):
        """Create objective function for optimization."""
        def objective(params):
            if time.time() > deadline:
                return 1e10
            pos = [(params[2*i], params[2*i+1]) for i in range(N)]
            rots = [params[2*N + i] for i in range(N)]
            theta = params[3*N]

            S = calc_outer_side(pos, rots, theta)

            # Penalty for overlaps: hexagon-hexagon minimum distance
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_sep = hex_hex_min_dist(rots[i], rots[j])
                    if dist < min_sep - 1e-9:
                        penalty += (min_sep - dist) ** 2

            return S + 50.0 * penalty

        return objective

    # Generate diverse initial configurations
    def honeycomb_init():
        """12 hexagons in a honeycomb-like arrangement."""
        centers = []
        # 4-3-4-1 or similar
        rows = [
            [(0, 0), (2, 0), (4, 0)],
            [(1, sqrt3), (3, sqrt3), (5, sqrt3)],
            [(0, 2*sqrt3), (2, 2*sqrt3), (4, 2*sqrt3)],
            [(1, 3*sqrt3), (3, 3*sqrt3)],
        ]
        for row in rows:
            for x, y in row:
                centers.append((x, y))
        # Center
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        centered = [(x - cx, y - cy) for x, y in centers]
        rots = [0.0] * N
        theta = 0.0
        return centered, rots, theta

    def ring_init():
        """6 in a ring around center, 6 more."""
        centers = []
        # Inner ring of 6
        for k in range(6):
            a = math.radians(60 * k)
            centers.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
        # Outer 6
        for k in range(6):
            a = math.radians(60 * k + 30)
            centers.append((4.0 * math.cos(a), 4.0 * math.sin(a)))
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        centered = [(x - cx, y - cy) for x, y in centers]
        rots = [0.0] * N
        theta = 0.0
        return centered, rots, theta

    def cluster_init():
        """19-hex cluster minus 7 outer ones."""
        centers = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                    x = q + r / 2.0
                    y = r * sqrt3 / 2.0
                    centers.append((x, y))
        # Remove 7 outermost
        centers.sort(key=lambda p: -(p[0]**2 + p[1]**2))
        centers = centers[:12]
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        centered = [(x - cx, y - cy) for x, y in centers]
        rots = [0.0] * N
        theta = 0.0
        return centered, rots, theta

    def linear_init():
        """Two rows of 6."""
        centers = []
        for k in range(6):
            centers.append((2.0*k, 0.0))
            centers.append((2.0*k + 1.0, sqrt3))
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        centered = [(x - cx, y - cy) for x, y in centers]
        rots = [0.0] * N
        theta = 0.0
        return centered, rots, theta

    def hex_rotated_init():
        """Honeycomb with some rotations."""
        pos, rots, theta = honeycomb_init()
        rots = [0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0]
        return pos, rots, theta

    def compact_init():
        """Tighter honeycomb: 3-4-3-2."""
        centers = []
        # Row 0: 3 hexes
        for k in range(3):
            centers.append((2.0*k, 0.0))
        # Row 1: 4 hexes
        for k in range(4):
            centers.append((1.0 + 2.0*k, sqrt3))
        # Row 2: 3 hexes
        for k in range(3):
            centers.append((2.0*k, 2*sqrt3))
        # Row 3: 2 hexes
        for k in range(2):
            centers.append((1.0 + 2.0*k, 3*sqrt3))
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        centered = [(x - cx, y - cy) for x, y in centers]
        rots = [0.0] * N
        theta = 0.0
        return centered, rots, theta

    inits = [
        honeycomb_init(),
        ring_init(),
        cluster_init(),
        linear_init(),
        hex_rotated_init(),
        compact_init(),
    ]

    best_S = 1e10
    best_params = None

    for init_idx, (pos_init, rots_init, theta_init) in enumerate(inits):
        if time.time() > deadline:
            break

        init_flat = []
        for x, y in pos_init:
            init_flat.extend([x, y])
        for r in rots_init:
            init_flat.append(r)
        init_flat.append(theta_init)

        obj = make_objective(pos_init, rots_init, theta_init)
        score = obj(init_flat)
        if score < best_S:
            best_S = score
            best_params = init_flat[:]

        # Local optimization
        try:
            res = minimize(obj, init_flat, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9,
                                    'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

        # Try a second pass with smaller initial step
        if time.time() < deadline - 10.0:
            try:
                res2 = minimize(obj, best_params if best_params is not None else init_flat,
                                method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                         'adaptive': True})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_params = res2.x
            except Exception:
                pass

    if best_params is None:
        # Fallback
        pos, rots, theta = honeycomb_init()
        S = calc_outer_side(pos, rots, theta)
        inner = [[x, y, r] for (x, y), r in zip(pos, rots)]
        return inner, (0.0, 0.0), S, theta

    # Extract final result
    pos = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
    rots = [best_params[2*N + i] for i in range(N)]
    theta = best_params[3*N]

    # Re-center: compute centroid and shift
    cx = sum(p[0] for p in pos) / N
    cy = sum(p[1] for p in pos) / N
    pos = [(x - cx, y - cy) for x, y in pos]

    final_S = calc_outer_side(pos, rots, theta)

    # Verify no overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            min_sep = hex_hex_min_dist(rots[i], rots[j])
            if dist < min_sep - 1e-6:
                # Nudge apart
                if dist > 1e-9:
                    push = (min_sep - dist + 0.01) / 2.0
                    ux, uy = dx/dist, dy/dist
                    pos[i] = (pos[i][0] + push*ux, pos[i][1] + push*uy)
                    pos[j] = (pos[j][0] - push*ux, pos[j][1] - push*uy)
                else:
                    pos[i] = (pos[i][0] + 0.1, pos[i][1])
                    pos[j] = (pos[j][0] - 0.1, pos[j][1])

    # Recompute after nudging
    cx = sum(p[0] for p in pos) / N
    cy = sum(p[1] for p in pos) / N
    pos = [(x - cx, y - cy) for x, y in pos]
    final_S = calc_outer_side(pos, rots, theta)

    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], rots[i]])

    return inner, (0.0, 0.0), final_S, theta
# EVOLVE-BLOCK-END
