"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # --- Geometry helpers for regular hexagons (flat-top, side length 1) ---
    # A regular hexagon with side s, center (cx,cy), rotation theta has vertices:
    # v_k = (cx + s*cos(theta + k*pi/3), cy + s*sin(theta + k*pi/3)) for k=0..5

    def hex_vertices(cx, cy, s, theta):
        angles = theta + np.arange(6) * np.pi / 3.0
        return np.column_stack([cx + s * np.cos(angles),
                                cy + s * np.sin(angles)])

    def polygon_area(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def polygon_overlap_area(pts_a, pts_b):
        """Compute intersection area of two convex polygons via Sutherland-Hodgman."""
        def clip(subject, clipper):
            def inside(p, a, b):
                return np.cross(b - a, p - a) >= -1e-12
            def intersection(p, q, a, b):
                # line segment pq clipped by line ab
                d1 = np.cross(b - a, p - a)
                d2 = np.cross(b - a, q - a)
                if abs(d1 - d2) < 1e-14:
                    return p
                t = d1 / (d1 - d2)
                return p + t * (q - p)
            output = list(subject)
            for i in range(len(clipper)):
                a = clipper[i]
                b = clipper[(i + 1) % len(clipper)]
                if not output:
                    break
                input_list = output
                output = []
                for j in range(len(input_list)):
                    cur = input_list[j]
                    prev = input_list[j - 1]
                    if inside(cur, a, b):
                        if not inside(prev, a, b):
                            output.append(intersection(prev, cur, a, b))
                        output.append(cur)
                    elif inside(prev, a, b):
                        output.append(intersection(prev, cur, a, b))
            return np.array(output) if len(output) > 2 else None

        # Clip A by B
        clipped = clip(pts_a, pts_b)
        if clipped is None or len(clipped) < 3:
            return 0.0
        return polygon_area(clipped)

    def max_support(pts, direction):
        """Max projection of polygon onto a unit direction vector."""
        return np.max(pts @ direction)

    def min_support(pts, direction):
        return np.min(pts @ direction)

    def hexagon_fits_in_outer(inner_pts, outer_pts):
        """Check if inner polygon is fully inside outer polygon."""
        # For each edge of outer, check all inner vertices are on the correct side
        for i in range(len(outer_pts)):
            a = outer_pts[i]
            b = outer_pts[(i + 1) % len(outer_pts)]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            # All inner points should be on the same side as the edge interior
            # The interior is to the left of the edge (for CCW polygon)
            d_inner = np.min((inner_pts - a) @ normal)
            if d_inner < -1e-9:
                return False
        return True

    def compute_outer_side(inner_positions, inner_angles, outer_center, outer_rotation):
        """
        Given inner hexagon centers/angles and outer center/rotation,
        find the minimum outer side length such that all inner hexagons fit.
        The outer hexagon has vertices at outer_center + s * (cos(rot+k*pi/3), sin(rot+k*pi/3)).
        We need to find minimum s such that all inner hexagons are inside.
        
        For a fixed outer rotation and center, the constraint for each inner hexagon
        is that all its vertices must be inside the outer hexagon.
        This is equivalent to: for each of the 6 edges of the outer hexagon,
        the max projection of the inner hexagon onto the outward normal of that edge
        must be <= the projection of the outer hexagon's edge.
        
        Actually, let's use a simpler approach: binary search on s and check containment.
        """
        def check(s):
            outer_pts = hex_vertices(outer_center[0], outer_center[1], s, outer_rotation)
            for pos, ang in zip(inner_positions, inner_angles):
                inner_pts = hex_vertices(pos[0], pos[1], 1.0, ang)
                if not hexagon_fits_in_outer(inner_pts, outer_pts):
                    return False
            return True

        # Binary search for minimum s
        lo, hi = 1.0, 50.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if check(mid):
                hi = mid
            else:
                lo = mid
        return hi

    # --- Objective function for optimization ---
    # Variables: for each inner hexagon: (x, y, angle) => 3*N vars
    # Plus: outer_center_x, outer_center_y, outer_rotation => 3 more
    # Total: 3*N + 3 = 39 variables

    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        inner_params = params[:3 * N]
        outer_params = params[3 * N:]
        inner_positions = inner_params.reshape(N, 2)
        inner_angles = inner_params[2::3] if False else inner_params[2::3]  # wrong, fix below
        # Actually: inner_params is 3*N: [x0,y0,a0, x1,y1,a1, ...]
        inner_positions = inner_params[0::3] + 1j * inner_params[1::3]  # no, keep real
        inner_positions = np.column_stack([inner_params[0::3], inner_params[1::3]])
        inner_angles = inner_params[2::3]
        outer_cx = outer_params[0]
        outer_cy = outer_params[1]
        outer_rot = outer_params[2]

        # Compute required outer side
        s = compute_outer_side(inner_positions, inner_angles, (outer_cx, outer_cy), outer_rot)

        # Overlap penalty between inner hexagons
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                pts_i = hex_vertices(inner_positions[i, 0], inner_positions[i, 1], 1.0, inner_angles[i])
                pts_j = hex_vertices(inner_positions[j, 0], inner_positions[j, 1], 1.0, inner_angles[j])
                ov = polygon_overlap_area(pts_i, pts_j)
                overlap_penalty += ov * ov  # quadratic penalty

        # Also penalize if outer side is too large
        return s + 10.0 * overlap_penalty

    # --- Initial guess ---
    # Place hexagons in a rough hexagonal pattern
    def initial_guess():
        # Hexagonal lattice arrangement
        centers = []
        angles = []
        # 3 rows: 4, 4, 4 or a hexagonal cluster
        # Let's try a 3x4 grid with hexagonal offset
        dx = 2.1  # slightly more than 2 to avoid overlap
        dy = 1.85
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx * 0.5
                y = row * dy
                centers.append([x - 3 * dx * 0.5, y - dy])
                angles.append(0.0)
        # Center the whole arrangement
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        for c in centers:
            c[0] -= cx
            c[1] -= cy
        return np.array(centers).flatten().tolist() + [0.0, 0.0, 0.0]

    x0 = initial_guess()

    # --- Optimize with Nelder-Mead ---
    best_params = x0
    best_score = objective(x0)

    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4,
                                   'adaptive': True})
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass

    # --- Second phase: refine with tighter settings ---
    if time.time() < deadline - 20.0:
        try:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5,
                                        'adaptive': True})
            if result2.fun < best_score:
                best_params = result2.x
                best_score = result2.fun
        except Exception:
            pass

    # --- Third phase: try different initial configurations ---
    if time.time() < deadline - 30.0:
        for attempt in range(3):
            if time.time() > deadline - 10.0:
                break
            # Perturb initial guess
            np.random.seed(attempt * 100 + 42)
            x0_alt = x0 + np.random.randn(len(x0)) * 0.5
            try:
                result_alt = minimize(objective, x0_alt, method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4,
                                               'adaptive': True})
                if result_alt.fun < best_score:
                    best_params = result_alt.x
                    best_score = result_alt.fun
            except Exception:
                pass

    # --- Extract final result ---
    inner_params = best_params[:3 * N]
    outer_params = best_params[3 * N:]
    inner_positions = np.column_stack([inner_params[0::3], inner_params[1::3]])
    inner_angles = inner_params[2::3]
    outer_cx = outer_params[0]
    outer_cy = outer_params[1]
    outer_rot = outer_params[2]

    # Compute the actual outer side
    final_side = compute_outer_side(inner_positions, inner_angles, (outer_cx, outer_cy), outer_rot)

    # Build inner list: [x, y, angle_degrees]
    inner = []
    for i in range(N):
        inner.append([float(inner_positions[i, 0]), float(inner_positions[i, 1]),
                      float(np.degrees(inner_angles[i]))])

    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(np.degrees(outer_rot))
# EVOLVE-BLOCK-END
