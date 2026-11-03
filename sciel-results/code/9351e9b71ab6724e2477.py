import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = math.sqrt(3)


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin

    # Unit hexagon vertices at angle 0 (flat-top: vertex at 30° offset)
    # Actually use standard: vertices at 0, 60, 120, 180, 240, 300
    base_angles = np.array([math.radians(a) for a in range(0, 360, 60)])
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    def hex_verts(cx, cy, angle_deg):
        """Vertices of unit hexagon centered at (cx,cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        # Rotate the base vertices
        x = base_cos * ca - base_sin * sa
        y = base_cos * sa + base_sin * ca
        return np.column_stack([x + cx, y + cy])

    def min_distance_hex_hex(c1, a1, c2, a2):
        """
        Compute the minimum distance between two unit hexagons.
        If negative, they overlap (penetration depth).
        Uses the separating axis theorem: for each of the 12 edge normals
        (6 from each hexagon), compute the gap.
        The minimum over all axes of (max separation) gives the distance.
        If all axes show overlap, return the maximum penetration.
        """
        v1 = hex_verts(c1[0], c1[1], a1)
        v2 = hex_verts(c2[0], c2[1], a2)

        # Edge normals for each hexagon
        def edge_normals(verts):
            normals = []
            for i in range(6):
                dx = verts[(i+1)%6, 0] - verts[i, 0]
                dy = verts[(i+1)%6, 1] - verts[i, 1]
                # Perpendicular (outward normal for CCW polygon)
                n = np.array([-dy, dx])
                n /= np.linalg.norm(n)
                normals.append(n)
            return np.array(normals)

        n1 = edge_normals(v1)
        n2 = edge_normals(v2)
        all_normals = np.vstack([n1, n2])  # 12 normals

        # Project both hexagons onto each normal
        # For each normal, compute the gap
        min_gap = float('inf')
        for n in all_normals:
            p1 = v1 @ n
            p2 = v2 @ n
            # Gap: if intervals don't overlap, gap = min(p2) - max(p1) or min(p1) - max(p2)
            # If they overlap, gap is negative (penetration)
            gap1 = p2.min() - p1.max()  # v2 is to the right of v1
            gap2 = p1.min() - p2.max()  # v1 is to the right of v2
            gap = max(gap1, gap2)  # the actual separation (positive = separated)
            # But we want the minimum gap over all axes (the tightest constraint)
            # Actually for SAT, the hexagons are separated if there EXISTS an axis
            # where they don't overlap. The distance is the MINIMUM gap over all axes
            # where the gap is positive, or if no such axis exists, it's the negative
            # of the maximum penetration.
            # Let's think differently: the distance between two convex polygons is
            # the minimum over all separating axes of the gap.
            # If they overlap, all gaps are negative, and the distance is negative
            # (penetration depth = -min_gap where min_gap is the most negative).
            min_gap = min(min_gap, gap)

        # Actually this isn't quite right for distance. Let me use a different approach:
        # The minimum distance between two convex polygons can be computed as:
        # min over all axes of (max(0, gap)) if separated
        # But for our purpose, we want a smooth-ish function.
        # Let's use: the minimum gap over all 12 normals.
        # If min_gap > 0, the hexagons are separated by at least min_gap.
        # If min_gap < 0, they overlap (but this doesn't give exact penetration).
        # For optimization, we'll use a smooth approximation.

        # Better approach: compute the actual minimum vertex-to-edge distance
        # or use the fact that for two regular hexagons, the distance is well-defined.
        # Let's use a simpler but effective approach:
        # The minimum distance is min over all vertices of v1 of dist to v2,
        # and min over all vertices of v2 of dist to v1.
        # This is O(36) per pair, which is fine.

        # Point-to-convex-polygon distance (minimum distance from point to polygon)
        def point_to_convex_poly_dist(pt, poly):
            """Minimum distance from point to convex polygon (0 if inside)."""
            n = len(poly)
            # Check if point is inside
            inside = True
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i+1)%n]
                cross = (x2-x1)*(pt[1]-y1) - (y2-y1)*(pt[0]-x1)
                if cross < -1e-12:
                    inside = False
                    break
            if inside:
                return 0.0

            min_d = float('inf')
            for i in range(n):
                a = poly[i]
                b = poly[(i+1)%n]
                # Project pt onto segment ab
                ab = b - a
                t = np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-30)
                t = max(0.0, min(1.0, t))
                closest = a + t * ab
                d = np.linalg.norm(pt - closest)
                min_d = min(min_d, d)
            return min_d

        # Compute min distance
        min_dist = float('inf')
        for pt in v1:
            d = point_to_convex_poly_dist(pt, v2)
            min_dist = min(min_dist, d)
        for pt in v2:
            d = point_to_convex_poly_dist(pt, v1)
            min_dist = min(min_dist, d)

        return min_dist  # positive = separated, 0 = touching

    def smooth_nonoverlap_penalty(centers, angles, eps=0.01):
        """
        Smooth penalty for overlap. For each pair, if distance < 0 (overlap),
        apply a quadratic penalty. Use a smooth approximation.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_distance_hex_hex(centers[i], angles[i], centers[j], angles[j])
                if d < 0:
                    # Overlap: quadratic penalty
                    penalty += (d / eps) ** 2
                elif d < 2*eps:
                    # Near-touching: small penalty to encourage separation
                    penalty += 0.01 * ((2*eps - d) / eps) ** 2
        return penalty

    def compute_outer_side(centers, angles, outer_angle_deg):
        """Compute minimum outer hexagon side length for given configuration."""
        theta = math.radians(outer_angle_deg)
        normals = np.array([
            [math.cos(theta + math.pi/6 + k * math.pi/3),
             math.sin(theta + math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])

        all_verts = []
        for i in range(N):
            v = hex_verts(centers[i][0], centers[i][1], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)

        projs = all_verts @ normals.T
        min_projs = np.min(projs, axis=0)
        max_projs = np.max(projs, axis=0)
        widths = max_projs - min_projs
        side = np.max(widths) / SQRT3

        # Find center via least squares
        targets = (max_projs + min_projs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals, targets, rcond=None)

        # Verify and adjust
        shifted = all_verts - c
        shifted_projs = shifted @ normals.T
        A = side * SQRT3 / 2
        if np.max(np.abs(shifted_projs)) > A + 1e-9:
            side = np.max(np.abs(shifted_projs)) * 2 / SQRT3 + 1e-9

        return side, c

    def objective(params):
        """
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11] (36 params)
        Objective: outer side length + overlap penalty
        """
        if time.time() > deadline:
            return 1e10

        centers = params[0:3*N:3]
        angles = params[2:3*N+1:3]
        centers = np.array(centers)
        angles = np.array(angles)

        # Overlap penalty
        pen = smooth_nonoverlap_penalty(centers, angles)
        if pen > 1.0:
            return 10.0 + pen

        # Compute outer side (try a few angles, take best)
        best_side = float('inf')
        for oa in [0.0, 30.0]:
            s, _ = compute_outer_side(centers, angles, oa)
            best_side = min(best_side, s)

        return best_side + 0.1 * pen

    # --- Differential Evolution for global search ---
    # Bounds: centers in [-5, 5], angles in [0, 60] (hexagon has 60° symmetry)
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 60.0))  # angle

    # Initial population: use a good lattice arrangement
    def initial_pop():
        pop = []
        # Hexagonal lattice arrangement
        lattice = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = SQRT3 * (i + j / 2.0)
                y = 3.0 * j
                lattice.append((x, y))
        lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
        lat = np.array(lattice[:12])
        lat -= np.mean(lat, axis=0)

        # Create a few diverse initial points
        for trial in range(10):
            if trial == 0:
                # Lattice
                pts = lat.copy()
            elif trial == 1:
                # Ring + center
                pts = np.array([[0, 0]] + [
                    [2.0*math.cos(math.radians(60*k)), 2.0*math.sin(math.radians(60*k))]
                    for k in range(6)
                ] + [
                    [3.5*math.cos(math.radians(60*k+30)), 3.5*math.sin(math.radians(60*k+30))]
                    for k in range(5)
                ])
                pts -= np.mean(pts, axis=0)
            elif trial == 2:
                # 4x3 grid
                pts = []
                for r in range(3):
                    for c in range(4):
                        x = c * 2.0 - 3.0
                        y = r * 1.5 * SQRT3
                        if r % 2 == 1:
                            x += 1.0
                        pts.append([x, y])
                pts = np.array(pts)
                pts -= np.mean(pts, axis=0)
            else:
                # Random perturbation of lattice
                pts = lat + np.random.randn(12, 2) * 0.5

            params = []
            for i in range(N):
                params.append(pts[i][0])
                params.append(pts[i][1])
                params.append(0.0)  # Start with 0 angle
            pop.append(params)
        return pop

    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,  # We'll do our own polish
            init=initial_pop(),
            updating='immediate',
            callback=None,
            workers=1
        )
        best_params = de_result.x
        best_obj = de_result.fun
    except Exception:
        # Fallback to lattice
        lattice = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = SQRT3 * (i + j / 2.0)
                y = 3.0 * j
                lattice.append((x, y))
        lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
        lat = np.array(lattice[:12])
        lat -= np.mean(lat, axis=0)
        best_params = []
        for i in range(N):
            best_params.extend([lat[i][0], lat[i][1], 0.0])
        best_obj = objective(best_params)

    # --- SLSQP polish ---
    if time.time() < deadline - 10:
        try:
            slsqp_result = minimize(
                objective,
                best_params,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            if slsqp_result.fun < best_obj:
                best_obj = slsqp_result.fun
                best_params = slsqp_result.x
        except Exception:
            pass

    # --- Final refinement: try more outer angles and compute exact side ---
    centers = np.array(best_params[0:3*N:3])
    angles = np.array(best_params[2:3*N+1:3])

    # Validate no overlap
    overlap_ok = True
    for i in range(N):
        for j in range(i+1, N):
            d = min_distance_hex_hex(centers[i], angles[i], centers[j], angles[j])
            if d < -1e-6:  # Allow tiny numerical error
                overlap_ok = False
                break
        if not overlap_ok:
            break

    if not overlap_ok:
        # Fallback to lattice
        lattice = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = SQRT3 * (i + j / 2.0)
                y = 3.0 * j
                lattice.append((x, y))
        lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
        lat = np.array(lattice[:12])
        lat -= np.mean(lat, axis=0)
        centers = lat
        angles = np.zeros(N)

    # Find best outer angle and side
    best_side = float('inf')
    best_outer_angle = 0.0
    best_center = np.zeros(2)
    for oa in np.linspace(0, 60, 61):
        s, c = compute_outer_side(centers, angles, oa)
        if s < best_side:
            best_side = s
            best_outer_angle = oa
            best_center = c

    # Fine-tune outer angle
    if time.time() < deadline - 5:
        for oa in np.linspace(best_outer_angle - 5, best_outer_angle + 5, 21):
            s, c = compute_outer_side(centers, angles, oa % 60)
            if s < best_side:
                best_side = s
                best_outer_angle = oa % 60
                best_center = c

    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])

    return inner, [float(best_center[0]), float(best_center[1])], float(best_side), float(best_outer_angle)
