"""Structural change: treat the 12-hexagon cluster as a rigid body (honeycomb lattice),
search over (relative orientation, global rotation) to minimize the bounding regular
hexagon side length via a fast support-function scan, then refine with Nelder-Mead."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)
INV_SQRT3 = 2.0 / SQRT3

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # --- Build the tight honeycomb lattice of 12 unit hexagons ---
    # 3 rows of 4, staggered. Unit hexagon side=1, width=2*sqrt(3), height=2.
    # Row spacing = 1.5 (apothem*2 for adjacent rows), col spacing = sqrt(3)
    # Stagger: odd rows shifted by sqrt(3)/2
    def make_honeycomb(scale=1.0):
        pts = []
        for r in range(3):
            for c in range(4):
                x = c * SQRT3 * scale
                y = r * 1.5 * scale
                if r % 2 == 1:
                    x += SQRT3 * scale / 2.0
                pts.append((x, y))
        pts = np.array(pts)
        # Center the cluster
        pts -= pts.mean(axis=0)
        return pts

    base_pts = make_honeycomb(1.0)

    def rotate_pts(pts, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def outer_side_from_pts(pts, outer_angle_rad):
        """Minimum side of regular hexagon (centered at origin, rotated by outer_angle_rad)
        to contain all points in pts (using support function)."""
        max_required = 0.0
        theta = outer_angle_rad
        for k in range(6):
            edge_angle = theta + np.radians(k * 60)
            nx = np.cos(edge_angle)
            ny = np.sin(edge_angle)
            proj = pts[:, 0] * nx + pts[:, 1] * ny
            # The hexagon with side S has support S*sqrt(3)/2 = S * apothem_factor
            # apothem = S * sqrt(3)/2, so S = proj_max / (sqrt(3)/2) = proj_max * 2/sqrt(3)
            d = np.max(proj)
            if d > 0:
                required_S = d * INV_SQRT3
                if required_S > max_required:
                    max_required = required_S
        return max_required

    def full_hexagon_side(rel_angle, outer_angle):
        """For a given relative rotation of the honeycomb cluster and outer hexagon rotation,
        compute the minimum outer side. We also need to check all 6 inner hexagons' vertices."""
        # Rotate the base points
        rot_pts = rotate_pts(base_pts, rel_angle)
        
        # Build all 72 vertices (12 hexagons * 6 vertices)
        all_verts = []
        for i in range(N):
            cx, cy = rot_pts[i]
            verts = hexagon_vertices(cx, cy, 1.0, rel_angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        return outer_side_from_pts(all_verts, outer_angle)

    def objective(params):
        if time.time() > deadline:
            return 1e6
        rel_angle = params[0]
        outer_angle = params[1]
        return full_hexagon_side(rel_angle, outer_angle)

    # --- Phase 1: Coarse grid search over relative and outer angles ---
    best_side = np.inf
    best_rel = 0.0
    best_outer = 0.0

    # The honeycomb has 6-fold symmetry in its local frame, so rel_angle in [0, pi/3)
    # outer_angle in [0, pi/3)
    n_coarse = 36
    rel_angles = np.linspace(0, np.pi / 3, n_coarse, endpoint=False)
    outer_angles = np.linspace(0, np.pi / 3, n_coarse, endpoint=False)

    for ra in rel_angles:
        for oa in outer_angles:
            s = full_hexagon_side(ra, oa)
            if s < best_side:
                best_side = s
                best_rel = ra
                best_outer = oa

    # --- Phase 2: Nelder-Mead refinement from best grid point ---
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                np.array([best_rel, best_outer]),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_side:
                best_side = result.fun
                best_rel = result.x[0]
                best_outer = result.x[1]
        except Exception:
            pass

    # --- Phase 3: Multi-restart Nelder-Mead from nearby grid points ---
    if time.time() < deadline - 15:
        # Take top 5 grid points and refine
        grid_scores = []
        for ra in rel_angles:
            for oa in outer_angles:
                s = full_hexagon_side(ra, oa)
                grid_scores.append((s, ra, oa))
        grid_scores.sort()
        for s, ra, oa in grid_scores[:5]:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    objective,
                    np.array([ra, oa]),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < best_side:
                    best_side = result.fun
                    best_rel = result.x[0]
                    best_outer = result.x[1]
            except Exception:
                pass

    # --- Phase 4: Fine outer-angle sweep around best ---
    if time.time() < deadline - 8:
        for da in np.linspace(-0.02, 0.02, 21):
            for db in np.linspace(-0.02, 0.02, 21):
                test_ra = best_rel + da
                test_oa = best_outer + db
                s = full_hexagon_side(test_ra, test_oa)
                if s < best_side:
                    best_side = s
                    best_rel = test_ra
                    best_outer = test_oa

    # --- Final extraction ---
    # The outer hexagon is centered at origin
    outer_cx, outer_cy = 0.0, 0.0
    outer_a = best_outer
    rel_angle = best_rel

    # Build inner hexagons
    rot_pts = rotate_pts(base_pts, rel_angle)
    inner = []
    for i in range(N):
        inner.append([rot_pts[i, 0], rot_pts[i, 1], np.degrees(rel_angle)])

    # Verify: compute exact side
    all_verts = []
    for i in range(N):
        verts = hexagon_vertices(rot_pts[i, 0], rot_pts[i, 1], 1.0, rel_angle)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    exact_side = outer_side_from_pts(all_verts, outer_a)

    # Validate: check that hexagons are disjoint (they should be by construction)
    # SAT check
    def sat_disjoint(poly1, poly2, tol=1e-8):
        for poly in [poly1, poly2]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = poly1 @ normal
                proj2 = poly2 @ normal
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    return True
        return False

    valid = True
    hex_verts_list = []
    for i in range(N):
        verts = hexagon_vertices(rot_pts[i, 0], rot_pts[i, 1], 1.0, rel_angle)
        hex_verts_list.append(verts)
    for i in range(N):
        for j in range(i + 1, N):
            if not sat_disjoint(hex_verts_list[i], hex_verts_list[j]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: scale up the honeycomb slightly
        base_pts_scaled = make_honeycomb(1.02)
        rot_pts_s = rotate_pts(base_pts_scaled, rel_angle)
        inner = []
        for i in range(N):
            inner.append([rot_pts_s[i, 0], rot_pts_s[i, 1], np.degrees(rel_angle)])
        all_verts_s = []
        for i in range(N):
            verts = hexagon_vertices(rot_pts_s[i, 0], rot_pts_s[i, 1], 1.0, rel_angle)
            all_verts_s.append(verts)
        all_verts_s = np.vstack(all_verts_s)
        exact_side = outer_side_from_pts(all_verts_s, outer_a)

    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
