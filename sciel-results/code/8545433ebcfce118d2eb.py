"""Caging formulation with smooth separation barrier and vectorized support-function containment.
Structural change: use smooth penalty (not hard reject) for disjointness, vectorized objective,
and a two-phase strategy (coarse global search + fine local polish)."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def all_inner_vertices(params):
        """Compute all 72 vertices (12 hexagons x 6) from params."""
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        verts_list = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            verts_list.append(verts)
        return np.vstack(verts_list)  # shape (72, 2)

    def outer_side_vectorized(all_verts, outer_cx, outer_cy, outer_angle_rad):
        """Vectorized minimum outer hexagon side via support function."""
        offsets = all_verts - np.array([outer_cx, outer_cy])
        max_required = 0.0
        theta = outer_angle_rad
        for k in range(6):
            edge_angle = theta + np.radians(k * 60)
            nx = np.cos(edge_angle)
            ny = np.sin(edge_angle)
            dists = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_dist = np.max(dists)
            if max_dist > 0:
                required_S = max_dist * 2.0 / np.sqrt(3.0)
                if required_S > max_required:
                    max_required = required_S
        return max_required

    def pair_gap(h1_verts, h2_verts):
        """Minimum gap between two hexagons using edge normals. Negative = overlapping."""
        min_gap = np.inf
        for poly in [h1_verts, h2_verts]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(h1_verts, normal)
                proj2 = np.dot(h2_verts, normal)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                gap = min(max1 - max2, max2 - min1)
                if gap < min_gap:
                    min_gap = gap
        return min_gap

    def smooth_overlap_penalty(all_hexes, weight=100.0):
        """Sum of squared negative gaps (overlap) for all pairs."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = pair_gap(all_hexes[i], all_hexes[j])
                if gap < 0:
                    penalty += weight * gap * gap
        return penalty

    def make_hexes(params):
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        return inner_hexes

    def objective(params):
        if time.time() > deadline:
            return 1e6
        inner_hexes = make_hexes(params)
        all_verts = np.vstack(inner_hexes)
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        side = outer_side_vectorized(all_verts, outer_cx, outer_cy, outer_a)
        penalty = smooth_overlap_penalty(inner_hexes, weight=50.0)
        return side + penalty

    def sat_disjoint(poly1, poly2, tol=1e-9):
        """Check if two polygons are disjoint using SAT."""
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
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    return True
        return False

    def is_valid(params):
        """Check all pairs are disjoint."""
        inner_hexes = make_hexes(params)
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                    return False
        return True

    # Generate diverse initial configurations
    np.random.seed(42)
    configs = []
    sqrt3 = np.sqrt(3)

    # Config 1: Honeycomb 3x4 staggered
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.0
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 * 1.0 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Honeycomb with 30° rotation
    hx2, hy2, ha2 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.0
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 * 1.0 / 2
            hx2.append(x); hy2.append(y); ha2.append(np.pi / 6)
    hx2 = np.array(hx2); hy2 = np.array(hy2); ha2 = np.array(ha2)
    hx2 -= np.mean(hx2); hy2 -= np.mean(hy2)
    configs.append(np.concatenate([hx2, hy2, ha2, [0.0, 0.0, np.pi / 6]]))

    # Config 3: Two rings (6 inner + 6 outer)
    cx3, cy3, ca3 = [], [], []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 1.75
        cx3.append(r * np.cos(angle))
        cy3.append(r * np.sin(angle))
        ca3.append(0.0)
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        r = 3.1
        cx3.append(r * np.cos(angle))
        cy3.append(r * np.sin(angle))
        ca3.append(0.0)
    cx3 = np.array(cx3); cy3 = np.array(cy3); ca3 = np.array(ca3)
    cx3 -= np.mean(cx3); cy3 -= np.mean(cy3)
    configs.append(np.concatenate([cx3, cy3, ca3, [0.0, 0.0, 0.0]]))

    # Config 4: Linear arrangement (two rows of 6)
    gx, gy, ga = [], [], []
    for i in range(6):
        gx.append(i * sqrt3 - 2.5 * sqrt3)
        gy.append(0.0)
        ga.append(0.0)
        gx.append(i * sqrt3 - 2.5 * sqrt3)
        gy.append(1.5)
        ga.append(0.0)
    gx = np.array(gx); gy = np.array(gy); ga = np.array(ga)
    gx -= np.mean(gx); gy -= np.mean(gy)
    configs.append(np.concatenate([gx, gy, ga, [0.0, 0.0, 0.0]]))

    # Config 5: Random perturbation of honeycomb
    rx5, ry5, ra5 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.0 + np.random.uniform(-0.1, 0.1)
            y = r * 1.5 + np.random.uniform(-0.1, 0.1)
            if r % 2 == 1:
                x += sqrt3 * 1.0 / 2
            rx5.append(x); ry5.append(y); ra5.append(np.random.uniform(-0.05, 0.05))
    rx5 = np.array(rx5); ry5 = np.array(ry5); ra5 = np.array(ra5)
    rx5 -= np.mean(rx5); ry5 -= np.mean(ry5)
    configs.append(np.concatenate([rx5, ry5, ra5, [0.0, 0.0, 0.0]]))

    # Config 6: Compact cluster
    cx6, cy6, ca6 = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 1.8 + 0.3 * np.sin(3 * angle)
        cx6.append(r * np.cos(angle))
        cy6.append(r * np.sin(angle))
        ca6.append(angle)
    cx6 = np.array(cx6); cy6 = np.array(cy6); ca6 = np.array(ca6)
    cx6 -= np.mean(cx6); cy6 -= np.mean(cy6)
    configs.append(np.concatenate([cx6, cy6, ca6, [0.0, 0.0, 0.0]]))

    # Evaluate all configs
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # Phase 1: L-BFGS-B from best config
    bounds = [(-8, 8)] * 12 + [(-8, 8)] * 12 + [(-np.pi, np.pi)] * 12 + [(-2, 2), (-2, 2), (-np.pi / 3, np.pi / 3)]

    try:
        result = minimize(
            objective, best_params, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-restart local refinement
    if time.time() < deadline - 25:
        for trial in range(12):
            if time.time() > deadline - 18:
                break
            # Alternate between perturbing best and perturbing other configs
            if trial % 2 == 0:
                perturbed = best_params + np.random.normal(0, 0.02, len(best_params))
            else:
                base = configs[trial % len(configs)]
                perturbed = base + np.random.normal(0, 0.08, len(base))

            try:
                result = minimize(
                    objective, perturbed, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x.copy()

                # Quick NM polish
                if time.time() < deadline - 18:
                    result2 = minimize(
                        objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if result2.fun < best_obj:
                        best_obj = result2.fun
                        best_params = result2.x.copy()
            except Exception:
                pass

    # Phase 4: Fine-grained outer angle sweep
    if time.time() < deadline - 12:
        for outer_a_deg in np.linspace(-30, 30, 25):
            test_params = best_params.copy()
            test_params[38] = np.radians(outer_a_deg)
            obj = objective(test_params)
            if obj < best_obj:
                best_obj = obj
                best_params = test_params.copy()

    # Phase 5: Final fine-tune of outer position and angle
    if time.time() < deadline - 8:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Final extraction and validation
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    outer_cx = params[36]
    outer_cy = params[37]
    outer_a = params[38]

    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)

    # Validate
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: use a widely-spaced valid configuration
        fallback = configs[0].copy()
        for i in range(12):
            fallback[i] *= 1.5
            fallback[i + 12] *= 1.5
        inner_hexes_fb = []
        for i in range(N):
            verts = hexagon_vertices(fallback[i], fallback[i+12], 1.0, fallback[i+24])
            inner_hexes_fb.append(verts)
        valid_fb = True
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(inner_hexes_fb[i], inner_hexes_fb[j]):
                    valid_fb = False
                    break
            if not valid_fb:
                break
        if valid_fb:
            params = fallback
            inner_hexes = inner_hexes_fb
            outer_cx = params[36]
            outer_cy = params[37]
            outer_a = params[38]
            inner_x = params[0:12]
            inner_y = params[12:24]
            inner_a = params[24:36]
        else:
            # Last resort: very widely spaced
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    # Recompute exact side length
    all_verts = np.vstack(inner_hexes)
    exact_side = outer_side_vectorized(all_verts, outer_cx, outer_cy, outer_a)

    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
