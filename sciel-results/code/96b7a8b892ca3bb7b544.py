"""Caging formulation: minimize outer hexagon side via support-function containment,
with smooth separation barrier. DE global search + SLSQP/NM local refinement."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)
INV_SQRT3 = 2.0 / SQRT3  # apothem -> side factor

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Minimum outer hexagon side to contain all inner hexagons (support function)."""
        max_required = 0.0
        theta = outer_angle_rad
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for k in range(6):
                    edge_angle = theta + np.radians(k * 60)
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * INV_SQRT3
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def smooth_separation(hex1_verts, hex2_verts):
        """Smooth measure of overlap using edge-normal projections.
        Returns 0 if well-separated, negative if overlapping."""
        min_gap = np.inf
        for poly in [hex1_verts, hex2_verts]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(hex1_verts, normal)
                proj2 = np.dot(hex2_verts, normal)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                gap = min(max1 - max2, max2 - min1)
                if gap < min_gap:
                    min_gap = gap
        return min_gap  # negative = overlapping

    def make_hexes(params):
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
        return inner_hexes, outer_cx, outer_cy, outer_a

    def objective(params):
        if time.time() > deadline:
            return 1e6
        inner_hexes, outer_cx, outer_cy, outer_a = make_hexes(params)
        side = outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_a)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = smooth_separation(inner_hexes[i], inner_hexes[j])
                if gap < 0:
                    penalty += 80.0 * gap * gap
        return side + penalty

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
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    return True
        return False

    def validate(params):
        inner_hexes, _, _, _ = make_hexes(params)
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                    return False
        return True

    # --- Initial configurations ---
    np.random.seed(42)
    configs = []

    # Config 1: Tight honeycomb (3x4 staggered)
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * SQRT3 * 1.02
            y = r * 1.52
            if r % 2 == 1:
                x += SQRT3 * 1.02 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Honeycomb with 30° rotation
    hx2, hy2, ha2 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * SQRT3 * 1.02
            y = r * 1.52
            if r % 2 == 1:
                x += SQRT3 * 1.02 / 2
            hx2.append(x); hy2.append(y); ha2.append(np.pi / 6)
    hx2 = np.array(hx2); hy2 = np.array(hy2); ha2 = np.array(ha2)
    hx2 -= np.mean(hx2); hy2 -= np.mean(hy2)
    configs.append(np.concatenate([hx2, hy2, ha2, [0.0, 0.0, np.pi / 6]]))

    # Config 3: Hexagonal ring (6 outer + 6 inner)
    cx3, cy3, ca3 = [], [], []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 1.8
        cx3.append(r * np.cos(angle))
        cy3.append(r * np.sin(angle))
        ca3.append(angle)
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        r = 3.2
        cx3.append(r * np.cos(angle))
        cy3.append(r * np.sin(angle))
        ca3.append(angle)
    cx3 = np.array(cx3); cy3 = np.array(cy3); ca3 = np.array(ca3)
    cx3 -= np.mean(cx3); cy3 -= np.mean(cy3)
    configs.append(np.concatenate([cx3, cy3, ca3, [0.0, 0.0, 0.0]]))

    # Config 4: 4x3 grid
    gx, gy, ga = [], [], []
    for r in range(3):
        for c in range(4):
            gx.append(c * SQRT3 * 1.02 - 1.5 * SQRT3 * 1.02)
            gy.append(r * 1.52 - 1.0 * 1.52)
            ga.append(0.0)
    gx = np.array(gx); gy = np.array(gy); ga = np.array(ga)
    gx -= np.mean(gx); gy -= np.mean(gy)
    configs.append(np.concatenate([gx, gy, ga, [0.0, 0.0, 0.0]]))

    # Config 5: Random perturbation of honeycomb
    rx5, ry5, ra5 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * SQRT3 * 1.02 + np.random.uniform(-0.15, 0.15)
            y = r * 1.52 + np.random.uniform(-0.15, 0.15)
            if r % 2 == 1:
                x += SQRT3 * 1.02 / 2
            rx5.append(x); ry5.append(y); ra5.append(np.random.uniform(-0.08, 0.08))
    rx5 = np.array(rx5); ry5 = np.array(ry5); ra5 = np.array(ra5)
    rx5 -= np.mean(rx5); ry5 -= np.mean(ry5)
    configs.append(np.concatenate([rx5, ry5, ra5, [0.0, 0.0, 0.0]]))

    # Config 6: Two rows of 6, tightly packed
    rx6, ry6, ra6 = [], [], []
    for i in range(6):
        rx6.append(i * SQRT3 * 1.02 - 2.5 * SQRT3 * 1.02)
        ry6.append(0.0)
        ra6.append(0.0)
        rx6.append(i * SQRT3 * 1.02 - 2.5 * SQRT3 * 1.02)
        ry6.append(1.52)
        ra6.append(0.0)
    rx6 = np.array(rx6); ry6 = np.array(ry6); ra6 = np.array(ra6)
    rx6 -= np.mean(rx6); ry6 -= np.mean(ry6)
    configs.append(np.concatenate([rx6, ry6, ra6, [0.0, 0.0, 0.0]]))

    # Evaluate all configs
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # --- Phase 1: Differential Evolution for global search ---
    bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(-np.pi, np.pi)] * 12 + [(-2, 2), (-2, 2), (-np.pi / 6, np.pi / 6)]

    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                objective, bounds,
                seed=42, maxiter=30, popsize=12,
                tol=1e-6, mutation=(0.3, 1.0), recombination=0.8,
                polish=True, init='sobol',
                updating='immediate'
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # --- Phase 2: L-BFGS-B from best config ---
    if time.time() < deadline - 25:
        try:
            result = minimize(
                objective, best_params, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # --- Phase 3: Nelder-Mead polish ---
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # --- Phase 4: Multi-restart local refinement ---
    if time.time() < deadline - 15:
        for trial in range(8):
            if time.time() > deadline - 10:
                break
            perturbed = best_params + np.random.normal(0, 0.03, len(best_params))
            if trial % 2 == 1:
                base = configs[trial % len(configs)]
                perturbed = base + np.random.normal(0, 0.1, len(base))

            try:
                result = minimize(
                    objective, perturbed, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x.copy()

                if time.time() < deadline - 10:
                    result2 = minimize(
                        objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                    )
                    if result2.fun < best_obj:
                        best_obj = result2.fun
                        best_params = result2.x.copy()
            except Exception:
                pass

    # --- Phase 5: Fine-grained outer angle sweep ---
    if time.time() < deadline - 8:
        for outer_a_deg in np.linspace(-30, 30, 13):
            test_params = best_params.copy()
            test_params[38] = np.radians(outer_a_deg)
            obj = objective(test_params)
            if obj < best_obj:
                best_obj = obj
                best_params = test_params.copy()

    # --- Final extraction and validation ---
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
            fallback[i] *= 1.3
            fallback[i + 12] *= 1.3
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
            inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 35.0, 0.0

    exact_side = outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_a)

    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
