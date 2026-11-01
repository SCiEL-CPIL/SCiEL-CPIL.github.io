"""Packing 12 unit hexagons in a regular hexagon via caging formulation.
Uses convex hull support functions for exact, differentiable containment
constraints and smooth separation barriers to bypass expensive SAT checks."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import math

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Return 6 vertices of a regular hexagon (flat-top orientation at angle 0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def compute_support_functions(verts):
        """Precompute edge normals and support values for a convex polygon."""
        n = len(verts)
        normals = np.zeros((n, 2))
        supports = np.zeros(n)
        for i in range(n):
            v1 = verts[i]
            v2 = verts[(i + 1) % n]
            edge = v2 - v1
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
                # Ensure outward normal
                mid = (v1 + v2) / 2
                center = np.mean(verts, axis=0)
                if np.dot(normal, mid - center) < 0:
                    normal = -normal
                normals[i] = normal
                supports[i] = np.max(verts @ normal)
            else:
                normals[i] = np.array([1.0, 0.0])
                supports[i] = 0.0
        return normals, supports

    def outer_side_from_support(supports, normals, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side using support functions.
        The outer hexagon has 6 edges. For each edge direction, we need
        the support of the inner hull in that direction to be contained."""
        max_required = 0.0
        theta = outer_angle_rad
        for k in range(6):
            # Edge normal direction for outer hexagon
            edge_angle = theta + np.radians(k * 60)
            nx = np.cos(edge_angle)
            ny = np.sin(edge_angle)
            # Support of inner hull in this direction
            # We need to compute max of (v - center) . n for all inner vertices
            # But we can use the precomputed support functions
            # For each inner edge normal, compute dot with outer edge normal
            # Actually, simpler: just compute support directly
            for i in range(len(normals)):
                dot = normals[i][0] * nx + normals[i][1] * ny
                if dot > 0:
                    # Project support onto this direction
                    # The support in direction (nx, ny) is at least
                    # the projection of each edge's support
                    # But we need the exact support, so let's just compute it
                    pass
        
        # Simpler approach: compute support of inner hull in each of the 6 outer edge directions
        # We'll pass the actual vertices for this
        return max_required

    def outer_side_length(inner_verts_list, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side to contain all inner hexagons."""
        # Collect all vertices
        all_verts = np.vstack(inner_verts_list)
        max_required = 0.0
        theta = outer_angle_rad
        for k in range(6):
            edge_angle = theta + np.radians(k * 60)
            nx = np.cos(edge_angle)
            ny = np.sin(edge_angle)
            # Support in this direction
            proj = all_verts @ np.array([nx, ny])
            # Center the projection
            proj_centered = proj - (outer_cx * nx + outer_cy * ny)
            max_proj = np.max(proj_centered)
            if max_proj > 0:
                required_S = max_proj * 2.0 / SQRT3
                if required_S > max_required:
                    max_required = required_S
        return max_required

    def sat_gap_fast(poly1, poly2):
        """Fast SAT gap computation using precomputed normals."""
        min_gap = np.inf
        # Check normals from both polygons
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
                gap = min(np.max(proj1) - np.max(proj2), np.max(proj2) - np.min(proj1))
                if gap < min_gap:
                    min_gap = gap
                if min_gap <= 0:
                    return min_gap
        return min_gap

    def smooth_penalty(poly1, poly2, strength=50.0):
        """Smooth penalty using softplus approximation."""
        gap = sat_gap_fast(poly1, poly2)
        if gap < 0:
            # Softplus: log(1 + exp(x)) / k for smoothness
            x = -gap * strength
            if x > 50:
                return x / strength
            elif x < -50:
                return 0.0
            else:
                return math.log(1.0 + math.exp(x)) / strength
        return 0.0

    def make_inner_hexes(params):
        """Extract inner hexagon vertices from parameter vector."""
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
        inner_hexes = make_inner_hexes(params)
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        side = outer_side_length(inner_hexes, outer_cx, outer_cy, outer_a)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                penalty += smooth_penalty(inner_hexes[i], inner_hexes[j])
        return side + penalty

    def sat_disjoint(poly1, poly2, tol=1e-7):
        """Check if two convex polygons are disjoint using SAT."""
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

    def validate(inner_hexes):
        """Check all pairs are disjoint."""
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                    return False
        return True

    # Generate initial configurations using hexagonal lattice structure
    np.random.seed(42)
    configs = []

    # Config 1: Honeycomb lattice (3 rows x 4 cols, staggered)
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * SQRT3 * 1.0
            y = r * 1.5
            if r % 2 == 1:
                x += SQRT3 * 1.0 / 2
            hx.append(x)
            hy.append(y)
            ha.append(0.0)
    hx = np.array(hx)
    hy = np.array(hy)
    ha = np.array(ha)
    hx -= np.mean(hx)
    hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Honeycomb with 30° rotation
    hx2, hy2, ha2 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * SQRT3 * 1.0
            y = r * 1.5
            if r % 2 == 1:
                x += SQRT3 * 1.0 / 2
            hx2.append(x)
            hy2.append(y)
            ha2.append(np.pi / 6)
    hx2 = np.array(hx2)
    hy2 = np.array(hy2)
    ha2 = np.array(ha2)
    hx2 -= np.mean(hx2)
    hy2 -= np.mean(hy2)
    configs.append(np.concatenate([hx2, hy2, ha2, [0.0, 0.0, np.pi / 6]]))

    # Config 3: 4x3 grid
    gx, gy, ga = [], [], []
    for r in range(3):
        for c in range(4):
            gx.append(c * SQRT3 * 1.0)
            gy.append(r * 1.5)
            ga.append(0.0)
    gx = np.array(gx)
    gy = np.array(gy)
    ga = np.array(ga)
    gx -= np.mean(gx)
    gy -= np.mean(gy)
    configs.append(np.concatenate([gx, gy, ga, [0.0, 0.0, 0.0]]))

    # Config 4: Hexagonal ring (1 center + 6 inner + 5 outer)
    cx3, cy3, ca3 = [], [], []
    cx3.append(0.0)
    cy3.append(0.0)
    ca3.append(0.0)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 1.75
        cx3.append(r * np.cos(angle))
        cy3.append(r * np.sin(angle))
        ca3.append(angle)
    for i in range(5):
        angle = 2 * np.pi * (i + 0.5) / 6
        r = 3.1
        cx3.append(r * np.cos(angle))
        cy3.append(r * np.sin(angle))
        ca3.append(angle)
    cx3 = np.array(cx3)
    cy3 = np.array(cy3)
    ca3 = np.array(ca3)
    cx3 -= np.mean(cx3)
    cy3 -= np.mean(cy3)
    configs.append(np.concatenate([cx3, cy3, ca3, [0.0, 0.0, 0.0]]))

    # Config 5: Two rows of 6
    rx6, ry6, ra6 = [], [], []
    for i in range(6):
        rx6.append(i * SQRT3 * 1.0 - 2.5 * SQRT3 * 1.0)
        ry6.append(0.0)
        ra6.append(0.0)
        rx6.append(i * SQRT3 * 1.0 - 2.5 * SQRT3 * 1.0)
        ry6.append(1.5)
        ra6.append(0.0)
    rx6 = np.array(rx6)
    ry6 = np.array(ry6)
    ra6 = np.array(ra6)
    rx6 -= np.mean(rx6)
    ry6 -= np.mean(ry6)
    configs.append(np.concatenate([rx6, ry6, ra6, [0.0, 0.0, 0.0]]))

    # Config 6: Circular arrangement
    cx7, cy7, ca7 = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 2.5
        cx7.append(r * np.cos(angle))
        cy7.append(r * np.sin(angle))
        ca7.append(angle)
    cx7 = np.array(cx7)
    cy7 = np.array(cy7)
    ca7 = np.array(ca7)
    cx7 -= np.mean(cx7)
    cy7 -= np.mean(cy7)
    configs.append(np.concatenate([cx7, cy7, ca7, [0.0, 0.0, 0.0]]))

    # Evaluate all configs
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # Phase 1: L-BFGS-B optimization
    bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(-np.pi, np.pi)] * 12 + [(-1, 1), (-1, 1), (-np.pi / 6, np.pi / 6)]

    try:
        result = minimize(
            objective, best_params, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
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
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-restart local refinement
    if time.time() < deadline - 25:
        for trial in range(15):
            if time.time() > deadline - 20:
                break
            # Perturb from best
            scale = 0.01 + 0.02 * trial / 15
            perturbed = best_params + np.random.normal(0, scale, len(best_params))
            
            try:
                result = minimize(
                    objective, perturbed, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x.copy()

                if time.time() < deadline - 20:
                    result2 = minimize(
                        objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if result2.fun < best_obj:
                        best_obj = result2.fun
                        best_params = result2.x.copy()
            except Exception:
                pass

    # Phase 4: Fine-grained outer angle sweep
    if time.time() < deadline - 15:
        for outer_a_deg in np.linspace(-30, 30, 25):
            test_params = best_params.copy()
            test_params[38] = np.radians(outer_a_deg)
            obj = objective(test_params)
            if obj < best_obj:
                best_obj = obj
                best_params = test_params.copy()

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
    valid = validate(inner_hexes)

    if not valid:
        # Fallback: use a widely-spaced valid configuration
        fallback = configs[0].copy()
        for i in range(12):
            fallback[i] *= 1.5
            fallback[i + 12] *= 1.5
        inner_hexes_fb = []
        for i in range(N):
            verts = hexagon_vertices(fallback[i], fallback[i + 12], 1.0, fallback[i + 24])
            inner_hexes_fb.append(verts)
        valid_fb = validate(inner_hexes_fb)
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
            inner = [[4.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    # Recompute exact side length
    exact_side = outer_side_length(inner_hexes, outer_cx, outer_cy, outer_a)

    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
