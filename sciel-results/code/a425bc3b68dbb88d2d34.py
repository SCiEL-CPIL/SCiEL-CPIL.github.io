"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def hexagon_edge_normals(side, angle_rad):
        """Outward normals of a regular hexagon with given side and rotation."""
        # For a flat-top hexagon, edges are at angles 0, 60, 120, 180, 240, 300 degrees
        # Edge normals point outward
        edge_angles = angle_rad + np.radians(np.arange(0, 360, 60))
        # Normal to each edge is perpendicular, pointing outward
        normals = np.column_stack([
            np.cos(edge_angles + np.pi/2),
            np.sin(edge_angles + np.pi/2)
        ])
        return normals

    def minkowski_support_distance(cx1, cy1, a1, cx2, cy2, a2):
        """Compute minimum separation distance using Minkowski difference support function.
        
        For two convex polygons P and Q, the minimum distance is related to the
        support function of the Minkowski difference P - Q.
        
        Returns: negative value if overlapping (penetration depth), positive if separated.
        """
        # Get vertices and normals for both hexagons
        verts1 = hexagon_vertices(cx1, cy1, 1.0, a1)
        verts2 = hexagon_vertices(cx2, cy2, 1.0, a2)
        
        normals1 = hexagon_edge_normals(1.0, a1)
        normals2 = hexagon_edge_normals(1.0, a2)
        
        center1 = np.array([cx1, cy1])
        center2 = np.array([cx2, cy2])
        
        min_gap = np.inf
        
        # Check all candidate directions (normals from both hexagons)
        all_normals = np.vstack([normals1, normals2])
        
        for normal in all_normals:
            # Project both hexagons onto this normal direction
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            # Determine which side each center is on
            d_center = np.dot(center1 - center2, normal)
            
            if d_center >= 0:
                # center1 is on the positive side
                gap = np.min(proj2) - np.max(proj1)
            else:
                # center1 is on the negative side
                gap = np.min(proj1) - np.max(proj2)
            
            if gap < min_gap:
                min_gap = gap
        
        return min_gap

    def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
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
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def objective(params):
        if time.time() > deadline:
            return 1e6
        
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
        
        # Smooth penalty for overlap using Minkowski support distance
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = minkowski_support_distance(
                    inner_x[i], inner_y[i], inner_a[i],
                    inner_x[j], inner_y[j], inner_a[j]
                )
                if sep < 0:
                    # Quadratic penalty for penetration
                    penalty += 50.0 * sep * sep
        
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        return side + penalty

    # Generate diverse initial configurations
    np.random.seed(42)
    configs = []
    sqrt3 = np.sqrt(3)

    # Config 1: Honeycomb lattice (3x4 staggered)
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.01
            y = r * 1.51
            if r % 2 == 1:
                x += sqrt3 * 1.01 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Honeycomb with 30-degree rotation
    hx2, hy2, ha2 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.01
            y = r * 1.51
            if r % 2 == 1:
                x += sqrt3 * 1.01 / 2
            hx2.append(x); hy2.append(y); ha2.append(np.pi / 6)
    hx2 = np.array(hx2); hy2 = np.array(hy2); ha2 = np.array(ha2)
    hx2 -= np.mean(hx2); hy2 -= np.mean(hy2)
    configs.append(np.concatenate([hx2, hy2, ha2, [0.0, 0.0, np.pi / 6]]))

    # Config 3: Compact circular arrangement
    cx, cy, ca = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 2.6
        cx.append(r * np.cos(angle))
        cy.append(r * np.sin(angle))
        ca.append(angle)
    cx = np.array(cx); cy = np.array(cy); ca = np.array(ca)
    cx -= np.mean(cx); cy -= np.mean(cy)
    configs.append(np.concatenate([cx, cy, ca, [0.0, 0.0, 0.0]]))

    # Config 4: Two rows of 6 with tight spacing
    rx, ry, ra = [], [], []
    for i in range(6):
        rx.append(i * sqrt3 * 1.01 - 2.5 * sqrt3 * 1.01)
        ry.append(0.0)
        ra.append(0.0)
        rx.append(i * sqrt3 * 1.01 - 2.5 * sqrt3 * 1.01)
        ry.append(1.51)
        ra.append(0.0)
    rx = np.array(rx); ry = np.array(ry); ra = np.array(ra)
    rx -= np.mean(rx); ry -= np.mean(ry)
    configs.append(np.concatenate([rx, ry, ra, [0.0, 0.0, 0.0]]))

    # Config 5: Random perturbation of honeycomb
    rx5, ry5, ra5 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.01 + np.random.uniform(-0.12, 0.12)
            y = r * 1.51 + np.random.uniform(-0.12, 0.12)
            if r % 2 == 1:
                x += sqrt3 * 1.01 / 2
            rx5.append(x); ry5.append(y); ra5.append(np.random.uniform(-0.06, 0.06))
    rx5 = np.array(rx5); ry5 = np.array(ry5); ra5 = np.array(ra5)
    rx5 -= np.mean(rx5); ry5 -= np.mean(ry5)
    configs.append(np.concatenate([rx5, ry5, ra5, [0.0, 0.0, 0.0]]))

    # Config 6: 4x3 grid with hexagonal offset
    hx6, hy6, ha6 = [], [], []
    for r in range(4):
        for c in range(3):
            x = c * sqrt3 * 1.01
            y = r * 1.51
            if r % 2 == 1:
                x += sqrt3 * 1.01 / 2
            hx6.append(x); hy6.append(y); ha6.append(0.0)
    hx6 = np.array(hx6); hy6 = np.array(hy6); ha6 = np.array(ha6)
    hx6 -= np.mean(hx6); hy6 -= np.mean(hy6)
    configs.append(np.concatenate([hx6, hy6, ha6, [0.0, 0.0, 0.0]]))

    # Config 7: Tighter honeycomb with slight rotation
    hx7, hy7, ha7 = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 0.99
            y = r * 1.49
            if r % 2 == 1:
                x += sqrt3 * 0.99 / 2
            hx7.append(x); hy7.append(y); ha7.append(np.pi / 12)
    hx7 = np.array(hx7); hy7 = np.array(hy7); ha7 = np.array(ha7)
    hx7 -= np.mean(hx7); hy7 -= np.mean(hy7)
    configs.append(np.concatenate([hx7, hy7, ha7, [0.0, 0.0, np.pi / 12]]))

    # Evaluate all configs and pick best
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 50:
        try:
            from scipy.optimize import differential_evolution
            bounds_de = [(-4, 4)] * 12 + [(-4, 4)] * 12 + [(-np.pi/3, np.pi/3)] * 12 + [(-0.5, 0.5), (-0.5, 0.5), (-np.pi/6, np.pi/6)]
            
            def de_objective(params):
                if time.time() > deadline - 5:
                    return 1e6
                return objective(params)
            
            result = differential_evolution(
                de_objective, bounds_de,
                seed=42, maxiter=200, popsize=15,
                tol=1e-6, polish=True
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Phase 2: L-BFGS-B refinement
    bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(-np.pi, np.pi)] * 12 + [(-1, 1), (-1, 1), (-np.pi / 6, np.pi / 6)]
    
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective, best_params, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Phase 3: Nelder-Mead refinement
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Phase 4: Multiple local refinements from perturbed best
    if time.time() < deadline - 10:
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            perturbation_scale = 0.015 + 0.01 * trial
            perturbed = best_params + np.random.normal(0, perturbation_scale, len(best_params))
            try:
                result = minimize(
                    objective, perturbed, method='Nelder-Mead',
                    options={'maxiter': 2500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass

    # Phase 5: Final fine-tuning
    if time.time() < deadline - 3:
        try:
            result = minimize(
                objective, best_params, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Extract solution
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    outer_cx = params[36]
    outer_cy = params[37]
    outer_a = params[38]
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    
    # Verify validity using Minkowski support distance
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            sep = minkowski_support_distance(
                inner_x[i], inner_y[i], inner_a[i],
                inner_x[j], inner_y[j], inner_a[j]
            )
            if sep < -1e-4:  # Allow tiny numerical tolerance
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid configuration with wider spacing
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    # Recompute exact side length without penalty
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    exact_side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
    
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
