"""Rewritten program: independent per-hexagon optimization with smooth penalty."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg, scale=1.0):
        """Get vertices of a regular hexagon with side length=scale, centered at (cx,cy), rotated by angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = np.array([[cx + scale * math.cos(angle_rad + i * math.pi / 3),
                           cy + scale * math.sin(angle_rad + i * math.pi / 3)]
                          for i in range(6)])
        return verts

    def hexagon_overlap_depth(v1, v2):
        """
        Compute the minimum penetration depth between two convex hexagons using SAT.
        Returns 0 if disjoint, negative value if overlapping (magnitude = penetration depth).
        This is a smooth-ish function suitable for penalty-based optimization.
        """
        # All 12 edge normals (6 from each hexagon)
        axes = []
        for i in range(6):
            edge = v1[(i + 1) % 6] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(normal)
            if n > 1e-12:
                axes.append(normal / n)
        for i in range(6):
            edge = v2[(i + 1) % 6] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(normal)
            if n > 1e-12:
                axes.append(normal / n)

        min_sep = float('inf')
        for normal in axes:
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()

            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                # Overlapping on this axis
                overlap1 = max1 - min2
                overlap2 = max2 - min1
                sep = -min(overlap1, overlap2)

            min_sep = min(min_sep, sep)

        if min_sep == float('inf'):
            return 0.0
        return min_sep

    def compute_outer_side(all_verts, outer_angle_deg):
        """
        Compute minimum side length of outer hexagon (centered at origin, rotated by outer_angle_deg)
        to contain all points in all_verts.
        """
        if len(all_verts) == 0:
            return 0.0
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        return 2.0 * max_proj / math.sqrt(3.0)

    def build_vertices(params):
        """
        params: array of length 3*N + 1 = [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, outer_angle]
        Returns list of vertex arrays and outer_angle.
        """
        verts = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            verts.append(hex_vertices(x, y, a))
        outer_angle = params[3 * N]
        return verts, outer_angle

    def overlap_penalty(verts, weight=100.0):
        """Sum of squared overlaps for all pairs, weighted."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = hexagon_overlap_depth(verts[i], verts[j])
                if d < 0:
                    penalty += weight * d * d
        return penalty

    def objective(params):
        if time.time() > deadline - 2:
            return 1e10
        verts, oa = build_vertices(params)
        all_verts = np.vstack(verts)
        side = compute_outer_side(all_verts, oa)
        pen = overlap_penalty(verts)
        return side + pen

    # Initial layout: honeycomb cluster
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2_all = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist == 2:
                ring2_all.append((q, r))
    ring2_all.sort(key=lambda p: math.atan2(axial_to_cartesian(p[0], p[1])[1],
                                            axial_to_cartesian(p[0], p[1])[0]))
    ring2_selected = [ring2_all[i] for i in range(0, 12, 2)][:5]
    while len(ring2_selected) < 5:
        ring2_selected.append(ring2_all[len(ring2_selected)])

    cluster_axial = [(0, 0)] + ring1 + ring2_selected
    cluster_cart = []
    for q, r in cluster_axial:
        x, y = axial_to_cartesian(q, r)
        cluster_cart.append((x, y))
    cx = sum(p[0] for p in cluster_cart) / len(cluster_cart)
    cy = sum(p[1] for p in cluster_cart) / len(cluster_cart)
    cluster_cart = [(x - cx, y - cy) for x, y in cluster_cart]

    # Build initial params: [x1,y1,0, x2,y2,0, ..., xN,yN,0, outer_angle=0]
    x0 = []
    for x, y in cluster_cart:
        x0.extend([x, y, 0.0])
    x0.append(0.0)  # outer angle
    x0 = np.array(x0)

    # Bounds: positions in [-5,5], angles in [0,360], outer angle in [0,360]
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)] * N + [(0.0, 360.0)]

    best_score = float('inf')
    best_params = x0.copy()

    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(objective, bounds, maxiter=50, popsize=15,
                                               tol=1e-4, seed=42, polish=False,
                                               init='latinhypercube')
            if de_result.fun < best_score:
                # Verify validity
                verts, oa = build_vertices(de_result.x)
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if hexagon_overlap_depth(verts[i], verts[j]) < -1e-4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_score = compute_outer_side(np.vstack(verts), oa)
                    best_params = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: Local refinement with L-BFGS-B
    if time.time() < deadline - 15:
        try:
            local_result = minimize(objective, best_params, method='L-BFGS-B',
                                    bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
            if local_result.fun < best_score:
                verts, oa = build_vertices(local_result.x)
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if hexagon_overlap_depth(verts[i], verts[j]) < -1e-4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_score = compute_outer_side(np.vstack(verts), oa)
                    best_params = local_result.x.copy()
        except Exception:
            pass

    # Phase 3: COBYLA for further refinement (derivative-free)
    if time.time() < deadline - 10:
        try:
            coby_result = minimize(objective, best_params, method='COBYLA',
                                   bounds=bounds, options={'maxiter': 300, 'rhobeg': 0.1})
            if coby_result.fun < best_score:
                verts, oa = build_vertices(coby_result.x)
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if hexagon_overlap_depth(verts[i], verts[j]) < -1e-4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_score = compute_outer_side(np.vstack(verts), oa)
                    best_params = coby_result.x.copy()
        except Exception:
            pass

    # Phase 4: Try a few random restarts with local optimization
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            # Perturb the best solution
            perturbation = rng.normal(0, 0.5, size=len(best_params))
            perturbation[-1] = rng.uniform(0, 360)  # outer angle
            trial_params = best_params + perturbation
            # Clip to bounds
            for i in range(len(trial_params)):
                lo, hi = bounds[i]
                trial_params[i] = np.clip(trial_params[i], lo, hi)

            try:
                local_result = minimize(objective, trial_params, method='L-BFGS-B',
                                        bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
                if local_result.fun < best_score:
                    verts, oa = build_vertices(local_result.x)
                    valid = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            if hexagon_overlap_depth(verts[i], verts[j]) < -1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_score = compute_outer_side(np.vstack(verts), oa)
                        best_params = local_result.x.copy()
            except Exception:
                pass

    # Finalize
    verts, outer_angle = build_vertices(best_params)
    all_verts = np.vstack(verts)

    # Center the configuration
    centers = np.array([[best_params[3 * i], best_params[3 * i + 1]] for i in range(N)])
    cx_final = np.mean(centers[:, 0])
    cy_final = np.mean(centers[:, 1])
    for i in range(N):
        best_params[3 * i] -= cx_final
        best_params[3 * i + 1] -= cy_final

    # Recompute final side after centering
    verts, outer_angle = build_vertices(best_params)
    all_verts = np.vstack(verts)
    final_side = compute_outer_side(all_verts, outer_angle)

    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if hexagon_overlap_depth(verts[i], verts[j]) < -1e-4:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: use honeycomb layout
        positions = np.array(cluster_cart)
        angles = [0.0] * N
        outer_angle = 0.0
        verts = [hex_vertices(positions[i][0], positions[i][1], 0.0) for i in range(N)]
        all_verts = np.vstack(verts)
        final_side = compute_outer_side(all_verts, outer_angle)
        inner = [[positions[i][0], positions[i][1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), final_side, outer_angle

    inner = [[best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, outer_angle

# EVOLVE-BLOCK-END
