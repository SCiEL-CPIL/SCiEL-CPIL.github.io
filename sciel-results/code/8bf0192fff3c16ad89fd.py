import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 142.0
    SQRT3 = math.sqrt(3)
    MIN_DIST = SQRT3 - 1e-6

    def vec_hex_verts(cx, cy, angle):
        """Vectorized: compute 6 vertices of a unit hexagon."""
        thetas = angle + np.arange(6) * (np.pi / 3)
        return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

    def vec_sat_disjoint(verts1, verts2, eps=1e-7):
        """Vectorized SAT: check if two hexagons (6x2 arrays) are disjoint. Returns True if disjoint."""
        # 12 edge normals (6 from each hexagon)
        edges1 = np.roll(verts1, -1, axis=0) - verts1
        edges2 = np.roll(verts2, -1, axis=0) - verts2
        norms1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        norms2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        norms = np.vstack([norms1, norms2])
        norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-15)
        # Project all 12 vertices onto each of 12 axes
        proj1 = verts1 @ norms.T  # 6x12
        proj2 = verts2 @ norms.T  # 6x12
        min1 = proj1.min(axis=0)  # 12
        max1 = proj1.max(axis=0)
        min2 = proj2.min(axis=0)
        max2 = proj2.max(axis=0)
        # Disjoint if any axis separates
        return bool(np.any(max1 < min2 - eps) or np.any(max2 < min1 - eps))

    def vec_all_disjoint(centers, angles, eps=1e-7):
        """Check all 66 pairs are disjoint. Returns True if all disjoint."""
        verts_list = [vec_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if not vec_sat_disjoint(verts_list[i], verts_list[j], eps):
                    return False
        return True

    def vec_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Compute minimum outer hexagon side length."""
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            na = otheta + math.pi / 6 + k * np.pi / 3
            nx, ny = math.cos(na), math.sin(na)
            for i in range(N):
                verts = vec_hex_verts(centers[i][0], centers[i][1], angles[i])
                proj = (verts[:, 0] - ocx) * nx + (verts[:, 1] - ocy) * ny
                if proj.max() > max_proj:
                    max_proj = proj.max()
        return 2.0 * max_proj / SQRT3

    def unpack(x):
        """Unpack 39-dim vector to (centers, angles, ocx, ocy, otheta)."""
        centers = np.array([[x[3 * i], x[3 * i + 1]] for i in range(N)])
        angles = np.array([x[3 * i + 2] for i in range(N)])
        ocx, ocy, otheta = x[36], x[37], x[38]
        return centers, angles, ocx, ocy, otheta

    def objective(x):
        """DE objective: outer side + large penalty for overlap."""
        if time.time() > deadline - 5.0:
            return 1e10
        centers, angles, ocx, ocy, otheta = unpack(x)
        # Penalty for center-distance violations (cheap pre-check)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2
                if d2 < MIN_DIST ** 2:
                    penalty += (MIN_DIST ** 2 - d2) * 10.0
        if penalty > 0.5:
            return vec_outer_side(centers, angles, ocx, ocy, otheta) + penalty * 100.0
        # Full SAT check
        if not vec_all_disjoint(centers, angles):
            return vec_outer_side(centers, angles, ocx, ocy, otheta) + 50.0
        return vec_outer_side(centers, angles, ocx, ocy, otheta)

    def nm_objective(x):
        """Nelder-Mead objective with hard constraint."""
        centers, angles, ocx, ocy, otheta = unpack(x)
        if not vec_all_disjoint(centers, angles):
            return 1e8
        return vec_outer_side(centers, angles, ocx, ocy, otheta)

    # --- Initial configuration: hexagonal lattice ---
    d = SQRT3 + 0.01
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * SQRT3 / 2.0])
    basis = [u, v, v - u, -u, -v, u - v]
    ring1 = [u, v, v - u, -u, -v, u - v]
    centers0 = [[0.0, 0.0]] + [list(b) for b in ring1]
    # Ring 2: pick 5 shortest candidates
    ring2_cands = []
    for i in range(6):
        p1 = ring1[i]
        for b in basis:
            p2 = p1 + b
            if np.linalg.norm(p2) < 0.01:
                continue
            if any(np.linalg.norm(p2 - r1) < 0.01 for r1 in ring1):
                continue
            if any(np.linalg.norm(p2 - rc) < 0.01 for rc in ring2_cands):
                continue
            ring2_cands.append(p2)
    ring2_cands.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers0.append(list(ring2_cands[i]))
    angles0 = [30.0] * N
    x0 = []
    for i in range(N):
        x0.extend([centers0[i][0], centers0[i][1], angles0[i]])
    x0.extend([0.0, 0.0, 0.0])

    # --- Differential Evolution (global search) ---
    best_x = np.array(x0)
    best_side = objective(best_x)

    bounds = [(-5, 5), (-5, 5), (-180, 180)] * N + [(-2, 2), (-2, 2), (-30, 30)]

    if time.time() < deadline - 60.0:
        try:
            de_result = differential_evolution(
                objective, bounds,
                seed=42, maxiter=80, popsize=25,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, init='sobol',
                updating='immediate',
                time_limit=max(0, deadline - time.time() - 60.0)
            )
            if de_result.fun < best_side:
                best_side = de_result.fun
                best_x = de_result.x
        except Exception:
            pass

    # --- Nelder-Mead local polish (multiple restarts) ---
    def nm_polish(x_start, max_iter=500):
        """Local Nelder-Mead polish."""
        try:
            res = minimize(nm_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-10})
            if res.fun < 1e7:
                return res.x, res.fun
        except Exception:
            pass
        return x_start, nm_objective(x_start)

    # Polish from DE result
    if time.time() < deadline - 20.0:
        best_x, best_side = nm_polish(best_x, max_iter=400)

    # Additional Nelder-Mead restarts from perturbed initial configs
    restart_configs = [best_x]
    # Perturb angles
    for trial in range(4):
        if time.time() > deadline - 25.0:
            break
        x_trial = list(best_x)
        for i in range(N):
            x_trial[3 * i + 2] = random.uniform(-30, 30)
        # Slight position perturbation
        for i in range(N):
            x_trial[3 * i] += random.uniform(-0.1, 0.1)
            x_trial[3 * i + 1] += random.uniform(-0.1, 0.1)
        x_trial = np.array(x_trial)
        restart_configs.append(x_trial)

    for x_trial in restart_configs:
        if time.time() > deadline - 15.0:
            break
        x_polished, side_polished = nm_polish(x_trial, max_iter=300)
        if side_polished < best_side:
            best_side = side_polished
            best_x = x_polished

    # --- Final high-precision outer hexagon polish ---
    centers, angles, ocx, ocy, otheta = unpack(best_x)

    def polish_outer(ocx, ocy, otheta):
        """Multi-scale search for optimal outer hexagon parameters."""
        best_side = vec_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        scales = [0.1, 0.01, 0.001, 0.0001]
        for scale in scales:
            if time.time() > deadline - 5.0:
                break
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da in [-scale * 10, 0, scale * 10]:
                            s = vec_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                            if s < best_side - 1e-13:
                                best_side = s
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta += da
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    if time.time() < deadline - 5.0:
        ocx, ocy, otheta, best_side = polish_outer(ocx, ocy, otheta)

    # --- Build return value ---
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
