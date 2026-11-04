import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Initial centers: flower pattern (1 center + 6 at sqrt(3) + 4 at 3)
    eps = 1e-12
    base_centers = np.array([[0.0, 0.0]])
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        base_centers = np.vstack([base_centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        base_centers = np.vstack([base_centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(base_centers)

    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis, centers):
        """
        Compute the minimum outer hexagon side length given outer rotation, inner rotations, and centers.
        Uses projection onto the 6 face normals of the outer hexagon.
        Vectorized for speed.
        """
        normal_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            vx = base_vertices[:, 0] * c - base_vertices[:, 1] * s
            vy = base_vertices[:, 0] * s + base_vertices[:, 1] * c
            vx += centers[i][0]
            vy += centers[i][1]
            dots = vx * normals[:, 0] + vy * normals[:, 1]
            max_proj = max(max_proj, np.max(np.abs(dots)))

        return 2.0 * max_proj / np.sqrt(3)

    def check_non_overlap(centers, phis, tol=1e-10):
        """
        Check if hexagons are non-overlapping.
        Returns True if all pairs are separated by at least tol.
        Uses center distance heuristic for speed: if center distance < 2*sqrt(3), do detailed check.
        """
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                dist = np.linalg.norm(centers[i] - centers[j])
                # Minimum distance between two unit hexagons depends on relative rotation
                # Conservative bound: if dist < 2*sqrt(3) - tol, they might overlap
                if dist < 2.0 * np.sqrt(3) - tol:
                    # Detailed check: compute minimum distance between edges
                    v1 = _get_hex_vertices(centers[i], phis[i])
                    v2 = _get_hex_vertices(centers[j], phis[j])
                    min_dist = _min_poly_dist(v1, v2)
                    if min_dist < tol:
                        return False
        return True

    def _get_hex_vertices(center, angle):
        c, s = np.cos(angle), np.sin(angle)
        vx = base_vertices[:, 0] * c - base_vertices[:, 1] * s
        vy = base_vertices[:, 0] * s + base_vertices[:, 1] * c
        return np.column_stack([vx + center[0], vy + center[1]])

    def _min_poly_dist(p1, p2):
        """Minimum distance between two convex hexagons."""
        min_d = float('inf')
        for i in range(6):
            a1, a2 = p1[i], p1[(i+1) % 6]
            for j in range(6):
                b1, b2 = p2[j], p2[(j+1) % 6]
                d = _seg_seg_dist(a1, a2, b1, b2)
                if d < min_d:
                    min_d = d
        return min_d

    def _seg_seg_dist(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        if a < 1e-15 and e < 1e-15:
            return np.linalg.norm(p1 - p3)
        if a < 1e-15:
            t = np.clip(f / e, 0, 1)
            return np.linalg.norm(p1 - (p3 + t * d2))
        c = np.dot(d1, r)
        if e < 1e-15:
            s = np.clip(c / a, 0, 1)
            return np.linalg.norm((p1 + s * d1) - p3)
        b = np.dot(d1, d2)
        denom = a * e - b * b
        if denom > 1e-15:
            s = np.clip((b * f - c * e) / denom, 0, 1)
            t = (b * s + f) / e
            if t < 0:
                t = 0
                s = np.clip(c / a, 0, 1)
            elif t > 1:
                t = 1
                s = np.clip((b + f) / a, 0, 1)
            return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
        else:
            t = np.clip(f / e, 0, 1)
            return np.linalg.norm((p1 + np.clip(c / a, 0, 1) * d1) - (p3 + t * d2))

    def smooth_objective(params, beta=100.0):
        """
        Smooth approximation using log-sum-exp.
        params: [theta_out, phi_0..10, dx_0, dy_0, ..., dx_10, dy_10]
        """
        theta_out = params[0]
        phis = params[1:1+n_hex]
        displacements = params[1+n_hex:].reshape(n_hex, 2)
        centers = base_centers + displacements

        normal_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        all_abs_dots = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            vx = base_vertices[:, 0] * c - base_vertices[:, 1] * s
            vy = base_vertices[:, 0] * s + base_vertices[:, 1] * c
            vx += centers[i][0]
            vy += centers[i][1]
            dots = vx * normals[:, 0] + vy * normals[:, 1]
            all_abs_dots.append(np.abs(dots))

        all_abs_dots = np.concatenate(all_abs_dots)
        max_val = np.max(all_abs_dots)
        if max_val == 0:
            return 0.0
        scaled = (all_abs_dots - max_val) * beta
        log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - np.max(scaled))) / len(all_abs_dots)) / beta + np.max(scaled) / beta

        side = 2.0 * log_sum_exp / np.sqrt(3)

        # Add overlap penalty
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                dist = np.linalg.norm(centers[i] - centers[j])
                # Soft constraint: penalize if too close
                min_allowed = 2.0 * np.sqrt(3) * 0.999  # Slightly relaxed for smoothness
                if dist < min_allowed:
                    penalty += (min_allowed - dist) ** 2 * 10.0

        return side + penalty

    def objective(params):
        """
        params: [theta_out, phi_0..10, dx_0, dy_0, ..., dx_10, dy_10]
        Returns outer side length + overlap penalty.
        """
        theta_out = params[0]
        phis = params[1:1+n_hex]
        displacements = params[1+n_hex:].reshape(n_hex, 2)
        centers = base_centers + displacements

        # Check non-overlap
        if not check_non_overlap(centers, phis, tol=1e-8):
            # Compute penalty based on how much they overlap
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    v1 = _get_hex_vertices(centers[i], phis[i])
                    v2 = _get_hex_vertices(centers[j], phis[j])
                    min_dist = _min_poly_dist(v1, v2)
                    if min_dist < 0:
                        penalty += abs(min_dist) * 100.0
            if penalty > 0:
                return 100.0 + penalty

        side = compute_outer_side(theta_out, phis, centers)
        return side

    # Analytical initial guess
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex + [0.0] * (2 * n_hex))

    best_score = objective(x0)
    best_params = x0.copy()

    # Phase 1: Multi-start Powell optimization
    n_starts = 12
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex + [0.0] * (2*n_hex)),
        np.array([0.0] + [0.0] * n_hex + [0.0] * (2*n_hex)),
        np.array([np.pi/6] + [0.0] * n_hex + [0.0] * (2*n_hex)),
        np.array([0.0] + [np.pi/6] * n_hex + [0.0] * (2*n_hex)),
        np.array([np.deg2rad(15)] + [np.deg2rad(15)] * n_hex + [0.0] * (2*n_hex)),
        np.array([np.deg2rad(45)] + [np.deg2rad(45)] * n_hex + [0.0] * (2*n_hex)),
    ]

    np.random.seed(42)
    for i in range(n_starts):
        if time.time() > deadline - 50:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = x0.copy()
            # Random small perturbations
            x_start[1+n_hex:] = np.random.uniform(-0.1, 0.1, 2*n_hex)
            x_start[1:1+n_hex] += np.random.uniform(-0.05, 0.05, n_hex)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score - 1e-12:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Smooth surrogate optimization
    if time.time() < deadline - 30:
        for beta_val in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 25:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params,
                             method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                true_score = objective(res.x)
                if true_score < best_score - 1e-12:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 3: Coordinate descent with golden-section refinement
    if time.time() < deadline - 20:
        for iteration in range(50):
            if time.time() > deadline - 15:
                break

            improved = False
            for j in range(len(best_params)):
                if time.time() > deadline - 15:
                    break

                # Golden-section search around current value
                current_val = best_params[j]
                # Try a range of values
                for delta in [0.01, -0.01, 0.001, -0.001, 0.0001, -0.0001,
                             0.00001, -0.00001, 0.000001, -0.000001]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_score - 1e-14:
                            best_score = test_score
                            best_params = test_param.copy()
                            improved = True
                            current_val = test_val
                    except Exception:
                        continue

            if not improved:
                break

    # Phase 4: Final high-precision Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_score - 1e-14:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Ultra-fine polish
    if time.time() < deadline - 5:
        for iteration in range(20):
            if time.time() > deadline - 3:
                break

            improved = False
            for j in range(len(best_params)):
                if time.time() > deadline - 3:
                    break

                current_val = best_params[j]
                for delta in [1e-9, -1e-9, 1e-10, -1e-10, 1e-11, -1e-11]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_score - 1e-16:
                            best_score = test_score
                            best_params = test_param.copy()
                            improved = True
                    except Exception:
                        continue

            if not improved:
                break

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:1+n_hex]
    displacements = best_params[1+n_hex:].reshape(n_hex, 2)
    final_centers = base_centers + displacements

    inner = []
    for i in range(n_hex):
        inner.append([final_centers[i][0], final_centers[i][1], np.rad2deg(phis[i])])

    side = best_score + 1e-8  # Margin for floating point safety

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
