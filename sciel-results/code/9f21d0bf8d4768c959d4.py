import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
    eps = 1e-12
    centers = np.array([[0.0, 0.0]])

    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers)

    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis):
        """
        Compute minimum outer hexagon side length via projection onto 6 normals.
        Vectorized for speed.
        """
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        # Precompute all rotated vertices
        all_dots = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]  # (6, 2)
            dots = verts @ normals.T  # (6, 6)
            all_dots.append(dots)

        all_dots = np.array(all_dots)  # (n_hex, 6, 6)
        max_proj = np.max(np.abs(all_dots))

        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, beta=100.0):
        """
        Smooth approximation using log-sum-exp.
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]

        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        all_abs_dots = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            all_abs_dots.append(np.abs(dots).ravel())

        all_abs_dots = np.concatenate(all_abs_dots)

        max_val = np.max(all_abs_dots)
        if max_val == 0:
            return 0.0
        shifted = (all_abs_dots - max_val) * beta
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)) / len(all_abs_dots)) / beta

        return 2.0 * log_sum_exp / np.sqrt(3)

    def epigraph_objective(params, beta=200.0):
        """
        Epigraph formulation: minimize t + penalty * max(0, f(params) - t)^2
        where f is the smooth surrogate.
        params: [theta_out, phi_0, ..., phi_10, t]
        """
        theta_out = params[0]
        phis = params[1:-1]
        t = params[-1]
        
        f_val = smooth_objective(np.concatenate([[theta_out], phis]), beta=beta)
        penalty = 100.0 * max(0, f_val - t) ** 2
        return t + penalty

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Analytical initial guess: all inner hexagons at 30 degrees (flat-top), outer at 30 degrees
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

    best_score = objective(x0)
    best_params = x0.copy()

    # Phase 1: Multi-start Powell optimization with various starting points
    n_starts = 16
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex),  # Analytical
        np.array([0.0] + [0.0] * n_hex),  # All zero
        np.array([np.pi/6] + [0.0] * n_hex),  # Outer at 30, inner at 0
        np.array([0.0] + [np.pi/6] * n_hex),  # Outer at 0, inner at 30
        np.array([np.deg2rad(0)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(60)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(15)] + [np.deg2rad(15)] * n_hex),
        np.array([np.deg2rad(45)] + [np.deg2rad(45)] * n_hex),
        np.array([np.deg2rad(20)] + [np.deg2rad(25)] * n_hex),
        np.array([np.deg2rad(25)] + [np.deg2rad(20)] * n_hex),
    ]

    np.random.seed(42)
    for i in range(n_starts):
        if time.time() > deadline - 45:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score - 1e-14:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Smooth surrogate optimization with adaptive beta (increased budget)
    if time.time() < deadline - 35:
        for beta_val in [50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params,
                              method='Nelder-Mead',
                              options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                true_score = objective(res.x)
                if true_score < best_score - 1e-14:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 3: Epigraph formulation optimization
    if time.time() < deadline - 20:
        for beta_val in [100.0, 300.0]:
            if time.time() > deadline - 15:
                break
            try:
                # Initialize t to current best
                t_init = best_score
                x_epigraph = np.concatenate([best_params, [t_init]])
                
                res = minimize(lambda p: epigraph_objective(p, beta=beta_val), x_epigraph,
                              method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                # Extract rotation params (exclude t)
                rot_params = res.x[:-1]
                true_score = objective(rot_params)
                if true_score < best_score - 1e-14:
                    best_score = true_score
                    best_params = rot_params.copy()
            except Exception:
                pass

    # Phase 4: High-precision Nelder-Mead polish on the nonsmooth objective
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_score - 1e-16:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Coordinate descent with golden-section refinement on rotations
    if time.time() < deadline - 5:
        for iteration in range(40):
            if time.time() > deadline - 3:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 3:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                # Try multiple step sizes
                for delta in [1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 5e-2, -5e-2]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local - 1e-15:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue

                # If we found improvement, do golden-section refinement
                if best_local < best_score - 1e-15:
                    # Refine around best_local_val
                    lo = best_local_val - 1e-4
                    hi = best_local_val + 1e-4
                    for _ in range(25):
                        if time.time() > deadline - 3:
                            break
                        gr = (np.sqrt(5) - 1) / 2
                        m1 = hi - gr * (hi - lo)
                        m2 = lo + gr * (hi - lo)
                        p1 = best_params.copy()
                        p1[j] = m1
                        p2 = best_params.copy()
                        p2[j] = m2
                        f1 = objective(p1)
                        f2 = objective(p2)
                        if f1 < f2:
                            hi = m2
                            if f1 < best_local - 1e-17:
                                best_local = f1
                                best_local_val = m1
                        else:
                            lo = m1
                            if f2 < best_local - 1e-17:
                                best_local = f2
                                best_local_val = m2

                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Phase 6: Final ultra-fine polish
    if time.time() < deadline - 2:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-18, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_score - 1e-18:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]

    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])

    side = best_score + 1e-10  # Tiny margin for floating point safety

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
