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
        Fully vectorized for speed.
        """
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        # Vectorized: compute rotated vertices for all hexagons at once
        c = np.cos(phis)  # (n_hex,)
        s = np.sin(phis)  # (n_hex,)
        # Rotation matrix applied to base_vertices: (n_hex, 6, 2)
        # R = [[c, -s], [s, c]] for each hex
        # verts[i] = base_vertices @ R_i.T + centers[i]
        # = [[bv_x*c - bv_y*s, bv_x*s + bv_y*c] for bv in base_vertices] + center_i
        bv_x = base_vertices[:, 0]  # (6,)
        bv_y = base_vertices[:, 1]  # (6,)
        # (n_hex, 6) each component
        vx = bv_x[None, :] * c[:, None] - bv_y[None, :] * s[:, None] + centers[:, 0][:, None]
        vy = bv_x[None, :] * s[:, None] + bv_y[None, :] * c[:, None] + centers[:, 1][:, None]
        
        # Project onto normals: dots[i,j,k] = verts[i,j] . normals[k]
        # dots shape: (n_hex, 6, 6)
        dots_x = vx[:, :, None] * normals[None, None, 0]  # (n_hex, 6, 6)
        dots_y = vy[:, :, None] * normals[None, None, 1]
        all_abs_dots = np.abs(dots_x + dots_y)  # (n_hex, 6, 6)
        max_proj = np.max(all_abs_dots)

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

        c = np.cos(phis)
        s = np.sin(phis)
        bv_x = base_vertices[:, 0]
        bv_y = base_vertices[:, 1]
        vx = bv_x[None, :] * c[:, None] - bv_y[None, :] * s[:, None] + centers[:, 0][:, None]
        vy = bv_x[None, :] * s[:, None] + bv_y[None, :] * c[:, None] + centers[:, 1][:, None]
        
        dots_x = vx[:, :, None] * normals[None, None, 0]
        dots_y = vy[:, :, None] * normals[None, None, 1]
        all_abs_dots = np.abs(dots_x + dots_y).ravel()

        max_val = np.max(all_abs_dots)
        if max_val == 0:
            return 0.0
        shifted = (all_abs_dots - max_val) * beta
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)) / len(all_abs_dots)) / beta

        return 2.0 * log_sum_exp / np.sqrt(3)

    def displacement_objective(params, beta=100.0):
        """
        Alternative smooth surrogate: sum of squared displacements of vertices
        beyond the inscribed circle of the outer hexagon.
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]

        # Inscribed circle radius of outer hexagon with side S: r = S * sqrt(3)/2
        # We want all vertices within this circle.
        # Displacement = max(0, |v| - r)^2 summed over all vertices
        # But we need to minimize S, so we use: S = 2*max|v|/sqrt(3)
        # Smooth version: S_smooth = 2 * softmax(|v|) / sqrt(3)
        
        c = np.cos(phis)
        s = np.sin(phis)
        bv_x = base_vertices[:, 0]
        bv_y = base_vertices[:, 1]
        vx = bv_x[None, :] * c[:, None] - bv_y[None, :] * s[:, None] + centers[:, 0][:, None]
        vy = bv_x[None, :] * s[:, None] + bv_y[None, :] * c[:, None] + centers[:, 1][:, None]
        
        # Distances from origin (outer hexagon center)
        dists = np.sqrt(vx**2 + vy**2).ravel()
        max_d = np.max(dists)
        if max_d == 0:
            return 0.0
        shifted = (dists - max_d) * beta
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = max_d + np.log(np.sum(np.exp(shifted)) / len(dists)) / beta
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
    n_starts = 12
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
        if time.time() > deadline - 50:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 200, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score - 1e-14:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Smooth surrogate optimization with adaptive beta
    if time.time() < deadline - 40:
        for beta_val in [50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 35:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params,
                              method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                true_score = objective(res.x)
                if true_score < best_score - 1e-14:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

        # Also try displacement-based surrogate
        if time.time() < deadline - 30:
            for beta_val in [100.0, 300.0]:
                if time.time() > deadline - 25:
                    break
                try:
                    res = minimize(lambda p: displacement_objective(p, beta=beta_val), best_params,
                                  method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                    true_score = objective(res.x)
                    if true_score < best_score - 1e-14:
                        best_score = true_score
                        best_params = res.x.copy()
                except Exception:
                    pass

    # Phase 3: Epigraph formulation optimization
    if time.time() < deadline - 25:
        for beta_val in [100.0, 300.0]:
            if time.time() > deadline - 20:
                break
            try:
                t_init = best_score
                x_epigraph = np.concatenate([best_params, [t_init]])
                
                res = minimize(lambda p: epigraph_objective(p, beta=beta_val), x_epigraph,
                              method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                rot_params = res.x[:-1]
                true_score = objective(rot_params)
                if true_score < best_score - 1e-14:
                    best_score = true_score
                    best_params = rot_params.copy()
            except Exception:
                pass

    # Phase 4: High-precision Nelder-Mead polish on the nonsmooth objective
    if time.time() < deadline - 15:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_score - 1e-16:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Coordinate descent with golden-section refinement on rotations
    if time.time() < deadline - 8:
        for iteration in range(60):
            if time.time() > deadline - 4:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 4:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                # Try multiple step sizes including finer ones
                for delta in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 5e-2, -5e-2]:
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
                    lo = best_local_val - 1e-5
                    hi = best_local_val + 1e-5
                    for _ in range(30):
                        if time.time() > deadline - 4:
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
    if time.time() < deadline - 3:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-17, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_score - 1e-18:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 7: One more coordinate descent pass with very fine steps
    if time.time() < deadline - 1:
        for j in range(n_hex + 1):
            if time.time() > deadline - 0.5:
                break
            current_val = best_params[j]
            best_local = best_score
            best_local_val = current_val
            for delta in [1e-7, -1e-7, 1e-8, -1e-8]:
                test_val = current_val + delta
                test_param = best_params.copy()
                test_param[j] = test_val
                try:
                    test_score = objective(test_param)
                    if test_score < best_local - 1e-16:
                        best_local = test_score
                        best_local_val = test_val
                except Exception:
                    continue
            best_params[j] = best_local_val
            best_score = best_local

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
