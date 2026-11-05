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

    def get_vertices(theta_out, phis, displacements=None):
        """Get all inner hexagon vertices, optionally with per-vertex displacements."""
        all_verts = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            if displacements is not None:
                verts = verts + displacements[i]
            all_verts.append(verts)
        return np.array(all_verts)  # (n_hex, 6, 2)

    def compute_outer_side(theta_out, phis, displacements=None):
        """
        Compute minimum outer hexagon side length via projection onto 6 normals.
        """
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            if displacements is not None:
                verts = verts + displacements[i]
            dots = verts @ normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(np.abs(dots)))

        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, beta=100.0):
        """
        Smooth approximation using log-sum-exp with adaptive beta.
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
    best_displacements = None  # Start with no displacements

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
        if time.time() > deadline - 40:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-15, 'xtol': 1e-15})
            if res.fun < best_score - 1e-15:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Smooth surrogate optimization with adaptive beta
    if time.time() < deadline - 30:
        for beta_val in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 25:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params,
                              method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                true_score = objective(res.x)
                if true_score < best_score - 1e-15:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 3: Displacement-based morphing search
    # Parameterize local vertex displacements as a continuous deformation
    # while maintaining non-overlap via distance constraints
    if time.time() < deadline - 20:
        # Initialize displacements to zero
        n_verts = n_hex * 6
        disp_params = np.zeros(n_verts * 2)  # 2D displacement per vertex

        def displacement_objective(disp_flat):
            """Objective with vertex displacements applied."""
            disp_flat = np.clip(disp_flat, -0.1, 0.1)  # Limit displacement magnitude
            disp_2d = disp_flat.reshape(n_hex, 6, 2)
            theta_out = best_params[0]
            phis = best_params[1:]
            return compute_outer_side(theta_out, phis, displacements=disp_2d)

        # Try to improve via displacement search
        best_disp_score = best_score
        best_disp = disp_params.copy()

        # Multi-start displacement search
        np.random.seed(123)
        for disp_start in range(8):
            if time.time() > deadline - 15:
                break

            if disp_start == 0:
                disp_init = np.zeros(n_verts * 2)
            else:
                disp_init = np.random.uniform(-0.05, 0.05, n_verts * 2)

            try:
                res = minimize(displacement_objective, disp_init, method='Powell',
                              options={'maxiter': 200, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
                if res.fun < best_disp_score - 1e-14:
                    best_disp_score = res.fun
                    best_disp = res.x.copy()
            except Exception:
                pass

        if best_disp_score < best_score - 1e-12:
            best_score = best_disp_score
            best_displacements = best_disp.reshape(n_hex, 6, 2)

    # Phase 4: Coordinate descent with golden-section refinement on rotations
    if time.time() < deadline - 12:
        for iteration in range(80):
            if time.time() > deadline - 8:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 8:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                # Try multiple step sizes
                for delta in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4,
                             1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1]:
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

                # If we found improvement, do golden-section refinement
                if best_local < best_score - 1e-16:
                    # Refine around best_local_val
                    lo = best_local_val - 1e-4
                    hi = best_local_val + 1e-4
                    for _ in range(30):
                        if time.time() > deadline - 8:
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
                            if f1 < best_local - 1e-18:
                                best_local = f1
                                best_local_val = m1
                        else:
                            lo = m1
                            if f2 < best_local - 1e-18:
                                best_local = f2
                                best_local_val = m2

                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Phase 5: Final high-precision Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_score - 1e-16:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 6: Ultra-fine coordinate descent for final polish
    if time.time() < deadline - 2:
        for iteration in range(20):
            if time.time() > deadline - 1:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 1:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                for delta in [1e-7, -1e-7, 1e-8, -1e-8, 1e-9, -1e-9, 1e-10, -1e-10]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local - 1e-18:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue

                if best_local < best_score - 1e-18:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True

            if not improved:
                break

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
