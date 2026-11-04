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

    # Precompute rotation matrices for base vertices
    # For a given angle phi, rotation matrix R(phi) applied to base_vertices
    # R(phi) = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]

    def compute_outer_side(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        Uses projection onto the 6 face normals of the outer hexagon.
        For a centered regular hexagon, containment requires |v . n| <= apothem for all vertices.
        Vectorized for speed.
        """
        # Outer hexagon face normals: at angles 30 + k*60 + theta_out
        normal_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        # Compute all vertex positions
        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            # Rotate base vertices
            vx = base_vertices[:, 0] * c - base_vertices[:, 1] * s
            vy = base_vertices[:, 0] * s + base_vertices[:, 1] * c
            # Translate
            vx += centers[i][0]
            vy += centers[i][1]
            # Project onto normals
            dots = vx * normals[:, 0] + vy * normals[:, 1]
            max_proj = max(max_proj, np.max(np.abs(dots)))

        # For a regular hexagon with side S, the apothem is S * sqrt(3) / 2
        # So S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, beta=100.0):
        """
        Smooth approximation of the max projection using log-sum-exp.
        This provides a differentiable surrogate that helps optimizers navigate.
        Vectorized for speed.
        """
        theta_out = params[0]
        phis = params[1:]

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

        # Log-sum-exp for stable computation
        max_val = np.max(all_abs_dots)
        if max_val == 0:
            return 0.0
        scaled = (all_abs_dots - max_val) * beta
        # Stable log-sum-exp
        max_scaled = np.max(scaled)
        log_sum_exp = max_val + max_scaled / beta + np.log(np.sum(np.exp(scaled - max_scaled))) / beta

        return 2.0 * log_sum_exp / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Analytical initial guess based on known optimal configuration
    # All inner hexagons at 30 degrees (flat-top), outer at 30 degrees
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

    best_score = objective(x0)
    best_params = x0.copy()

    # Phase 1: Multi-start Powell optimization with various starting points
    n_starts = 20
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex),  # Analytical
        np.array([0.0] + [0.0] * n_hex),  # All zero
        np.array([np.pi/6] + [0.0] * n_hex),  # Outer at 30, inner at 0
        np.array([0.0] + [np.pi/6] * n_hex),  # Outer at 0, inner at 30
        np.array([np.deg2rad(0)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(60)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(15)] + [np.deg2rad(15)] * n_hex),
        np.array([np.deg2rad(45)] + [np.deg2rad(45)] * n_hex),
        np.array([np.deg2rad(30)] + [np.deg2rad(0)] * n_hex),
        np.array([np.deg2rad(30)] + [np.deg2rad(60)] * n_hex),
    ]

    np.random.seed(42)
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            # Random starts with symmetry-breaking
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-16, 'xtol': 1e-16})
            if res.fun < best_score - 1e-15:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Smooth surrogate optimization to refine with multiple betas
    if time.time() < deadline - 25:
        for beta_val in [50.0, 100.0, 200.0, 400.0]:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params, 
                             method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                # Evaluate the true objective at the smooth optimum
                true_score = objective(res.x)
                if true_score < best_score - 1e-15:
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
            for j in range(n_hex + 1):
                if time.time() > deadline - 15:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                # Use golden-section search for 1D optimization
                # Define bounds based on symmetry: angles in [0, pi/3]
                a = 0.0
                b = np.pi / 3

                # Golden ratio conjugate
                gr = (np.sqrt(5) - 1) / 2

                # Initial points
                c = b - gr * (b - a)
                d = a + gr * (b - a)

                fc = objective(np.concatenate(([best_params[:j]], [c], [best_params[j+1:]])))
                fd = objective(np.concatenate(([best_params[:j]], [d], [best_params[j+1:]])))

                # Iterate golden-section search
                for _ in range(30):
                    if time.time() > deadline - 15:
                        break
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = objective(np.concatenate(([best_params[:j]], [c], [best_params[j+1:]])))
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = objective(np.concatenate(([best_params[:j]], [d], [best_params[j+1:]])))

                    if abs(b - a) < 1e-14:
                        break

                # Final candidate
                test_val = (a + b) / 2
                test_param = best_params.copy()
                test_param[j] = test_val
                test_score = objective(test_param)

                if test_score < best_local - 1e-16:
                    best_local = test_score
                    best_local_val = test_val

                if best_local < best_score - 1e-16:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Phase 4: Final high-precision Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_score - 1e-16:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Ultra-fine coordinate descent for final polish
    if time.time() < deadline - 5:
        for iteration in range(30):
            if time.time() > deadline - 3:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 3:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                for delta in [1e-8, -1e-8, 1e-9, -1e-9, 1e-10, -1e-10, 1e-11, -1e-11]:
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

    side = best_score + 1e-8  # Margin for floating point safety

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
