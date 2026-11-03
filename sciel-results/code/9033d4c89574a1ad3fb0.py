import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
import time


def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Coefficients for H0, H4, H8, H12 as arrays indexed by power//2
    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    # Precompute ∫ x^{2k} e^{-πx²} dx for k=0..12
    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Signs for Q
    signs = np.array([(-1.0) ** m for m in range(7)])

    def build_P(a0, a4, a8):
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        P = np.zeros(7)
        P[0] += a0 * H0_arr[0]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        for m in range(7):
            P[m] += a12 * H12_arr[m]
        return P

    def objective_from_P(P):
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        return norm_sq * norm_sq_hat

    def objective(params):
        a0, a4, a8 = params
        P = build_P(a0, a4, a8)
        return objective_from_P(P)

    # Root-parameterization approach:
    # The three coefficients (a0, a4, a8) define a cubic in the Hermite basis.
    # We parameterize via roots: let the "Hermite polynomial" be 
    #   Q(t) = t^3 + p*t^2 + q*t + r
    # and map this to (a0, a4, a8) via a linear transformation.
    # 
    # Actually, a cleaner approach: since the objective is homogeneous degree 4,
    # we can work on the unit sphere. But the root parameterization gives us
    # a 3D -> 3D map (roots -> coefficients) that may explore the landscape better.
    #
    # Alternative: use the fact that we have 3 free parameters and the objective
    # is smooth. Use multiple restarts with different initial points.
    
    def obj_2d(params):
        """Objective on the 2D sphere using spherical coordinates."""
        theta, phi = params
        a0 = np.sin(theta) * np.cos(phi)
        a4 = np.sin(theta) * np.sin(phi)
        a8 = np.cos(theta)
        P = build_P(a0, a4, a8)
        return objective_from_P(P)

    # Convert seed to spherical coordinates
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    x0_pos = -x0  # Flip to make a0 > 0 (objective is even)
    norm0 = np.linalg.norm(x0_pos)
    x0_unit = x0_pos / norm0
    theta0 = np.arccos(np.clip(x0_unit[2], -1, 1))
    phi0 = np.arctan2(x0_unit[1], x0_unit[0])
    if phi0 < 0:
        phi0 += 2 * np.pi

    best_result = None
    best_score = np.inf

    def update_best(score, result):
        nonlocal best_score, best_result
        if score < best_score:
            best_score = score
            best_result = result.copy()

    # Phase 1: Differential Evolution on the 2D sphere
    if time.time() < deadline - 80:
        try:
            bounds = [(0.001, np.pi - 0.001), (0.0, 2 * np.pi)]
            de_result = differential_evolution(
                obj_2d, bounds,
                seed=42, maxiter=300, popsize=40,
                tol=1e-12, atol=1e-14,
                polish=True, init='latinhypercube',
                mutation=(0.5, 1.5), recombination=0.9
            )
            if de_result.fun < best_score:
                best_theta, best_phi = de_result.x
                a0 = np.sin(best_theta) * np.cos(best_phi)
                a4 = np.sin(best_theta) * np.sin(best_phi)
                a8 = np.cos(best_theta)
                update_best(de_result.fun, np.array([a0, a4, a8]))
        except Exception:
            pass

    # Phase 2: Multiple local refinements from different starting points
    if time.time() < deadline - 30:
        # Start points: seed, DE result, and random points
        start_points = [[theta0, phi0]]
        if best_result is not None:
            # Convert best_result back to spherical
            br_norm = np.linalg.norm(best_result)
            if br_norm > 1e-15:
                br_unit = best_result / br_norm
                th = np.arccos(np.clip(br_unit[2], -1, 1))
                ph = np.arctan2(br_unit[1], br_unit[0])
                if ph < 0:
                    ph += 2 * np.pi
                start_points.append([th, ph])
        
        # Add a few random starting points
        rng = np.random.RandomState(123)
        for _ in range(5):
            th = rng.uniform(0.01, np.pi - 0.01)
            ph = rng.uniform(0, 2 * np.pi)
            start_points.append([th, ph])
        
        for sp in start_points:
            if time.time() > deadline - 15:
                break
            try:
                nm_result = minimize(
                    obj_2d, sp,
                    method='Nelder-Mead',
                    options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18,
                             'adaptive': True}
                )
                if nm_result.fun < best_score:
                    best_theta, best_phi = nm_result.x
                    a0 = np.sin(best_theta) * np.cos(best_phi)
                    a4 = np.sin(best_theta) * np.sin(best_phi)
                    a8 = np.cos(best_theta)
                    update_best(nm_result.fun, np.array([a0, a4, a8]))
            except Exception:
                pass

    # Phase 3: High-precision polish in 3D
    if best_result is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_result,
                method='BFGS',
                options={'maxiter': 50000, 'gtol': 1e-18, 'disp': False}
            )
            if result.fun < best_score:
                update_best(result.fun, result.x)
        except Exception:
            pass

    # Phase 4: Final ultra-precise Nelder-Mead polish in 3D
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 200000, 'xatol': 1e-20, 'fatol': 1e-22,
                         'adaptive': True}
            )
            if result.fun < best_score:
                update_best(result.fun, result.x)
        except Exception:
            pass

    # If no valid result, fall back to seed
    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return best_result
