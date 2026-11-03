import numpy as np
from scipy.optimize import minimize, differential_evolution
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

    # Signs for Q: Q[m] gets coefficient from P[2m] * (-1)^m
    signs = np.array([(-1.0) ** m for m in range(7)])

    def objective_from_coeffs(a0, a4, a8):
        """Compute the bound from three free coefficients."""
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

        P = np.zeros(7)
        P[0] += a0 * H0_arr[0]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        for m in range(7):
            P[m] += a12 * H12_arr[m]

        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)

        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)

        return norm_sq * norm_sq_hat

    def objective(params):
        """Objective function for optimizer (takes array of 3 params)."""
        a0, a4, a8 = params
        return objective_from_coeffs(a0, a4, a8)

    def objective_spherical(theta, phi):
        """Spherical parameterization: removes scale ambiguity.
        theta in [0, pi], phi in [0, 2*pi]
        """
        a0 = np.sin(theta) * np.cos(phi)
        a4 = np.sin(theta) * np.sin(phi)
        a8 = np.cos(theta)
        return objective_from_coeffs(a0, a4, a8)

    # Seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_score = objective(x0)

    best_result = x0.copy()
    best_score = seed_score

    # Convert seed to spherical coordinates
    seed_norm = np.linalg.norm(x0)
    if seed_norm > 0:
        x0_unit = x0 / seed_norm
        theta_init = np.arccos(np.clip(x0_unit[2], -1.0, 1.0))
        phi_init = np.arctan2(x0_unit[1], x0_unit[0])
        if phi_init < 0:
            phi_init += 2 * np.pi
    else:
        theta_init = np.pi / 2
        phi_init = 0.0

    # Strategy 1: Differential Evolution in spherical coordinates (2D)
    if time.time() < deadline - 60:
        try:
            bounds_sph = [(1e-8, np.pi - 1e-8), (1e-8, 2 * np.pi - 1e-8)]
            de_result = differential_evolution(
                lambda p: objective_spherical(p[0], p[1]),
                bounds=bounds_sph,
                seed=42,
                maxiter=300,
                tol=1e-12,
                atol=1e-14,
                popsize=30,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                # Convert back to Cartesian
                th, ph = de_result.x
                best_result = np.array([
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th)
                ])
        except Exception:
            pass

    # Strategy 2: Local optimization in Cartesian from multiple starting points
    start_points = [x0.copy()]

    if best_result is not None and np.allclose(best_result, x0) == False:
        start_points.append(best_result.copy())
        # Add perturbations around the best result
        for scale in [0.001, 0.01, 0.1, 0.5]:
            for dim in range(3):
                perturbed = best_result.copy()
                perturbed[dim] += scale
                start_points.append(perturbed.copy())
                perturbed[dim] -= scale
                start_points.append(perturbed.copy())

    # Add diverse starting points
    seed_perturbations = [
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.10, 0.04, 0.0003]),
        np.array([-1.16, 0.04, 0.0003]),
        np.array([-1.13, 0.03, 0.0003]),
        np.array([-1.13, 0.05, 0.0003]),
        np.array([-1.13, 0.04, 0.0002]),
        np.array([-1.13, 0.04, 0.0004]),
        np.array([-1.13, 0.04, 0.00035]),
        np.array([-1.13, 0.04, 0.00025]),
        np.array([-1.12, 0.041, 0.00031]),
        np.array([-1.14, 0.039, 0.00029]),
        np.array([-1.11, 0.042, 0.00032]),
        np.array([-1.15, 0.038, 0.00028]),
        np.array([-1.20, 0.05, 0.0005]),
        np.array([-1.00, 0.03, 0.0002]),
        np.array([-1.30, 0.06, 0.0006]),
        np.array([-0.90, 0.02, 0.0001]),
        np.array([-1.13, 0.04, 0.0003]),  # seed again for consistency
    ]
    start_points.extend(seed_perturbations)

    for sp in start_points:
        if time.time() > deadline - 40:
            break

        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True}
            )

            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()

        except Exception:
            continue

    # Strategy 3: BFGS polish
    if best_result is not None and time.time() < deadline - 25:
        try:
            result = minimize(
                objective,
                best_result,
                method='BFGS',
                options={'maxiter': 100000, 'gtol': 1e-20, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Strategy 4: L-BFGS-B polish
    if best_result is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 100000, 'ftol': 1e-20, 'gtol': 1e-20}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Strategy 5: Powell's method for final refinement
    if best_result is not None and time.time() < deadline - 8:
        try:
            result = minimize(
                objective,
                best_result,
                method='Powell',
                options={'maxiter': 100000, 'xtol': 1e-16, 'ftol': 1e-19}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Strategy 6: Final ultra-high-precision Nelder-Mead
    if best_result is not None and time.time() < deadline - 3:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 200000, 'xatol': 1e-18, 'fatol': 1e-20,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # If no valid result, fall back to seed
    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return best_result
