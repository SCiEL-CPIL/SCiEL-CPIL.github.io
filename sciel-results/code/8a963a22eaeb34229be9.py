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

    # The objective is homogeneous of degree 4 in (a0, a4, a8).
    # We can fix the scale by normalizing: set a0 = -1 (since seed has a0 < 0)
    # and search over (a4/a0, a8/a0) = (u, v) where u = a4/a0, v = a8/a0.
    # Then a0 = -1, a4 = -u, a8 = -v.
    # Actually, let's use: fix ||(a0,a4,a8)|| = 1 and search on the sphere.
    # Simpler: fix a0 = 1 and search (a4, a8), but the seed has a0 < 0.
    # Let's just fix the norm of the 3-vector to 1.
    
    def objective_normalized(theta, phi):
        """theta, phi are spherical coordinates on the unit sphere in R^3."""
        a0 = np.sin(theta) * np.cos(phi)
        a4 = np.sin(theta) * np.sin(phi)
        a8 = np.cos(theta)
        # We need a0 < 0 region for the seed. Let's allow all and check.
        # Actually the objective is symmetric under (a0,a4,a8) -> -(a0,a4,a8)
        # since it's degree 4. So we can restrict to a0 >= 0.
        P = build_P(a0, a4, a8)
        return objective_from_P(P)

    # Better approach: since objective is homogeneous degree 4, 
    # minimize over the unit sphere. Use 2D parameters (theta, phi).
    # But we need to be careful about the sign. The seed has a0 = -1.13.
    # Due to symmetry, (a0,a4,a8) and (-a0,-a4,-a8) give same objective.
    # So we can restrict to a0 > 0.
    
    def obj_2d(params):
        theta, phi = params
        # theta in [0, pi], phi in [0, 2*pi)
        # Restrict to a0 = sin(theta)*cos(phi) > 0
        a0 = np.sin(theta) * np.cos(phi)
        a4 = np.sin(theta) * np.sin(phi)
        a8 = np.cos(theta)
        P = build_P(a0, a4, a8)
        return objective_from_P(P)

    # Initial guess: convert seed to spherical
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    # Flip sign to make a0 > 0
    x0_pos = -x0  # [1.13, -0.04, -0.000308...]
    norm0 = np.linalg.norm(x0_pos)
    x0_unit = x0_pos / norm0
    theta0 = np.arccos(x0_unit[2])
    phi0 = np.arctan2(x0_unit[1], x0_unit[0])
    if phi0 < 0:
        phi0 += 2 * np.pi

    best_result = None
    best_score = np.inf

    # Phase 1: Differential Evolution on the 2D sphere
    if time.time() < deadline - 60:
        try:
            bounds = [(0.01, np.pi - 0.01), (0.0, 2 * np.pi)]
            de_result = differential_evolution(
                obj_2d, bounds,
                seed=42, maxiter=200, popsize=30,
                tol=1e-10, atol=1e-12,
                polish=True, init='latinhypercube'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_theta, best_phi = de_result.x
                a0 = np.sin(best_theta) * np.cos(best_phi)
                a4 = np.sin(best_theta) * np.sin(best_phi)
                a8 = np.cos(best_theta)
                best_result = np.array([a0, a4, a8])
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead on the 2D sphere
    if time.time() < deadline - 20:
        start_2d = [theta0, phi0]
        if best_result is not None:
            start_2d = [best_theta, best_phi]
        
        try:
            nm_result = minimize(
                obj_2d, start_2d,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-16,
                         'adaptive': True}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_theta, best_phi = nm_result.x
                a0 = np.sin(best_theta) * np.cos(best_phi)
                a4 = np.sin(best_theta) * np.sin(best_phi)
                a8 = np.cos(best_theta)
                best_result = np.array([a0, a4, a8])
        except Exception:
            pass

    # Phase 3: Convert back to 3D and do a final high-precision polish
    # in the original 3D space (unconstrained, since homogeneity means
    # the gradient is zero along the radial direction, but BFGS can still
    # find the exact minimum on the level set).
    if best_result is not None and time.time() < deadline - 10:
        try:
            # Scale back: the 3D objective is homogeneous, so we can
            # just minimize in 3D starting from our best point.
            result = minimize(
                objective,
                best_result,
                method='BFGS',
                options={'maxiter': 10000, 'gtol': 1e-16, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Phase 4: Final Nelder-Mead polish in 3D
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20,
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
