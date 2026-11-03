import numpy as np
from scipy.optimize import minimize
import time


def construct():
    t_start = time.time()
    deadline = 205.0

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

    # Seed point
    seed_point = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = objective(seed_point)
    best_result = seed_point.copy()

    def objective_sphere(params):
        theta, phi = params
        a0 = np.sin(theta) * np.cos(phi)
        a4 = np.sin(theta) * np.sin(phi)
        a8 = np.cos(theta)
        P = build_P(a0, a4, a8)
        return objective_from_P(P)

    # Convert seed to spherical coordinates
    norm_seed = np.linalg.norm(seed_point)
    seed_unit = seed_point / norm_seed
    theta_s = np.arccos(np.clip(seed_unit[2], -1, 1))
    phi_s = np.arctan2(seed_unit[1], seed_unit[0])
    if phi_s < 0:
        phi_s += 2 * np.pi

    starts = [[theta_s, phi_s]]

    # Generate random starts on the sphere
    rng = np.random.RandomState(42)
    for _ in range(100):
        z = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        theta = np.arccos(z)
        starts.append([theta, phi])
        
        # Also add random Cartesian points
        a0 = rng.uniform(-1, 1)
        a4 = rng.uniform(-1, 1)
        a8 = rng.uniform(-1, 1)
        norm = np.sqrt(a0**2 + a4**2 + a8**2)
        if norm > 1e-6:
            a0 /= norm; a4 /= norm; a8 /= norm
            theta = np.arccos(np.clip(a8, -1, 1))
            phi = np.arctan2(a4, a0)
            if phi < 0: phi += 2 * np.pi
            starts.append([theta, phi])

    for start in starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective_sphere, start, method='BFGS', 
                           options={'maxiter': 1000, 'gtol': 1e-10})
            if res.success and res.fun < best_score:
                best_score = res.fun
                theta, phi = res.x
                a0 = np.sin(theta) * np.cos(phi)
                a4 = np.sin(theta) * np.sin(phi)
                a8 = np.cos(theta)
                best_result = np.array([a0, a4, a8])
        except Exception:
            pass

    return best_result
