import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    t_start = time.time()
    deadline = 205.0

    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

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

    # Constraint: a0^2 + a4^2 + a8^2 = 1
    constraints = [{'type': 'eq', 'fun': lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1.0}]

    best_score = np.inf
    best_result = None

    # Multi-start SLSQP
    np.random.seed(42)
    n_starts = 100
    time_per_start = (deadline - 5.0) / n_starts
    
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Random point on sphere
        u = np.random.randn(3)
        u /= np.linalg.norm(u)
        
        try:
            res = minimize(
                objective, 
                u, 
                method='SLSQP', 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass

    # Final polish with Nelder-Mead on the 2D sphere
    if best_result is not None and time.time() < deadline - 5.0:
        # Convert to spherical
        a0, a4, a8 = best_result
        # Handle poles
        if abs(a0) > 0.999 and abs(a4) < 0.01:
            theta = 0.0 if a8 > 0 else np.pi
            phi = 0.0
        else:
            theta = np.arccos(np.clip(a8, -1.0, 1.0))
            phi = np.arctan2(a4, a0)
            if phi < 0:
                phi += 2 * np.pi

        def obj_2d(params):
            t, p = params
            a0 = np.sin(t) * np.cos(p)
            a4 = np.sin(t) * np.sin(p)
            a8 = np.cos(t)
            return objective([a0, a4, a8])

        try:
            res_nm = minimize(
                obj_2d, 
                [theta, phi], 
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                t, p = res_nm.x
                a0 = np.sin(t) * np.cos(p)
                a4 = np.sin(t) * np.sin(p)
                a8 = np.cos(t)
                best_result = np.array([a0, a4, a8])
        except Exception:
            pass

    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return best_result
