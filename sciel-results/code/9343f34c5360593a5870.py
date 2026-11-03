import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    t_start = time.time()
    deadline = 200.0  # 10s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint P(0)=0
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

    signs = np.array([(-1.0) ** m for m in range(7)])

    def get_P(params):
        a0, a4, a8 = params
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

    def objective_product(params):
        P = get_P(params)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        
        if norm_sq <= 0 or norm_sq_hat <= 0:
            return 1e30
        return norm_sq * norm_sq_hat

    def objective_log(params):
        P = get_P(params)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        
        if norm_sq <= 0 or norm_sq_hat <= 0:
            return 1e30
        return np.log(norm_sq) + np.log(norm_sq_hat)

    # Seed point
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_score = objective_product(x0)
    
    best_result = x0.copy()
    best_score = seed_score

    # Phase 1: Coarse grid search to find promising regions
    if time.time() < deadline - 120:
        # Grid around the seed point with wider range
        a0_range = np.linspace(-3.0, 1.0, 15)
        a4_range = np.linspace(-0.2, 0.3, 15)
        a8_range = np.linspace(-0.02, 0.02, 15)
        
        grid_best = best_score
        grid_best_x = best_result.copy()
        
        for a0 in a0_range:
            if time.time() > deadline - 120:
                break
            for a4 in a4_range:
                if time.time() > deadline - 120:
                    break
                for a8 in a8_range:
                    params = np.array([a0, a4, a8])
                    score = objective_product(params)
                    if score < grid_best:
                        grid_best = score
                        grid_best_x = params.copy()
        
        if grid_best < best_score:
            best_score = grid_best
            best_result = grid_best_x

    # Phase 2: Differential Evolution with bounds informed by grid search
    if time.time() < deadline - 80:
        # Use tighter bounds based on where we found good solutions
        center = best_result
        bounds = [
            (center[0] - 2.0, center[0] + 2.0),
            (center[1] - 0.3, center[1] + 0.3),
            (center[2] - 0.03, center[2] + 0.03)
        ]
        
        try:
            de_result = differential_evolution(
                objective_product, 
                bounds, 
                seed=123, 
                maxiter=500, 
                tol=1e-12,
                polish=False,
                popsize=20,
                update_interval=100,
                workers=1
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-start local optimization with L-BFGS-B
    start_points = [
        best_result,
        x0,
        np.array([-1.13, 0.04, 0.00030864]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-1.0, 0.03, 0.0002]),
        np.array([-1.3, 0.06, 0.0005]),
        np.array([-0.9, 0.02, 0.0001]),
        np.array([-1.15, 0.045, 0.00035]),
        np.array([-1.1, 0.035, 0.00025]),
        np.array([-1.25, 0.055, 0.00045]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            # L-BFGS-B with numerical gradient
            result = minimize(
                objective_product,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14,
                         'maxls': 200}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            continue
        
        # Also try Nelder-Mead from the same start
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                objective_product,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-17,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            continue

    # Phase 4: High-precision polish with BFGS on log-objective
    if best_result is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                objective_log,
                best_result,
                method='BFGS',
                options={'maxiter': 10000, 'gtol': 1e-15, 'disp': False}
            )
            new_score = objective_product(result.x)
            if new_score < best_score:
                best_score = new_score
                best_result = result.x.copy()
        except Exception:
            pass

    # Phase 5: Final ultra-precise Nelder-Mead
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective_product,
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

    # Phase 6: Additional DE with very tight bounds around best
    if time.time() < deadline - 10:
        try:
            tight_bounds = [
                (best_result[0] - 0.1, best_result[0] + 0.1),
                (best_result[1] - 0.01, best_result[1] + 0.01),
                (best_result[2] - 0.001, best_result[2] + 0.001)
            ]
            de_result = differential_evolution(
                objective_product, 
                tight_bounds, 
                seed=456, 
                maxiter=300, 
                tol=1e-14,
                polish=True,
                popsize=15,
                workers=1
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                objective_product,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-19, 'fatol': 1e-21,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    return best_result
