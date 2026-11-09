import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    time_limit = 610.0  # Safety margin before 630s kill

    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2*Y

    bin_idx_m1 = (V_m1 + 9).flatten()
    bin_idx_0 = V_0.flatten()
    bin_idx_1 = V_1.flatten()
    bin_idx_2 = V_2.flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def compute_ratio_from_logP(logP_flat, smooth_beta=0.0):
        z = logP_flat - np.max(logP_flat)
        P_flat = np.exp(z)
        sum_P = np.sum(P_flat)
        if sum_P < 1e-300:
            return 0.0
        P_flat = P_flat / sum_P

        counts_m1 = np.bincount(bin_idx_m1, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(bin_idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P_flat, minlength=num_bins_2)

        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))

        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)

        if smooth_beta > 1e-6:
            # Soft-max via log-sum-exp for smoothness
            H_vals = np.array([H_0, H_1, H_2])
            M = np.logaddexp.reduce(H_vals / smooth_beta) * smooth_beta
            # More standard soft-max: beta * log(sum(exp(H_i / beta)))
            # Actually let's use: M = beta * log(sum(exp(H_i / beta)))
            exp_H = np.exp(H_vals / smooth_beta)
            M = smooth_beta * np.log(np.sum(exp_H))
        else:
            M = max(H_0, H_1, H_2)

        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def objective(logP_flat, smooth_beta=0.0):
        r = compute_ratio_from_logP(logP_flat, smooth_beta)
        return -r

    def objective_smooth(logP_flat):
        return objective(logP_flat, smooth_beta=1.0)

    best_score = 0.0
    best_P = np.ones((10, 10)) / 100.0

    # Parametric ansatz: P(x,y) ∝ exp(-αx² - βy² - γxy)
    # Optimize (α, β, γ) to find good structured starting points
    def logP_from_params(params):
        alpha, beta, gamma = params
        x_vals = np.arange(n).reshape(1, -1)  # (1, 10)
        y_vals = np.arange(n).reshape(-1, 1)  # (10, 1)
        logP = -alpha * x_vals**2 - beta * y_vals**2 - gamma * x_vals * y_vals
        return logP.flatten()

    def objective_params(params):
        logP = logP_from_params(params)
        return objective(logP)

    # Grid search over (α, β, γ) to find promising structured distributions
    param_inits = []
    for alpha in np.linspace(-1.5, 1.5, 12):
        for beta in np.linspace(-1.5, 1.5, 12):
            for gamma in np.linspace(-2.0, 2.0, 15):
                param_inits.append((alpha, beta, gamma))

    # First pass: evaluate all parametric points, keep top ones
    scored_params = []
    for params in param_inits:
        if time.time() - start_time > time_limit - 60:
            break
        logP = logP_from_params(params)
        score = -objective(logP)
        scored_params.append((score, params))

    scored_params.sort(key=lambda t: t[0], reverse=True)

    # Direct optimization in 3D parameter space with smooth objective
    if time.time() - start_time < time_limit - 30:
        def obj_params_smooth(params):
            logP = logP_from_params(params)
            return objective_smooth(logP)

        # Start from top parametric grid points
        top_params = [p for _, p in scored_params[:10]]
        for init_p in top_params:
            if time.time() - start_time > time_limit - 30:
                break
            try:
                res = minimize(obj_params_smooth, np.array(init_p), method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                logP_opt = logP_from_params(res.x)
                score = -objective(logP_opt)
                if score > best_score:
                    best_score = score
                    z = logP_opt - np.max(logP_opt)
                    P_flat = np.exp(z)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
            except Exception:
                pass

    # Use top parametric points as initializations for L-BFGS-B
    init_points = []

    # Add uniform
    init_points.append(np.zeros(100))

    # Add top parametric ansatz points
    for score, params in scored_params[:20]:
        init_points.append(logP_from_params(params))

    # Add rank-1 structured points
    for la in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        for lb in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            logP = np.zeros(100)
            for y in range(10):
                for x in range(10):
                    logP[y*10+x] = la*x + lb*y
            init_points.append(logP)

    # Add diagonal concentration
    for c in range(-9, 10):
        logP = np.full(100, -20.0)
        for y in range(10):
            x = y + c
            if 0 <= x < 10:
                logP[y*10+x] = 0.0
        init_points.append(logP)

    # Add random points
    rng = np.random.default_rng(12345)
    for _ in range(20):
        init_points.append(rng.normal(0, 1.5, 100))

    for i, init in enumerate(init_points):
        if time.time() - start_time > time_limit - 10:
            break
        try:
            # Use smooth objective for better gradients
            res = minimize(objective_smooth, init, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            score = -objective(res.x)
            if score > best_score:
                best_score = score
                z = res.x - np.max(res.x)
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # High-precision polish: refine best with tighter tolerances
    if time.time() - start_time < time_limit - 5:
        z = np.log(np.clip(best_P, 1e-300, None)).flatten()
        for trial in range(12):
            if time.time() - start_time > time_limit - 5:
                break
            jitter = rng.normal(0, 0.03, 100)
            try:
                # Use smooth objective for gradient-based polish
                res = minimize(objective_smooth, z + jitter, method='L-BFGS-B',
                               options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-13})
                score = -objective(res.x)
                if score > best_score:
                    best_score = score
                    z2 = res.x - np.max(res.x)
                    P_flat = np.exp(z2)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
                    z = z2
            except Exception:
                pass

        # Final exact-evaluation polish with even tighter tolerance
        if time.time() - start_time < time_limit - 3:
            for trial in range(4):
                if time.time() - start_time > time_limit - 3:
                    break
                jitter = rng.normal(0, 0.01, 100)
                try:
                    res = minimize(objective, z + jitter, method='L-BFGS-B',
                                   options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-14})
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        z2 = res.x - np.max(res.x)
                        P_flat = np.exp(z2)
                        P_flat = P_flat / np.sum(P_flat)
                        best_P = P_flat.reshape(10, 10)
                        z = z2
                except Exception:
                    pass

    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s

    return best_P
