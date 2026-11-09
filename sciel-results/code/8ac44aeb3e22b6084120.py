import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    time_limit = 620.0  # Safety margin before 630s kill

    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = (X - Y).flatten()
    V_0 = X.flatten()
    V_1 = (X + Y).flatten()
    V_2 = (X + 2*Y).flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def compute_ratio_from_logP(logP_flat):
        z = logP_flat - np.max(logP_flat)
        P_flat = np.exp(z)
        sum_P = np.sum(P_flat)
        if sum_P < 1e-300:
            return 0.0
        P_flat = P_flat / sum_P

        counts_m1 = np.bincount(V_m1 + 9, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(V_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(V_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(V_2, weights=P_flat, minlength=num_bins_2)

        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))

        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)

        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def objective_params(params):
        alpha, beta, gamma = params
        x_vals = np.arange(n).reshape(1, -1)
        y_vals = np.arange(n).reshape(-1, 1)
        # logP = -alpha x^2 - beta y^2 - gamma xy
        logP = -alpha * x_vals**2 - beta * y_vals**2 - gamma * x_vals * y_vals
        return -compute_ratio_from_logP(logP.flatten())

    best_score = 0.0
    best_P = np.ones((10, 10)) / 100.0
    best_params = np.zeros(3)

    # Multi-start optimization in 3D param space
    rng = np.random.default_rng(42)
    
    # Initial guesses
    inits = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 0.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [2.0, 0.5, 0.0],
        [0.5, 2.0, 0.0],
        [1.0, 1.0, 2.0],
        [1.0, 1.0, -2.0],
    ]
    
    # Add random starts
    for _ in range(20):
        inits.append(rng.normal(0, 1.5, 3))

    for init in inits:
        if time.time() - start_time > time_limit - 60:
            break
        try:
            res = minimize(objective_params, init, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
                # Update best_P
                x_vals = np.arange(n).reshape(1, -1)
                y_vals = np.arange(n).reshape(-1, 1)
                logP = -best_params[0] * x_vals**2 - best_params[1] * y_vals**2 - best_params[2] * x_vals * y_vals
                z = logP.flatten() - np.max(logP.flatten())
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # Refine in full 100D space starting from the best parametric solution
    if time.time() - start_time < time_limit - 30:
        x_vals = np.arange(n).reshape(1, -1)
        y_vals = np.arange(n).reshape(-1, 1)
        logP_init = (-best_params[0] * x_vals**2 - best_params[1] * y_vals**2 - best_params[2] * x_vals * y_vals).flatten()
        
        # Also try starting from uniform
        logP_init_uniform = np.zeros(100)
        
        starts = [logP_init, logP_init_uniform]
        
        # Add jittered starts
        for _ in range(5):
            starts.append(logP_init + rng.normal(0, 0.5, 100))
            
        def objective_full(logP_flat):
            return -compute_ratio_from_logP(logP_flat)

        for init in starts:
            if time.time() - start_time > time_limit - 10:
                break
            try:
                res = minimize(objective_full, init, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                score = -res.fun
                if score > best_score:
                    best_score = score
                    z = res.x - np.max(res.x)
                    P_flat = np.exp(z)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
            except Exception:
                pass

    # Final validity check
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s

    return best_P
