import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    time_limit = 610.0

    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = (X - Y).flatten()
    V_0 = X.flatten()
    V_1 = (X + Y).flatten()
    V_2 = (X + 2*Y).flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    # Precompute bin indices
    bin_idx_m1 = (V_m1 + 9).astype(int)
    bin_idx_0 = V_0.astype(int)
    bin_idx_1 = V_1.astype(int)
    bin_idx_2 = V_2.astype(int)

    def compute_ratio(P_flat):
        """Compute D/M ratio from probability vector."""
        P = P_flat / P_flat.sum()
        
        counts_m1 = np.bincount(bin_idx_m1, weights=P, minlength=num_bins_m1)
        counts_0 = np.bincount(bin_idx_0, weights=P, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P, minlength=num_bins_2)

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

    def compute_ratio_smooth(P_flat, beta=1.0):
        """Smooth version using log-sum-exp for max."""
        P = P_flat / P_flat.sum()
        
        counts_m1 = np.bincount(bin_idx_m1, weights=P, minlength=num_bins_m1)
        counts_0 = np.bincount(bin_idx_0, weights=P, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P, minlength=num_bins_2)

        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))

        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)
        
        H_vals = np.array([H_0, H_1, H_2])
        max_H = np.max(H_vals)
        if max_H < 1e-9:
            return 0.0
        # Soft max via log-sum-exp
        M = beta * np.log(np.sum(np.exp((H_vals - max_H) / beta))) + max_H
        
        if M < 1e-9:
            return 0.0
        return H_m1 / M

    best_score = 0.0
    best_P = np.ones((10, 10)) / 100.0

    # Strategy 1: Analytical candidate - uniform on full grid
    P_uniform = np.ones(100)
    score = compute_ratio(P_uniform)
    if score > best_score:
        best_score = score
        best_P = P_uniform.reshape(10, 10)

    # Strategy 2: Try to maximize H(x-y) while minimizing M
    # Key insight: concentrate on patterns where x-y is spread out but x, x+y, x+2y are concentrated
    
    # Candidate: uniform on a "diagonal band" that maximizes x-y spread
    # Try various structured distributions
    
    rng = np.random.default_rng(42)
    
    # Parametric family: log P(x,y) = a*x + b*y + c*x*y + d*(x-y)^2 + e
    def logP_param(params):
        a, b, c, d = params
        x_vals = np.arange(n).reshape(1, -1).astype(float)
        y_vals = np.arange(n).reshape(-1, 1).astype(float)
        logP = a * x_vals + b * y_vals + c * x_vals * y_vals + d * (x_vals - y_vals)**2
        return logP.flatten()

    def objective_param(params, use_smooth=False):
        logP = logP_param(params)
        z = logP - np.max(logP)
        P = np.exp(z)
        P = P / P.sum()
        if use_smooth:
            return -compute_ratio_smooth(P, beta=0.5)
        return -compute_ratio(P)

    # Grid search over parametric family
    param_grid = []
    for a in np.linspace(-2, 2, 8):
        for b in np.linspace(-2, 2, 8):
            for c in np.linspace(-1, 1, 6):
                for d in np.linspace(-1, 1, 6):
                    param_grid.append([a, b, c, d])
    
    for params in param_grid:
        if time.time() - start_time > time_limit - 100:
            break
        score_p = -objective_param(params, use_smooth=False)
        if score_p > best_score:
            best_score = score_p
            logP = logP_param(params)
            z = logP - np.max(logP)
            P = np.exp(z)
            P = P / P.sum()
            best_P = P.reshape(10, 10)

    # Strategy 3: Direct optimization in 100-dim space with multiple initializations
    init_points = []
    
    # Uniform
    init_points.append(np.zeros(100))
    
    # From best parametric
    if best_score > 0:
        init_points.append(np.log(np.clip(best_P, 1e-300, None)).flatten())
    
    # Concentrated on anti-diagonals (maximize x-y spread)
    for c_offset in range(-9, 10):
        logP = np.full(100, -20.0)
        for y in range(10):
            x = y + c_offset
            if 0 <= x < 10:
                logP[y*10+x] = 0.0
        init_points.append(logP)
    
    # Random perturbations
    for _ in range(20):
        init_points.append(rng.normal(0, 2.0, 100))
    
    # Structured: high mass on corners to spread x-y
    corner_logP = np.full(100, -10.0)
    for x, y in [(0,0), (9,9), (0,9), (9,0)]:
        corner_logP[y*10+x] = 0.0
    init_points.append(corner_logP)
    
    # Try to minimize H(x+2y) by concentrating on few x+2y values
    # while spreading x-y
    for target_sum2 in range(0, 28, 3):
        logP = np.full(100, -20.0)
        for y in range(10):
            for x in range(10):
                if x + 2*y == target_sum2:
                    logP[y*10+x] = 0.0
        init_points.append(logP)

    # L-BFGS-B optimization from multiple starts
    for i, init in enumerate(init_points):
        if time.time() - start_time > time_limit - 30:
            break
        try:
            # First pass with smooth objective
            res = minimize(lambda lp: -compute_ratio_smooth(np.exp(lp - np.max(lp)) / np.sum(np.exp(lp - np.max(lp))), beta=0.5),
                          init, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            score_s = -res.fun
            if score_s > best_score:
                z = res.x - np.max(res.x)
                P = np.exp(z)
                P = P / P.sum()
                best_score = compute_ratio(P)
                best_P = P.reshape(10, 10)
        except Exception:
            pass

    # Strategy 4: High-precision polish with exact objective
    z_best = np.log(np.clip(best_P, 1e-300, None)).flatten()
    
    for trial in range(15):
        if time.time() - start_time > time_limit - 10:
            break
        jitter = rng.normal(0, 0.05, 100)
        try:
            res = minimize(lambda lp: -compute_ratio(np.exp(lp - np.max(lp)) / np.sum(np.exp(lp - np.max(lp)))),
                          z_best + jitter, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-13})
            score = -res.fun
            if score > best_score:
                best_score = score
                z2 = res.x - np.max(res.x)
                P = np.exp(z2)
                P = P / P.sum()
                best_P = P.reshape(10, 10)
                z_best = z2
        except Exception:
            pass

    # Strategy 5: Coordinate descent on binary cell selection
    # Try to find a subset of cells that maximizes the ratio
    if time.time() - start_time < time_limit - 20:
        # Start from best_P and try flipping cells
        P_work = best_P.flatten().copy()
        improved = True
        while improved and time.time() - start_time < time_limit - 15:
            improved = False
            for idx in range(100):
                # Try zeroing this cell
                P_test = P_work.copy()
                P_test[idx] = 0
                if P_test.sum() > 1e-9:
                    score_test = compute_ratio(P_test)
                    if score_test > best_score + 1e-10:
                        best_score = score_test
                        P_work = P_test
                        improved = True
                # Try increasing this cell
                P_test = P_work.copy()
                P_test[idx] = max(P_work[idx], 1e-10) * 2
                score_test = compute_ratio(P_test)
                if score_test > best_score + 1e-10:
                    best_score = score_test
                    P_work = P_test
                    improved = True
            if improved:
                best_P = P_work.reshape(10, 10)

    # Final normalization and validation
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s

    return best_P
