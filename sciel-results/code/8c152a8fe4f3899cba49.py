"""Convex optimization approach for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def construct():
    n = N
    
    # Use a structured initialization: alternating 0s and 1s with slight perturbation
    # to break symmetry and help the optimizer
    h_init = np.zeros(n)
    h_init[::2] = 1.0
    h_init[1::2] = 0.0
    # Adjust to satisfy sum constraint: currently sum = n/2 if n even, which is exactly what we need
    # But add small random perturbation to break exact symmetry
    np.random.seed(42)
    h_init = h_init + 0.01 * np.random.randn(n)
    h_init = np.clip(h_init, 0.0, 1.0)
    # Project to exact sum
    diff = (n / 2.0 - np.sum(h_init)) / n
    h_init = h_init + diff
    h_init = np.clip(h_init, 0.0, 1.0)
    # Final projection to ensure sum is exact
    diff = (n / 2.0 - np.sum(h_init)) / n
    h_init = h_init + diff
    h_init = np.clip(h_init, 0.0, 1.0)
    
    # Initial t should be an upper bound on the max correlation
    lags_init = np.correlate(h_init, 1 - h_init, mode="full")
    t_init = np.max(lags_init) + 0.05
    
    x0 = np.concatenate([np.array([t_init]), h_init])
    
    bounds = [(0, None)] + [(0, 1)] * n
    
    def objective(x):
        return x[0]
    
    def objective_grad(x):
        grad = np.zeros(n + 1)
        grad[0] = 1.0
        return grad
    
    # Sum constraint: sum(h) = n/2
    sum_A = np.zeros((1, n + 1))
    sum_A[0, 1:] = 1.0
    sum_constraint = LinearConstraint(sum_A, n/2.0, n/2.0)
    
    # Correlation constraints: t - C_k >= 0 for all k
    def corr_fun(x):
        t = x[0]
        h = x[1:]
        lags = np.correlate(h, 1 - h, mode="full")
        return t - lags
    
    def corr_jac(x):
        t = x[0]
        h = x[1:]
        n_h = len(h)
        num_lags = 2 * n_h - 1
        jac = np.zeros((num_lags, n + 1))
        # Gradient of t - C_k with respect to x[0] is 1
        jac[:, 0] = 1.0
        # Gradient of C_k with respect to h[j]
        for lag_idx in range(num_lags):
            lag = lag_idx - (n_h - 1)
            # C_lag = sum_i h[i] * (1 - h[i+lag]) for valid i
            # dC_lag/dh[j] = sum over i where j=i: (1 - h[i+lag]) + sum over i where j=i+lag: -h[i]
            # For each j, find all i such that i = j (contribution: 1 - h[j+lag]) or j = i+lag (contribution: -h[j-lag])
            for j in range(n_h):
                # Case 1: i = j, need 0 <= j+lag < n_h
                if 0 <= j + lag < n_h:
                    jac[lag_idx, 1 + j] += -(1.0 - h[j + lag])
                # Case 2: j = i + lag => i = j - lag, need 0 <= j-lag < n_h
                if 0 <= j - lag < n_h:
                    jac[lag_idx, 1 + j] += -h[j - lag]
        return jac
    
    corr_constraint = NonlinearConstraint(corr_fun, 0, np.inf, jac=corr_jac)
    
    constraints = [sum_constraint, corr_constraint]
    
    try:
        result = minimize(
            objective,
            x0,
            method='trust-constr',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'gtol': 1e-6, 'xtol': 1e-8}
        )
        
        h = result.x[1:]
        # Post-processing to ensure strict validity
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        if abs(current_sum - n / 2.0) > 1e-9 * n:
            diff = (n / 2.0 - current_sum) / n
            h = h + diff
            h = np.clip(h, 0.0, 1.0)
            # Second correction
            diff = (n / 2.0 - np.sum(h)) / n
            h = h + diff
            h = np.clip(h, 0.0, 1.0)
        
        # Final check
        if not np.all(np.isfinite(h)):
            return np.full(n, 0.5)
        if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
            return np.full(n, 0.5)
        if abs(np.sum(h) - n / 2.0) > 1e-9 * n:
            return np.full(n, 0.5)
            
        return h
    except Exception:
        # Fallback: use the structured initialization
        h = h_init
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = (n / 2.0 - current_sum) / n
        h = h + diff
        h = np.clip(h, 0.0, 1.0)
        return h
# EVOLVE-BLOCK-END
