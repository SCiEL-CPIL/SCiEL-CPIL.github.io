"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Fit degree-5 polynomial with regularization to avoid overfitting
    # Use ridge regression (L2 regularization) for stability
    # Build Vandermonde matrix for powers 0..5
    # Use scaled x to improve conditioning
    
    # Check time
    if time.time() - t_start > deadline - 5.0:
        # Fallback: simple linear fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            full = np.zeros(6, dtype=np.float64)
            full[:2] = coeff
            return full
        except:
            return np.zeros(6, dtype=np.float64)

    # Scale x to [-1, 1] range for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        # All x values are the same; just return mean
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)
    
    x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)

    # Try multiple regularization strengths and pick the best via cross-validation
    # But keep it simple: use a reasonable default regularization
    
    # Build design matrix for degree 5
    V = np.vander(x_scaled, N=6, increasing=True)  # columns: 1, x, x^2, ..., x^5
    
    # Ridge regression with different lambda values
    best_coeff = np.zeros(6, dtype=np.float64)
    best_mse = np.inf
    
    # Try a range of regularization strengths
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    for lam in lambdas:
        if time.time() - t_start > deadline - 5.0:
            break
        try:
            # Solve (V^T V + lam*I) c = V^T y
            VtV = V.T @ V
            Vty = V.T @ y
            A = VtV + lam * np.eye(6)
            coeff_scaled = np.linalg.solve(A, Vty)
            
            # Compute training MSE (not ideal for model selection, but fast)
            # Better: use a holdout validation
            # For now, use training fit with regularization as proxy
            pred = V @ coeff_scaled
            mse = np.mean((pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_coeff = coeff_scaled
        except:
            continue
    
    # Convert coefficients back to original x scale
    # We need to express p(x_scaled) in terms of x
    # x_scaled = (x - c) / s where c = (x_min+x_max)/2, s = x_range/2
    c = (x_min + x_max) / 2.0
    s = x_range / 2.0
    
    # Convert polynomial from x_scaled to x
    # p(x_scaled) = sum_{i=0}^{5} a_i * x_scaled^i
    # x_scaled = (x - c) / s
    # So p = sum a_i * ((x-c)/s)^i = sum a_i * s^{-i} * (x-c)^i
    # Expand (x-c)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-c)^{i-j}
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    for i in range(6):
        a_i = best_coeff[i]
        if abs(a_i) < 1e-15:
            continue
        # Expand (x - c)^i / s^i
        # (x - c)^i = sum_{j=0}^{i} C(i,j) * x^j * (-c)^(i-j)
        for j in range(i + 1):
            from math import comb
            binom_coeff = comb(i, j)
            power_c = (-c) ** (i - j)
            coeff_orig[j] += a_i * (binom_coeff * power_c) / (s ** i)
    
    # Verify finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback to simple fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 3)
            full = np.zeros(6, dtype=np.float64)
            full[:4] = coeff
            return full
        except:
            return np.zeros(6, dtype=np.float64)
    
    return coeff_orig
# EVOLVE-BLOCK-END
