"""Tikhonov-regularized polynomial closure via SVD."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Map x to [-1, 1] for Chebyshev-like conditioning
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-12:
        x_min, x_max = x_min - 1.0, x_min + 1.0
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    # Build Vandermonde in normalized x
    V = np.vander(x_norm, 6, increasing=True)

    # Tikhonov regularization: minimize ||V c - y||^2 + lambda * ||c||^2
    # Solution: c = (V^T V + lambda I)^{-1} V^T y
    # Use SVD for stability
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    
    # Try multiple lambda values, select via leave-one-out-like criterion
    best_score = np.inf
    best_coeff_norm = None
    
    # Lambda values spanning orders of magnitude
    lambdas = np.logspace(-6, 3, 40)
    
    for lam in lambdas:
        # Regularized solution via SVD: c = Vt^T U^T y / (s^2 + lambda)
        # But careful: (V^T V + lambda I)^{-1} V^T y = Vt^T diag(1/(s^2+lambda)) U^T y
        y_proj = U.T @ y
        c_norm = Vt.T @ (y_proj / (s**2 + lam))
        
        # Compute prediction error
        pred = V @ c_norm
        resid = y - pred
        mse = np.mean(resid**2)
        
        # Effective degrees of freedom for Tikhonov: sum(s^2/(s^2+lambda))
        dof = np.sum(s**2 / (s**2 + lam))
        
        # AIC-like criterion
        aic = n * np.log(mse + 1e-30) + 2 * dof
        
        if aic < best_score:
            best_score = aic
            best_coeff_norm = c_norm
    
    if best_coeff_norm is None:
        return np.zeros(6)
    
    # Convert from normalized x to original x
    # x_norm = 2*(x - x_min)/(x_max - x_min) - 1
    # So x = (x_norm + 1) * (x_max - x_min)/2 + x_min
    # Let a = (x_max - x_min)/2, b = x_min + a
    # x = a * x_norm + b
    # x_norm = (x - b) / a
    # c_norm[0] + c_norm[1]*x_norm + ... + c_norm[5]*x_norm^5
    # = c_norm[0] + c_norm[1]*(x-b)/a + ... + c_norm[5]*((x-b)/a)^5
    # = sum_{k=0}^{5} [sum_{i=k}^{5} c_norm[i] * C(i,k) * (1/a)^k * (-b)^(i-k)] * x^k
    
    a_scale = (x_max - x_min) / 2.0
    b_offset = x_min + a_scale
    
    coeff = np.zeros(6)
    for k in range(6):
        inner = 0.0
        for i in range(k, 6):
            from math import comb
            inner += best_coeff_norm[i] * comb(i, k) * (1.0 / a_scale)**k * (-b_offset)**(i - k)
        coeff[k] = inner
    
    # Ensure finite
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)
    return coeff
# EVOLVE-BLOCK-END
