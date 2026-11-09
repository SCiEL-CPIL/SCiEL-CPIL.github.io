"""Rewritten program using a distinct approach: kernel-regularized polynomial basis fit."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the subgrid tendency as a function of x.
    
    Uses a normalized basis (Hermite-like via standardization) to improve conditioning,
    then maps back to the original x-space coefficients. Adds light Tikhonov regularization
    to avoid overfitting in higher-degree terms.
    """
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        # Not enough data; return zeros
        return np.zeros(6)
    
    # Standardize x for stable polynomial fitting
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        sigma = 1.0
    
    x_std = (x - mu) / sigma
    
    # Build Vandermonde matrix in standardized space
    # V[i, j] = x_std[i]^j, for j = 0..5
    V_std = np.vander(x_std, N=6, increasing=True)
    
    # Tikhonov regularization: solve (V^T V + lambda * I) c = V^T y
    # Choose lambda relative to the scale of V^T V
    G = V_std.T @ V_std
    b = V_std.T @ y
    
    # lambda choice: small but non-zero to stabilize
    # Use relative regularization
    trace_G = np.trace(G)
    if trace_G > 0:
        lambda_reg = 1e-3 * trace_G / 6.0
    else:
        lambda_reg = 1e-3
    
    # Regularize only the higher-order terms more strongly (optional)
    # Here we use uniform regularization for simplicity
    A = G + lambda_reg * np.eye(6)
    
    try:
        d = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        d = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Now transform d (coefficients in x_std basis) back to x basis
    # y = sum_{k=0}^{5} d_k * x_std^k = sum_{k=0}^{5} d_k * ((x - mu)/sigma)^k
    # = sum_{k=0}^{5} d_k / sigma^k * sum_{j=0}^{k} binom(k, j) (-mu)^{k-j} x^j
    # So c_j = sum_{k=j}^{5} d_k / sigma^k * binom(k, j) * (-mu)^{k-j}
    
    from math import comb
    c = np.zeros(6)
    for j in range(6):
        c_j = 0.0
        for k in range(j, 6):
            c_j += d[k] / (sigma ** k) * comb(k, j) * ((-mu) ** (k - j))
        c[j] = c_j
    
    # Final sanity check
    if not np.all(np.isfinite(c)):
        c = np.zeros(6)
    
    return c
# EVOLVE-BLOCK-END
