"""Rewritten program using a distinct approach: kernel-regularized polynomial basis fit."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the subgrid tendency as a function of x.
    
    Uses a normalized basis (Hermite-like via standardization) to improve conditioning,
    then maps back to the original x-space coefficients. Uses SVD-based truncated least squares
    for robust regularization against overfitting and ill-conditioning.
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
    if sigma < 1e-12:
        sigma = 1.0
    
    x_std = (x - mu) / sigma
    
    # Build Vandermonde matrix in standardized space
    # V[i, j] = x_std[i]^j, for j = 0..5
    V_std = np.vander(x_std, N=6, increasing=True)
    
    # Use SVD for robust solving with regularization
    # V = U * S * Vt
    U, S, Vt = np.linalg.svd(V_std, full_matrices=False)
    
    # Determine cutoff for singular values
    # We want to regularize effectively. 
    # A common heuristic is to cut off singular values that are small compared to the max.
    # Given the task requires generalization to unseen regimes, we should be conservative.
    # Let's use a relative cutoff. 
    # If we keep too many, we overfit. If we keep too few, we underfit.
    # Degree 5 is fixed, so we can't drop degrees entirely, but we can shrink coefficients 
    # associated with small singular values.
    
    # Instead of hard truncation, we use Tikhonov in the SVD space (equivalent to Ridge Regression)
    # Minimize ||V d - y||^2 + lambda ||d||^2
    # Solution: d = Vt^T (S^2 + lambda I)^-1 U^T y
    
    # Choose lambda. 
    # The scale of S^2 is roughly related to the variance of the data in the basis.
    # Let's pick lambda such that it adds a small penalty to the norm of d.
    # trace(S^2) = ||V||_F^2
    # Let's try a slightly larger lambda than before to encourage smoothness.
    # Previous: 1e-6 * trace_G / 6. 
    # Let's try 1e-3 * max(S^2) or similar relative measure.
    
    if len(S) == 0:
        return np.zeros(6)
        
    max_S_sq = S[0]**2 if S[0] > 0 else 1.0
    # Lambda choice: 
    # We want to balance fit and smoothness. 
    # A value of 1e-2 to 1e-4 relative to max_S_sq is often a good starting point for polynomial fits.
    # Let's use 1e-3 * max_S_sq.
    lambda_reg = 1e-3 * max_S_sq
    
    # Compute d = Vt.T @ inv(S^2 + lambda I) @ U.T @ y
    # Efficiently:
    Uty = U.T @ y
    S2 = S**2
    denom = S2 + lambda_reg
    
    # Avoid division by zero if S is 0 (shouldn't happen for Vandermonde with distinct x, but safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_svd = (Vt.T * (Uty / denom))
        
    # The formula d = Vt.T * ( (U.T y) / (S^2 + lambda) ) is correct for diagonal S.
    # Let's verify:
    # V = U S Vt
    # (V Vt + lambda I) d = V y  -- NO, this is for left regularization.
    # Standard Ridge: (V^T V + lambda I) d = V^T y
    # V^T V = Vt^T S^2 Vt
    # (Vt^T S^2 Vt + lambda I) d = Vt^T U^T y
    # Multiply by Vt: (S^2 + lambda I) Vt d = U^T y
    # Let w = Vt d. Then (S^2 + lambda I) w = U^T y
    # w = (U^T y) / (S^2 + lambda)
    # d = Vt^T w
    
    w = Uty / denom
    d = Vt.T @ w
    
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
