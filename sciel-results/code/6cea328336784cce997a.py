"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fit_regularized_poly(x, y, degree=5, lam=1e-8):
    """
    Fit a degree-`degree` polynomial to (x, y) pairs using Tikhonov-regularized
    least squares in a centered/scaled basis, then convert back to monomial basis.
    
    Returns coefficients in ascending powers of the original x.
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_scaled = x - x_mean
    else:
        x_scaled = (x - x_mean) / x_std
    
    # Build Vandermonde matrix in scaled basis: V[i,j] = x_scaled[i]^j
    V = np.vander(x_scaled, N=degree + 1, increasing=True)
    
    # Tikhonov regularization: minimize ||V @ c - y||^2 + lam * ||c||^2
    # (Regularize higher-degree coefficients more heavily to avoid overfitting)
    # Use diagonal regularization weighted by degree
    reg_weights = np.array([1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])[:degree + 1]
    R = np.diag(lam * reg_weights)
    
    # Solve (V^T V + R) c = V^T y
    try:
        A = V.T @ V + R
        b = V.T @ y
        c_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: use polyfit with lower degree
        try:
            c_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, min(degree, 3))
            c_scaled = np.pad(c_scaled, (0, degree + 1 - len(c_scaled)))
        except Exception:
            c_scaled = np.zeros(degree + 1)
    
    # Convert from scaled basis to original monomial basis
    # y = sum_j c_scaled[j] * ((x - x_mean)/x_std)^j
    # We need coefficients in powers of x
    # Use polynomial composition: for each j, expand ((x - m)/s)^j
    coeff_original = np.zeros(degree + 1)
    for j in range(degree + 1):
        if abs(c_scaled[j]) < 1e-15:
            continue
        # Expand ((x - m)/s)^j = (1/s^j) * (x - m)^j
        # (x - m)^j = sum_{k=0}^{j} binom(j,k) * x^k * (-m)^(j-k)
        scale_factor = c_scaled[j] / (x_std ** j) if x_std > 1e-12 else c_scaled[j]
        for k in range(j + 1):
            from math import comb
            coeff_original[k] += scale_factor * comb(j, k) * ((-x_mean) ** (j - k))
    
    return coeff_original


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values
    mask = np.isfinite(training_x) & np.isfinite(training_subgrid_tendency)
    x = training_x[mask]
    y = training_subgrid_tendency[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Try degree 5 with regularization
    try:
        coeff = _fit_regularized_poly(x, y, degree=5, lam=1e-8)
    except Exception:
        # Fallback to degree 1
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff = np.zeros(6)
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
    
    return coeff
# EVOLVE-BLOCK-END
