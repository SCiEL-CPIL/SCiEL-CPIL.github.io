"""Rewritten program: fit a shared degree-five polynomial to unresolved tendency."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 145.0  # safety margin before 150s wall-clock kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _check_deadline():
    return _elapsed() > DEADLINE


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Returns 6 coefficients in ASCENDING powers of x.
    """
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = training_x.size
    if n == 0:
        return np.zeros(6)
    
    # Ensure at least as many points as we need for degree 5
    if n < 6:
        # Fall back to least degree possible
        deg = max(1, n - 1)
        deg = min(deg, 5)
        coeff = np.zeros(6)
        try:
            c = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg)
            coeff[:deg+1] = c
        except Exception:
            coeff[0] = np.mean(training_subgrid_tendency)
        return coeff
    
    # Strategy: try full degree-5 fit with various regularization levels
    # The key insight: we want to minimize validation MSE, but we only have
    # training data. A well-regularized polynomial should generalize better.
    
    # First, normalize x to improve numerical conditioning
    x_mean = np.mean(training_x)
    x_std = np.std(training_x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_norm = (training_x - x_mean) / x_std
    
    # Build Vandermonde matrix for normalized x
    # We'll fit in normalized space, then transform back
    V = np.polynomial.polynomial.polyvander(x_norm, 5)
    
    # Try multiple regularization strengths (Tikhonov)
    # alpha values: from 0 (pure least squares) to large (nearly constant)
    alphas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    best_coeff = None
    best_score = np.inf
    
    for alpha in alphas:
        if _check_deadline():
            break
        
        try:
            if alpha == 0.0:
                # Pure least squares
                c_norm, residuals, rank, sv = np.linalg.lstsq(V, training_subgrid_tendency, rcond=None)
            else:
                # Tikhonov regularization
                A = V.T @ V + alpha * np.eye(6)
                b = V.T @ training_subgrid_tendency
                c_norm = np.linalg.solve(A, b)
            
            # Transform coefficients from normalized space to original space
            # If y = sum(c_i * (x - mean)/std)^i, expand in powers of x
            # (x - mean)/std = x/std - mean/std
            # Let u = x/std + (-mean/std), so u = a*x + b where a=1/std, b=-mean/std
            # y = sum(c_i * (a*x + b)^i)
            # We need to convert c_norm (coefficients in u) to coefficients in x
            
            # Use polynomial composition: p(u) where u = a*x + b
            # p(x) = p(a*x + b)
            # This is a linear transformation on coefficients
            
            a_val = 1.0 / x_std
            b_val = -x_mean / x_std
            
            # Compute the transformation matrix
            # For each power i, (a*x + b)^i = sum_{j=0}^{i} C(i,j) * a^j * b^(i-j) * x^j
            T = np.zeros((6, 6))
            for i in range(6):
                for j in range(i + 1):
                    from math import comb
                    T[j, i] = comb(i, j) * (a_val ** j) * (b_val ** (i - j))
            
            c_original = T @ c_norm
            
            # Compute training MSE as a proxy
            pred = np.polynomial.polynomial.polyval(training_x, c_original)
            mse = np.mean((pred - training_subgrid_tendency) ** 2)
            
            # Use a simple bias-variance proxy: prefer lower training error
            # but penalize high-degree coefficients (overfitting indicator)
            coeff_penalty = np.sum(np.abs(c_original[3:]) ** 2) * 1e-3
            score = mse + coeff_penalty
            
            if score < best_score:
                best_score = score
                best_coeff = c_original.copy()
                
        except Exception:
            continue
    
    if best_coeff is None:
        # Fallback: simple linear fit
        try:
            c = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c
        except Exception:
            best_coeff = np.zeros(6)
            best_coeff[0] = np.mean(training_subgrid_tendency)
    
    # Final check: ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)
        best_coeff[0] = np.mean(training_subgrid_tendency)
    
    return best_coeff
# EVOLVE-BLOCK-END
