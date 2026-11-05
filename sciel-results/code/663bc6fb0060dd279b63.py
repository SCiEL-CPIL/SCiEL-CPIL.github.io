"""Rewritten program for unresolved-tendency closure task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import lstsq


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses regularized least-squares with cross-validation for lambda selection.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        # Not enough data for degree-5 fit; fall back to zero
        return np.zeros(6)

    # Center and scale x to improve conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Build Vandermonde matrix for polynomial basis [1, x, x^2, x^3, x^4, x^5]
    V = np.vander(x_norm, N=6, increasing=True)  # columns: x^0, x^1, ..., x^5

    # If we have enough data, use a held-out subset for model selection
    use_cv = n > 100 and (time.time() - start_time) < (DEADLINE - 30.0)
    
    if use_cv:
        # Split into train and validation subsets
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        val_size = max(10, n // 10)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        x_train = x_norm[train_idx]
        y_train = y[train_idx]
        x_val = x_norm[val_idx]
        y_val = y[val_idx]
        
        V_train = np.vander(x_train, N=6, increasing=True)
        V_val = np.vander(x_val, N=6, increasing=True)
    else:
        x_train = x_norm
        y_train = y
        V_train = V
        x_val = x_norm
        y_val = y
        V_val = V

    # Try several regularization strengths
    # Use log-spaced lambdas from very small to large
    lambdas = np.logspace(-12, 2, 15)
    
    best_coeff = None
    best_val_mse = np.inf
    best_train_coeff = None
    best_train_mse = np.inf

    for lam in lambdas:
        if time.time() > DEADLINE:
            break

        try:
            # Regularized least squares: solve (V^T V + lambda*I) c = V^T y
            A = V_train.T @ V_train + lam * np.eye(6)
            b = V_train.T @ y_train
            coeff_norm = np.linalg.solve(A, b)

            # Convert back to original x scale
            # We fit in normalized space: y = sum c_norm_i * ((x-x_mean)/x_std)^i
            # We need: y = sum c_i * x^i
            # Use Taylor expansion to convert from centered/scaled basis to raw x basis
            # This is complex, so instead we'll fit directly in raw x space with scaling
            
            # Actually, let's refit in raw x space with proper scaling
            # Better approach: fit in normalized space, then convert coefficients
            # For polynomial P(x) = sum a_i * ((x-x_mean)/x_std)^i
            # We want Q(x) = sum c_i * x^i such that Q(x) = P(x)
            # This requires expanding ((x-x_mean)/x_std)^i in terms of x^j
            
            # Simpler: just fit directly on raw x with good conditioning
            # Use the fact that we can compute the polynomial at any point
            # For now, let's just return the normalized coefficients and 
            # the evaluator will handle it... but that's wrong.
            
            # Let's do it properly: convert using polynomial arithmetic
            # P(x) = sum_{i=0}^{5} a_i * ((x - x_mean)/x_std)^i
            # Let u = (x - x_mean)/x_std, so x = u*x_std + x_mean
            # P(x) = sum a_i * u^i where u = (x - x_mean)/x_std
            # We need to express this as sum c_j * x^j
            
            # Use numpy polynomial conversion
            # First, build the polynomial in u: a_0 + a_1*u + ... + a_5*u^5
            # Then substitute u = (x - x_mean)/x_std
            # This is equivalent to: poly = sum a_i * ((x-x_mean)/x_std)^i
            # = sum a_i * (1/x_std^i) * (x-x_mean)^i
            # = sum a_i * (1/x_std^i) * sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^(i-j)
            
            coeff = np.zeros(6)
            for i in range(6):
                a_i = coeff_norm[i] / (x_std ** i)
                # Expand (x - x_mean)^i = sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^(i-j)
                for j in range(i + 1):
                    from math import comb
                    coeff[j] += a_i * comb(i, j) * ((-x_mean) ** (i - j))

            # Check if coefficients are finite
            if not np.all(np.isfinite(coeff)):
                continue

            # Compute validation MSE (or training MSE if no CV)
            pred_val = np.zeros_like(y_val)
            for i in range(6):
                pred_val += coeff[i] * (x_val ** i)
            val_mse = np.mean((pred_val - y_val) ** 2)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_coeff = coeff.copy()

            # Also track best training fit
            pred_train = np.zeros_like(y_train)
            for i in range(6):
                pred_train += coeff[i] * (x_train ** i)
            train_mse = np.mean((pred_train - y_train) ** 2)
            if train_mse < best_train_mse:
                best_train_mse = train_mse
                best_train_coeff = coeff.copy()

        except np.linalg.LinAlgError:
            continue

    if best_coeff is not None:
        return best_coeff

    # Fallback: use the best training fit
    if best_train_coeff is not None:
        return best_train_coeff

    # Final fallback: simple linear fit
    try:
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    except Exception:
        return np.zeros(6)
# EVOLVE-BLOCK-END
