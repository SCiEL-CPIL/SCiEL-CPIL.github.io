"""Robust IRLS polynomial closure with adaptive degree selection."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Standardize x for numerical conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    xs = (x - x_mean) / x_std

    # Standardize y to improve conditioning of the fit
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    ys = (y - y_mean) / y_std

    def fit_irls(xs, ys, degree, n_iter=20, huber_delta=1.345):
        """Fit polynomial of given degree using IRLS with Huber loss."""
        # Build Vandermonde matrix
        V = np.vander(xs, degree + 1, increasing=True)
        # Initialize weights
        w = np.ones(len(xs))
        for _ in range(n_iter):
            W = np.diag(w)
            try:
                # Weighted least squares
                coeff_s, _, _, _ = np.linalg.lstsq(W @ V, W @ ys, rcond=None)
            except np.linalg.LinAlgError:
                coeff_s = np.zeros(degree + 1)
            resid = ys - V @ coeff_s
            # Update weights based on Huber loss
            abs_resid = np.abs(resid)
            w = np.where(abs_resid <= huber_delta, 1.0, huber_delta / abs_resid)
        return coeff_s

    # Try degrees 1 through 5, select best via GCV-like criterion
    best_score = np.inf
    best_coeff = None
    for degree in range(1, 6):
        coeff_s = fit_irls(xs, ys, degree)
        # Compute residuals
        V = np.vander(xs, degree + 1, increasing=True)
        resid = ys - V @ coeff_s
        # Generalized cross-validation score
        # For WLS, use trace of leverage
        try:
            H = V @ np.linalg.pinv(V.T @ V) @ V.T
            trace_H = np.trace(H)
            gcv = np.sum(resid ** 2) / ((1 - trace_H / n) ** 2)
        except np.linalg.LinAlgError:
            gcv = np.sum(resid ** 2)
        # Add small penalty for higher degrees to avoid overfitting
        penalty = 0.01 * degree
        score = gcv + penalty
        if score < best_score:
            best_score = score
            best_coeff = coeff_s

    if best_coeff is None:
        best_coeff = np.zeros(6)

    # Unstandardize the polynomial
    # We fit: ys = sum c_i * xs^i
    # xs = (x - x_mean)/x_std
    # ys = (y - y_mean)/y_std
    # So: (y - y_mean)/y_std = sum c_i * ((x - x_mean)/x_std)^i
    # y = y_mean + y_std * sum c_i * ((x - x_mean)/x_std)^i
    # Expand in powers of x
    # (x - x_mean)/x_std = (1/x_std)*x - (x_mean/x_std)
    # Let a = 1/x_std, b = -x_mean/x_std
    # (a*x + b)^i = sum_{k=0}^i C(i,k) * (a*x)^k * b^(i-k)
    # = sum_{k=0}^i C(i,k) * a^k * b^(i-k) * x^k
    # So y = y_mean + y_std * sum_{i=0}^d c_i * sum_{k=0}^i C(i,k) * a^k * b^(i-k) * x^k
    # = y_mean + y_std * sum_{k=0}^d [sum_{i=k}^d c_i * C(i,k) * a^k * b^(i-k)] * x^k

    a = 1.0 / x_std
    b = -x_mean / x_std
    d = len(best_coeff) - 1
    coeff = np.zeros(6)
    for k in range(6):
        inner = 0.0
        for i in range(k, d + 1):
            if i < len(best_coeff):
                from math import comb
                inner += best_coeff[i] * comb(i, k) * (a ** k) * (b ** (i - k))
        coeff[k] = y_std * inner
    coeff[0] += y_mean

    # Ensure finite
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)
    return coeff
# EVOLVE-BLOCK-END
