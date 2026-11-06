"""Rewritten program: robust degree-5 polynomial closure with ridge + bandwidth-tuned smoothing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 145.0  # seconds wall-clock safety margin


def _fit_poly_ridge(x, y, degree, lam):
    """Fit polynomial via ridge regression in monomial basis."""
    n = len(x)
    # Build Vandermonde
    V = np.vander(x, N=degree + 1, increasing=True)
    # Ridge: (V^T V + lam*I) c = V^T y
    # Add small regularization to identity for numerical stability
    I = np.eye(degree + 1)
    A = V.T @ V + lam * I
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinalgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def _nadaraya_watson(x, y, x_eval, bandwidth):
    """Nadaraya-Watson kernel smoother with Gaussian kernel."""
    # x, y: 1D arrays; x_eval: 1D array
    # Returns smoothed y at x_eval
    diff = (x_eval[:, None] - x[None, :]) / bandwidth
    weights = np.exp(-0.5 * diff ** 2)
    # Avoid division by zero
    denom = weights.sum(axis=1)
    denom[denom < 1e-15] = 1e-15
    smoothed = (weights @ y) / denom
    return smoothed


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)

    # Basic stats
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        # Degenerate case: constant x
        return np.array([np.mean(y)] + [0.0] * 5)

    # Normalize x for better conditioning
    x_norm = (x - x_mean) / x_std

    degree = 5

    # Strategy:
    # 1. Fit a full degree-5 polynomial with small ridge on normalized x.
    # 2. Optionally smooth y with Nadaraya-Watson and refit.
    # 3. Select best via cross-validation on training data.
    # 4. Convert back to original x basis.

    best_coeff = None
    best_cv_score = np.inf

    # Candidate 1: Direct ridge fit, degree 5
    for lam in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
        c_norm = _fit_poly_ridge(x_norm, y, degree, lam)
        # Convert from normalized to original basis
        # c_orig = transform
        # We'll do this via Vandermonde in original basis
        V_orig = np.vander(x, N=degree + 1, increasing=True)
        # Predict using normalized coeff:
        # y_pred = sum c_norm[i] * ((x - x_mean)/x_std)^i
        # We need c_orig such that sum c_orig[i] * x^i = sum c_norm[i] * ((x-x_mean)/x_std)^i
        # This is a change of variables; easiest: fit in original basis using the predictions
        # Actually, let's just fit directly in original basis with ridge.
        pass

    # Simpler approach: fit directly in original x basis with ridge
    # But x may be large; let's use a scaled approach.
    # Alternative: fit in normalized basis, then convert.

    # Conversion from normalized to monomial basis:
    # ((x - mu)/sigma)^k = sum_{j=0}^{k} binom(k,j) (-mu/sigma)^(k-j) x^j / sigma^j... 
    # Actually: ((x-mu)/sigma)^k = (1/sigma^k) * (x-mu)^k = (1/sigma^k) * sum_{j=0}^k binom(k,j) x^j (-mu)^(k-j)
    # So c_orig[j] = sum_{k=j}^{degree} c_norm[k] * binom(k,j) * (-mu)^(k-j) / sigma^k

    def convert_norm_to_orig(c_norm, mu, sigma, deg):
        c_orig = np.zeros(deg + 1)
        for k in range(deg + 1):
            for j in range(k + 1):
                from math import comb
                c_orig[j] += c_norm[k] * comb(k, j) * ((-mu) ** (k - j)) / (sigma ** k)
        return c_orig

    # CV: simple k-fold on training data
    def cross_val_score(c_norm, mu, sigma, deg, n_folds=5):
        n = len(x_norm)
        indices = np.arange(n)
        np.random.seed(42)
        indices = np.random.permutation(indices)
        fold_size = n // n_folds
        total_mse = 0.0
        for i in range(n_folds):
            val_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            # Refit on train
            c_train = _fit_poly_ridge(x_norm[train_idx], y[train_idx], deg, 1e-4)
            # Predict on val
            V_val = np.vander(x_norm[val_idx], N=deg + 1, increasing=True)
            pred = V_val @ c_train
            mse = np.mean((pred - y[val_idx]) ** 2)
            total_mse += mse
        return total_mse / n_folds

    # Try several ridge values and smoothing levels
    candidates = []

    # Direct fits with various ridge
    for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        if time.time() > deadline - 10:
            break
        c_norm = _fit_poly_ridge(x_norm, y, degree, lam)
        cv = cross_val_score(c_norm, x_mean, x_std, degree)
        candidates.append((cv, c_norm, 'ridge', lam))

    # Nadaraya-Watson smoothed y, then fit polynomial
    # Try several bandwidths
    silverman_bw = 1.06 * np.std(y) * n ** (-1.0 / 5.0)
    bw_grid = [0.1 * silverman_bw, 0.25 * silverman_bw, 0.5 * silverman_bw,
               silverman_bw, 2.0 * silverman_bw, 4.0 * silverman_bw]

    for bw in bw_grid:
        if time.time() > deadline - 10:
            break
        y_smooth = _nadaraya_watson(x, y, x, bw)
        for lam in [1e-4, 1e-3, 1e-2, 0.1]:
            c_norm = _fit_poly_ridge(x_norm, y_smooth, degree, lam)
            cv = cross_val_score(c_norm, x_mean, x_std, degree)
            candidates.append((cv, c_norm, 'nw', bw))

    # Select best
    if candidates:
        candidates.sort(key=lambda t: t[0])
        best_cv, best_c_norm, best_type, best_param = candidates[0]
        best_coeff = convert_norm_to_orig(best_c_norm, x_mean, x_std, degree)
    else:
        # Fallback: simple linear fit
        c1 = np.polyfit(x, y, 1)
        best_coeff = np.zeros(6)
        best_coeff[0] = c1[1]
        best_coeff[1] = c1[0]

    # Final polish: refine with lower ridge using the best as initial
    if time.time() < deadline - 5:
        try:
            # Use the best coefficients to do a local refinement
            x_eval = np.linspace(x.min(), x.max(), 200)
            # Compute residuals
            V = np.vander(x, N=degree + 1, increasing=True)
            pred = V @ best_coeff
            resid = y - pred
            # Add a small correction: fit residual with lower degree
            for corr_deg in [1, 2, 3]:
                c_corr = np.polyfit(x, resid, corr_deg)
                # c_corr is in descending power order from np.polyfit
                # Convert to ascending
                c_corr_asc = np.zeros(6)
                for i, val in enumerate(c_corr):
                    c_corr_asc[degree - i] = val
                # Only add if it improves
                V_corr = np.vander(x, N=corr_deg + 1, increasing=True)
                pred_corr = V_corr @ c_corr_asc[:corr_deg + 1]
                new_coeff = best_coeff.copy()
                new_coeff[:corr_deg + 1] += c_corr_asc[:corr_deg + 1]
                new_pred = V @ new_coeff
                new_mse = np.mean((new_pred - y) ** 2)
                old_mse = np.mean((pred - y) ** 2)
                if new_mse < old_mse:
                    best_coeff = new_coeff
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.where(np.isfinite(best_coeff), best_coeff, 0.0)

    return best_coeff
# EVOLVE-BLOCK-END
