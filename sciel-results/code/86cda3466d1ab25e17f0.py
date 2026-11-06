"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Strategy:
    1. Exploit physical antisymmetry: the fast-slow coupling in Lorenz-96
       is antisymmetric in the slow state x (the forcing depends on the
       gradient of the slow field, which is odd under x -> -x). So we
       restrict to odd powers: x, x^3, x^5.
    2. Use Tikhonov (L2) regularization with cross-validated lambda.
    3. Include a fallback to full degree-5 if odd-only performs worse.
    4. Deadline guard: return best result before 150s.
    """
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n < 6:
        # Not enough data; return zeros
        return np.zeros(6)

    # Center and scale x for numerical stability in polynomial fitting
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    def time_check():
        return (time.time() - t_start) > DEADLINE

    # Build Vandermonde-like matrix for odd powers: x, x^3, x^5
    # In normalized space
    X_odd = np.column_stack([
        x_norm,
        x_norm ** 3,
        x_norm ** 5
    ])

    # Also build full degree-5: x, x^2, x^3, x^4, x^5 (and constant)
    # Actually the task says ASCENDING powers of local slow state x: c_0 + c_1*x + ... + c_5*x^5
    # So the polynomial is sum_{i=0}^{5} c_i * x^i
    # For odd-only: c_0=0, c_2=0, c_4=0, and we fit c_1, c_3, c_5
    # But we need to be careful: the antisymmetry is in the PHYSICAL x, not normalized.
    # However, for fitting purposes, we fit in normalized space and then transform back.

    # Actually, let's think about this more carefully.
    # The polynomial is: f(x) = c_0 + c_1*x + c_2*x^2 + c_3*x^3 + c_4*x^4 + c_5*x^5
    # If we restrict to odd powers: f(x) = c_1*x + c_3*x^3 + c_5*x^5
    # We fit in the original x space directly, or in normalized space.
    # Let's fit in normalized space for stability, then transform coefficients back.

    # In normalized space: f(x_norm) = a_0 + a_1*x_norm + a_2*x_norm^2 + a_3*x_norm^3 + a_4*x_norm^4 + a_5*x_norm^5
    # Odd-only in normalized: a_0=0, a_2=0, a_4=0, fit a_1, a_3, a_5
    # But wait: the antisymmetry is in the physical x, not x_norm.
    # x_norm = (x - x_mean)/x_std, so x = x_mean + x_std*x_norm
    # If f(x) is odd in x, then f(x_mean + x_std*t) is not necessarily odd in t unless x_mean=0.
    # So we should fit in the original x space, or account for the shift.

    # Better approach: fit directly in original x space with regularization.
    # For numerical stability, we can still use scaling but need to transform back.

    # Let's do it in original space with a well-conditioned approach.
    # Actually, for degree 5, direct least squares can be ill-conditioned.
    # Let's use the normalized approach but fit the full polynomial in normalized space,
    # then transform back. For the odd-only constraint, we need to be careful.

    # Alternative: just fit in original space with SVD-based least squares + regularization.
    # numpy's lstsq handles this well.

    # Let's try: fit full degree-5 with Tikhonov, and odd-only degree-5 with Tikhonov.
    # Compare via k-fold CV, pick the better one.

    def fit_full(X, y, lam):
        """Fit full degree-5 polynomial with Tikhonov regularization."""
        # X is (n, 6) Vandermonde matrix for x^0, x^1, ..., x^5
        # Minimize ||X @ c - y||^2 + lam * ||c||^2
        # Solution: c = (X^T X + lam*I)^{-1} X^T y
        XtX = X.T @ X
        Xty = X.T @ y
        A = XtX + lam * np.eye(6)
        try:
            c = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, Xty, rcond=None)[0]
        return c

    def fit_odd(X_full, y, lam):
        """Fit odd-only degree-5 polynomial with Tikhonov regularization.
        X_full is (n, 6) Vandermonde for x^0,...,x^5.
        We only use columns 1, 3, 5 (x^1, x^3, x^5).
        """
        X_odd = X_full[:, [1, 3, 5]]
        XtX = X_odd.T @ X_odd
        Xty = X_odd.T @ y
        A = XtX + lam * np.eye(3)
        try:
            c_odd = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            c_odd = np.linalg.lstsq(A, Xty, rcond=None)[0]
        # Map back to full 6-coeff vector
        c_full = np.zeros(6)
        c_full[1] = c_odd[0]
        c_full[3] = c_odd[1]
        c_full[5] = c_odd[2]
        return c_full

    def predict(c, x_test):
        """Predict using full 6-coeff polynomial."""
        return np.sum(c * x_test[:, None] ** np.arange(6)[None, :], axis=1)

    def kfold_cv(X, y, fit_fn, lam, k=5):
        """K-fold cross-validation for a given fit function and lambda."""
        n = len(y)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // k
        residuals = []
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else n
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            c = fit_fn(X_train, y_train, lam)
            pred = predict(c, X_val)
            residuals.append(np.mean((pred - y_val) ** 2))
        return np.mean(residuals)

    # Build Vandermonde matrix in original x space
    # Use scaled x for better conditioning: x_scaled = x / x_scale
    # where x_scale is chosen so that x_scaled is roughly in [-3, 3]
    x_scale = np.max(np.abs(x))
    if x_scale < 1e-12:
        x_scale = 1.0
    x_scaled = x / x_scale

    # Build Vandermonde for x_scaled: columns are x_scaled^0, x_scaled^1, ..., x_scaled^5
    powers = np.arange(6)
    X_vand = x_scaled[:, None] ** powers[None, :]

    # Regularization strengths to try
    lambdas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    # Try odd-only with various lambdas
    best_cv_odd = np.inf
    best_c_odd = np.zeros(6)
    best_lam_odd = 0.0

    for lam in lambdas:
        if time_check():
            break
        cv_score = kfold_cv(X_vand, y, fit_odd, lam)
        if cv_score < best_cv_odd:
            best_cv_odd = cv_score
            best_c_odd = fit_odd(X_vand, y, lam)
            best_lam_odd = lam

    # Try full with various lambdas
    best_cv_full = np.inf
    best_c_full = np.zeros(6)
    best_lam_full = 0.0

    for lam in lambdas:
        if time_check():
            break
        cv_score = kfold_cv(X_vand, y, fit_full, lam)
        if cv_score < best_cv_full:
            best_cv_full = cv_score
            best_c_full = fit_full(X_vand, y, lam)
            best_lam_full = lam

    # Pick the better model based on CV
    if best_cv_odd <= best_cv_full:
        c_scaled = best_c_odd
    else:
        c_scaled = best_c_full

    # Transform coefficients back from x_scaled to x
    # f(x) = sum_{i=0}^{5} c_scaled_i * (x/x_scale)^i
    #      = sum_{i=0}^{5} c_scaled_i / x_scale^i * x^i
    c_original = c_scaled / (x_scale ** powers)

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(c_original)):
        # Fallback: simple linear fit
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            c_original = np.zeros(6)
            c_original[:2] = c_lin
        except Exception:
            c_original = np.zeros(6)

    return c_original
# EVOLVE-BLOCK-END
