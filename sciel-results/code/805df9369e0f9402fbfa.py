# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 300.0  # seconds total budget
    TIME_MARGIN = 5.0  # safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Precompute some useful quantities
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        # Constant target
        return lambda X_test: np.full(len(X_test), y_mean)

    # Normalize X for better conditioning
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = 2.0 * (X - X_min) / X_range - 1.0  # map to [-1, 1]

    def denormalize(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        return 2.0 * (X_test - X_min) / X_range - 1.0

    # Candidate models to try
    candidates = []

    # 1. Polynomial features of various degrees with ridge regression
    def poly_features(Xn, degree):
        """Generate polynomial features up to given degree."""
        n, d = Xn.shape
        cols = [np.ones(n)]
        for deg in range(1, degree + 1):
            for i in range(d):
                for j in range(i, d):
                    # Only include monomials of total degree exactly 'deg'
                    # This is a simplification; we'll generate all monomials up to degree
                    pass
        # Better approach: generate all monomials up to 'degree'
        cols = [np.ones(n)]
        for deg in range(1, degree + 1):
            # Generate all combinations of exponents that sum to deg
            for combo in _monomials(d, deg):
                term = np.ones(n)
                for i, exp in enumerate(combo):
                    if exp > 0:
                        term = term * (Xn[:, i] ** exp)
                cols.append(term)
        return np.column_stack(cols)

    def _monomials(d, total_deg):
        """Generate all exponent vectors of length d summing to total_deg."""
        if total_deg == 0:
            yield (0,) * d
            return
        if d == 1:
            yield (total_deg,)
            return
        for first in range(total_deg + 1):
            for rest in _monomials(d - 1, total_deg - first):
                yield (first,) + rest

    # Try polynomial degrees 1, 2, 3, 4 with ridge
    for deg in [1, 2, 3, 4]:
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        try:
            Phi = poly_features(X_norm, deg)
            # Ridge regression
            for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
                A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
                b = Phi.T @ y
                w = np.linalg.solve(A, b)
                pred = Phi @ w
                mse = np.mean((pred - y) ** 2)
                candidates.append(('poly', deg, alpha, w, Phi.shape[1], mse))
        except Exception:
            pass

    # 2. RBF Kernel Ridge Regression
    if time.time() < DEADLINE - TIME_MARGIN:
        try:
            # Compute Gram matrix
            diff = X_norm[:, :, None] - X_norm[:, None, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
                K = np.exp(-gamma * dist_sq)
                for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
                    A = K + alpha * np.eye(n)
                    b = y
                    alpha_vec = np.linalg.solve(A, b)
                    pred = K @ alpha_vec
                    mse = np.mean((pred - y) ** 2)
                    candidates.append(('rbf', gamma, alpha, alpha_vec, None, mse))
        except Exception:
            pass

    # 3. Gaussian Process with Matern-5/2 kernel (simplified)
    if time.time() < DEADLINE - TIME_MARGIN:
        try:
            # Use a simple GP with Matern-5/2 kernel
            # For simplicity, we'll use a fixed length-scale and optimize signal variance
            for length_scale in [0.5, 1.0, 2.0]:
                for signal_var in [0.1, 1.0, 10.0]:
                    # Compute kernel matrix
                    diff = X_norm[:, :, None] - X_norm[:, None, :]
                    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
                    # Matern-5/2 kernel
                    K = signal_var * (1 + np.sqrt(5) * dist / length_scale + 5 * dist ** 2 / (3 * length_scale ** 2)) * np.exp(-np.sqrt(5) * dist / length_scale)
                    noise_var = 0.01
                    A = K + noise_var * np.eye(n)
                    L = np.linalg.cholesky(A)
                    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    pred = K @ alpha_vec
                    mse = np.mean((pred - y) ** 2)
                    candidates.append(('gp', length_scale, signal_var, alpha_vec, None, mse))
        except Exception:
            pass

    # 4. Rational function approximator: ratio of two linear basis expansions
    if time.time() < DEADLINE - TIME_MARGIN:
        try:
            # Use RBF basis for both numerator and denominator
            for gamma in [0.5, 1.0, 2.0]:
                diff = X_norm[:, :, None] - X_norm[:, None, :]
                dist_sq = np.sum(diff ** 2, axis=2)
                K = np.exp(-gamma * dist_sq)
                # Fit numerator and denominator separately using least squares
                # y * denom ≈ num
                # This is nonlinear; use iterative approach or simple heuristic
                # For simplicity, fit a linear model to y and use that as a baseline
                # Actually, let's just skip this complex approach and rely on the others
                pass
        except Exception:
            pass

    # Select the best candidate based on training MSE (as a proxy for test performance)
    if not candidates:
        # Fallback: simple linear regression
        X_aug = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            Xn = denormalize(X_test)
            Xn_aug = np.column_stack([np.ones(len(Xn)), Xn])
            return Xn_aug @ w
        return predict

    # Sort by training MSE and pick the best
    candidates.sort(key=lambda c: c[5])
    best = candidates[0]

    # Build the prediction function for the best candidate
    model_type = best[0]

    if model_type == 'poly':
        _, deg, alpha, w, n_features, _ = best
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            Xn = denormalize(X_test)
            Phi = poly_features(Xn, deg)
            return Phi @ w
    elif model_type == 'rbf':
        _, gamma, alpha, alpha_vec, _, _ = best
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            Xn = denormalize(X_test)
            # Compute kernel between test points and training points
            diff = Xn[:, :, None] - X_norm[:, None, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            K_test = np.exp(-gamma * dist_sq)
            return K_test @ alpha_vec
    elif model_type == 'gp':
        _, length_scale, signal_var, alpha_vec, _, _ = best
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            Xn = denormalize(X_test)
            diff = Xn[:, :, None] - X_norm[:, None, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
            K_test = signal_var * (1 + np.sqrt(5) * dist / length_scale + 5 * dist ** 2 / (3 * length_scale ** 2)) * np.exp(-np.sqrt(5) * dist / length_scale)
            return K_test @ alpha_vec
    else:
        # Fallback
        X_aug = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            Xn = denormalize(X_test)
            Xn_aug = np.column_stack([np.ones(len(Xn)), Xn])
            return Xn_aug @ w

    # Final sanity check: ensure predictions are finite
    def safe_predict(X_test):
        try:
            pred = predict(X_test)
            pred = np.asarray(pred, dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                # Replace non-finite values with mean
                pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        except Exception:
            return np.full(len(X_test), y_mean)

    return safe_predict
# EVOLVE-BLOCK-END
