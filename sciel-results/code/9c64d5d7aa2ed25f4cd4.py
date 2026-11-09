"""Symbolic regression fitter using random Fourier features with adaptive basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 290.0  # safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Normalize X for better numerical behavior
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std

    # Build a rich feature basis using random Fourier features + polynomial terms
    # This implicitly approximates an RBF kernel with finite-dimensional features

    # Number of random Fourier features - adaptive based on time
    n_fourier = 512

    # Generate random frequencies from a Gaussian (standard for RBF approximation)
    np.random.seed(42)
    # Scale frequencies based on input range
    freq_scale = 1.0 / X_std.mean()
    Omega = np.random.randn(n_fourier, d) * freq_scale

    # Random phases
    b = np.random.rand(n_fourier) * 2 * np.pi

    def fourier_features(X_input):
        """Compute random Fourier features."""
        X_f = (X_input - X_mean) / X_std
        # X_f: (n, d), Omega: (n_fourier, d)
        # Compute X_f @ Omega.T: (n, n_fourier)
        proj = X_f @ Omega.T
        # Features: [1, cos(proj + b), sin(proj + b)]
        cos_feat = np.cos(proj + b)
        sin_feat = np.sin(proj + b)
        return np.hstack([np.ones((len(X_input), 1)), cos_feat, sin_feat])

    # Compute training features
    F_train = fourier_features(X_train)

    # Also add polynomial features up to degree 2 for capturing low-order structure
    poly_cols = [np.ones((n, 1))]
    for j in range(d):
        poly_cols.append(X_norm[:, j:j+1])
    for j in range(d):
        poly_cols.append(X_norm[:, j:j+1] ** 2)
    if d >= 2:
        for i in range(d):
            for j in range(i+1, d):
                poly_cols.append(X_norm[:, i:i+1] * X_norm[:, j:j+1])

    F_poly = np.hstack(poly_cols)

    # Combine Fourier and polynomial features
    F_full = np.hstack([F_train, F_poly])

    # Use ridge regression for stability
    # Try different regularization strengths
    best_score = np.inf
    best_alpha = 1e-3
    best_w = None

    # Quick grid search over alpha
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        try:
            # Solve (F^T F + alpha*I) w = F^T y
            FtF = F_full.T @ F_full
            Fty = F_full.T @ y
            reg = alpha * np.eye(F_full.shape[1])
            w = np.linalg.solve(FtF + reg, Fty)

            # Compute training residual as proxy
            pred = F_full @ w
            mse = np.mean((pred - y) ** 2)

            # Simple validation: use first 80% for training, last 20% for validation
            split = int(0.8 * n)
            F_tr = F_full[:split]
            y_tr = y[:split]
            F_val = F_full[split:]
            y_val = y[split:]

            FtF_v = F_tr.T @ F_tr
            Fty_v = F_tr.T @ y_tr
            w_v = np.linalg.solve(FtF_v + reg, Fty_v)
            pred_val = F_val @ w_v
            val_mse = np.mean((pred_val - y_val) ** 2)

            if val_mse < best_score:
                best_score = val_mse
                best_alpha = alpha
                best_w = w
        except:
            continue

    if best_w is None:
        # Fallback: use ridge with default alpha
        alpha = 1e-3
        FtF = F_full.T @ F_full
        Fty = F_full.T @ y
        best_w = np.linalg.solve(FtF + alpha * np.eye(F_full.shape[1]), Fty)

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        F_pred = fourier_features(X)

        # Compute polynomial features for prediction
        X_p = (X - X_mean) / X_std
        poly_cols = [np.ones((len(X), 1))]
        for j in range(d):
            poly_cols.append(X_p[:, j:j+1])
        for j in range(d):
            poly_cols.append(X_p[:, j:j+1] ** 2)
        if d >= 2:
            for i in range(d):
                for j in range(i+1, d):
                    poly_cols.append(X_p[:, i:i+1] * X_p[:, j:j+1])

        F_poly_pred = np.hstack(poly_cols)
        F_full_pred = np.hstack([F_pred, F_poly_pred])

        return F_full_pred @ best_w

    return predict
# EVOLVE-BLOCK-END
