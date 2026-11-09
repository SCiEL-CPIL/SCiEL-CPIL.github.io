"""Symbolic regression via dense Fourier basis + ridge regression (with poly fallback)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before the 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    def fourier_features(Xraw, max_freq):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones((len(Xrn), 1))]
        for j in range(d):
            z = Xrn[:, j]
            for f in range(1, max_freq + 1):
                cols.append(np.sin(2.0 * np.pi * f * z).reshape(-1, 1))
                cols.append(np.cos(2.0 * np.pi * f * z).reshape(-1, 1))
        return np.column_stack(cols)

    def poly_features(Xraw, deg):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones((len(Xrn), 1))]
        for deg_ in range(1, deg + 1):
            for k in range(d):
                cols.append((Xrn[:, k] ** deg_).reshape(-1, 1))
        return np.column_stack(cols)

    def solve_ridge(F, yv, lam):
        # Solve (F'F + lam I) w = F' y  via Cholesky (fallback to lstsq)
        FtF = F.T @ F
        Fty = F.T @ yv
        reg = lam * np.eye(FtF.shape[0])
        reg[0, 0] += 1e-6  # keep the intercept lightly regularized
        A = FtF + reg
        try:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L.T, np.linalg.solve(L, Fty))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, Fty, rcond=None)[0]
        return w

    def predict_with(F_train, w, Xraw):
        return fourier_features(Xraw, F_train.shape[1] and max_freq_fit) @ w

    # ---- Choose the densest Fourier frequency grid that fits in time ----
    # Estimate cost of building features for n rows at a given max_freq.
    # features ~ n * (1 + 2*d*max_freq)
    def est_build_time(max_freq, rows):
        # rough: building sin/cos for d*max_freq columns over rows
        # scale from a tiny measured probe
        return None  # we use a measured probe instead

    # Measure a probe to calibrate build time per column-row
    probe_freq = 1
    Fp = fourier_features(X, probe_freq)
    t_probe = time.time() - t0
    # Fp has 1 + 2*d*probe_freq columns over n rows
    cols_probe = Fp.shape[1]
    cost_per_colrow = t_probe / max(cols_probe * n, 1)

    # Target: leave time for ridge solve + predict. Budget ~120s for building.
    budget_build = 120.0
    # Solve cost ~ n * cols^2 (matrix mult) ; estimate and include.
    def total_est(max_freq):
        cols = 1 + 2 * d * max_freq
        build = cost_per_colrow * cols * n
        # ridge: F'F is cols x cols, F is n x cols -> ~ 2*n*cols^2 flops
        # approximate seconds using a per-flop guess; keep conservative.
        solve = 1e-11 * (2 * n * cols * cols)
        return build + solve

    max_freq_fit = 1
    for mf in range(1, 60):
        if total_est(mf) < budget_build:
            max_freq_fit = mf
        else:
            break

    # Build the Fourier design matrix
    F = fourier_features(X, max_freq_fit)

    # Ridge parameter: scale by number of features for stability
    lam = 1e-3 * F.shape[1]
    w = solve_ridge(F, y, lam)

    # Evaluate training error to decide whether to keep
    def train_rmse_norm(pred):
        var_y = float(np.var(y))
        if var_y < 1e-12:
            return 0.0
        return float(np.mean((pred - y) ** 2) / var_y)

    pred_train = F @ w
    err = train_rmse_norm(pred_train)

    # Fallback: if Fourier fit is poor, try a degree-3 polynomial ridge
    if err > 0.5 and (time.time() - t0) < (DEADLINE - 60.0):
        Fp3 = poly_features(X, 3)
        lam3 = 1e-3 * Fp3.shape[1]
        w3 = solve_ridge(Fp3, y, lam3)
        pred3 = Fp3 @ w3
        err3 = train_rmse_norm(pred3)
        if err3 < err:
            # use polynomial
            def predict(Xraw):
                return poly_features(Xraw, 3) @ w3
            return predict

    def predict(Xraw):
        return fourier_features(Xraw, max_freq_fit) @ w

    return predict
# EVOLVE-BLOCK-END
