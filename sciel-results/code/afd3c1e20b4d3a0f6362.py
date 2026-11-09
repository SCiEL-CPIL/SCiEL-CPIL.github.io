# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill

    X = X_train.reshape(-1, 1)[:, 0]
    y = y_train.reshape(-1)
    n = len(X)

    # Sort by x for spline fitting
    order = np.argsort(X)
    Xs = X[order]
    ys = y[order]

    # Remove duplicate x values (keep mean)
    unique_x, idx_start, counts = np.unique(Xs, return_index=True, return_counts=True)
    if len(unique_x) < len(Xs):
        # Average y values for duplicate x
        y_sum = np.zeros(len(unique_x))
        for i, (x_val, s, c) in enumerate(zip(Xs, idx_start, counts)):
            pass
        # More efficient: use np.add.at
        y_sum = np.zeros(len(unique_x))
        np.add.at(y_sum, np.searchsorted(unique_x, Xs), ys)
        ys_unique = y_sum / counts
        Xs_unique = unique_x
    else:
        Xs_unique = Xs
        ys_unique = ys

    # Try PCHIP first (monotone, no overshoot)
    try:
        pchip = PchipInterpolator(Xs_unique, ys_unique)
        # Evaluate on training points to check quality
        train_pred_pchip = pchip(Xs_unique)
        train_mse_pchip = np.mean((train_pred_pchip - ys_unique) ** 2)
    except Exception:
        pchip = None
        train_mse_pchip = np.inf

    # Try CubicSpline
    try:
        cs = CubicSpline(Xs_unique, ys_unique)
        train_pred_cs = cs(Xs_unique)
        train_mse_cs = np.mean((train_pred_cs - ys_unique) ** 2)
    except Exception:
        cs = None
        train_mse_cs = np.inf

    # RBF kernel regression as fallback/alternative
    # Use Gaussian RBF with multiple bandwidths
    best_rbf = None
    best_rbf_mse = np.inf
    if time.time() - start_time < DEADLINE:
        # Try a few bandwidths
        x_range = Xs_unique.max() - Xs_unique.min()
        if x_range > 0:
            bandwidths = [x_range * f for f in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]]
            for bw in bandwidths:
                if time.time() - start_time >= DEADLINE:
                    break
                try:
                    # Compute kernel matrix
                    dists = np.abs(Xs_unique[:, None] - Xs_unique[None, :])
                    K = np.exp(-(dists ** 2) / (2 * bw ** 2))
                    K += 1e-10 * np.eye(n)
                    w_rbf = np.linalg.solve(K, ys_unique)
                    # Check training MSE
                    pred_train = K @ w_rbf
                    mse_rbf = np.mean((pred_train - ys_unique) ** 2)
                    if mse_rbf < best_rbf_mse:
                        best_rbf_mse = mse_rbf
                        best_rbf = (Xs_unique, w_rbf, bw)
                except Exception:
                    continue

    # Choose the best model based on training MSE
    # For noiseless data, we want the model that interpolates best
    candidates = []
    if pchip is not None:
        candidates.append(('pchip', pchip, train_mse_pchip))
    if cs is not None:
        candidates.append(('cs', cs, train_mse_cs))
    if best_rbf is not None:
        candidates.append(('rbf', best_rbf, best_rbf_mse))

    # Pick the one with lowest training MSE (should be near-zero for interpolation)
    if candidates:
        candidates.sort(key=lambda x: x[2])
        best_name, best_model, best_mse = candidates[0]
    else:
        # Last resort: linear interpolation
        best_model = np.interp
        best_name = 'interp'

    # Define predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float).reshape(-1, 1)[:, 0]
        if best_name == 'pchip':
            try:
                return best_model(X_test)
            except Exception:
                return np.interp(X_test, Xs_unique, ys_unique)
        elif best_name == 'cs':
            try:
                return best_model(X_test)
            except Exception:
                return np.interp(X_test, Xs_unique, ys_unique)
        elif best_name == 'rbf':
            Xs_ref, w_rbf, bw = best_model
            try:
                dists = np.abs(X_test[:, None] - Xs_ref[None, :])
                K = np.exp(-(dists ** 2) / (2 * bw ** 2))
                return K @ w_rbf
            except Exception:
                return np.interp(X_test, Xs_unique, ys_unique)
        else:
            return np.interp(X_test, Xs_unique, ys_unique)

    return predict
# EVOLVE-BLOCK-END
