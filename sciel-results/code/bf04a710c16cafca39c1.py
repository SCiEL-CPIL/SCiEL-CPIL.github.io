"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _poly_features(X, degree=2):
    """Build polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for j in range(d):
        cols.append(X[:, j])
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std

    best_score = np.inf
    best_predictor = None

    # Polynomial baselines
    for deg in [1, 2, 3]:
        if time.time() > deadline - 5.0:
            break
        try:
            F = _poly_features(X, deg)
            w, *_ = np.linalg.lstsq(F, y, rcond=None)
            pred = F @ w
            score = np.mean((pred - y) ** 2)
            if score < best_score:
                best_score = score
                best_predictor = lambda X, F=F, w=w: F @ w
        except Exception:
            pass

    # RBF kernel regression
    if time.time() < deadline - 10.0:
        try:
            n_sample = min(50, n_samples)
            idx = np.random.choice(n_samples, n_sample, replace=False)
            Xs = X[idx]
            dists = []
            for i in range(n_sample):
                for j in range(i + 1, n_sample):
                    dists.append(np.sum((Xs[i] - Xs[j]) ** 2))
            median_dist_sq = np.median(dists) if dists else 1.0

            bw_list = [max(median_dist_sq * f, 1e-6) for f in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]]

            Xn = X_norm
            for bw in bw_list:
                if time.time() > deadline - 5.0:
                    break
                try:
                    dist_sq = np.maximum(
                        np.sum(Xn ** 2, axis=1).reshape(-1, 1) +
                        np.sum(Xn ** 2, axis=1).reshape(1, -1) -
                        2.0 * Xn @ Xn.T, 0.0)
                    K = np.exp(-dist_sq / (2.0 * bw ** 2))
                    alpha = 1e-3
                    K_reg = K + alpha * np.eye(n_samples)
                    coeffs, *_ = np.linalg.lstsq(K_reg, y, rcond=None)
                    pred = K @ coeffs
                    score = np.mean((pred - y) ** 2)
                    if score < best_score:
                        best_score = score
                        Xn_store = Xn.copy()
                        c = coeffs.copy()
                        bw_use = bw
                        Xn_mean = X_mean.copy()
                        Xn_std = X_std.copy()
                        def best_predictor(X, Xn_store=Xn_store, c=c, bw_use=bw_use, Xn_mean=Xn_mean, Xn_std=Xn_std):
                            Xn_test = (X - Xn_mean) / Xn_std
                            dist_sq_t = np.maximum(
                                np.sum(Xn_test ** 2, axis=1).reshape(-1, 1) +
                                np.sum(Xn_store ** 2, axis=1).reshape(1, -1) -
                                2.0 * Xn_test @ Xn_store.T, 0.0)
                            K_test = np.exp(-dist_sq_t / (2.0 * bw_use ** 2))
                            return K_test @ c
                except Exception:
                    pass
        except Exception:
            pass

    if best_predictor is None:
        y_mean = np.mean(y)
        return lambda X: np.full(X.shape[0], y_mean)

    return best_predictor
# EVOLVE-BLOCK-END
