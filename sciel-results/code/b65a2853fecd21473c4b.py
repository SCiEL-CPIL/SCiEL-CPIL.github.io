"""Symbolic regression over 4 hidden 1D equations using 1D Gaussian Process."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=np.float64).ravel()
    y = np.asarray(y_train, dtype=np.float64).ravel()
    n = len(X)

    # Center and scale inputs for better GP behavior
    x_mean = X.mean()
    x_std = X.std()
    if x_std < 1e-12:
        x_std = 1.0
    Xc = (X - x_mean) / x_std

    # Center and scale outputs for numerical stability
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    yc = (y - y_mean) / y_std

    # RBF kernel
    def kernel(X1, X2, l, s2):
        # X1: (n,), X2: (m,)
        diff = X1[:, None] - X2[None, :]
        return s2 * np.exp(-0.5 * (diff ** 2) / (l ** 2))

    def log_marginal_likelihood(theta):
        l = np.exp(theta[0])
        s2 = np.exp(theta[1])
        noise = np.exp(theta[2])
        # Add jitter for numerical stability
        K = kernel(Xc, Xc, l, s2) + (noise + 1e-10) * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        # log p(y|X) = -0.5 * y^T K^{-1} y - 0.5 * log det K - n/2 * log(2*pi)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        lml = -0.5 * yc @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        return -lml  # minimize negative

    # Initial hyperparameters: length-scale, signal variance, noise
    # Try multiple starting points for robustness
    starts = [
        np.array([np.log(0.5), np.log(1.0), np.log(1e-6)]),
        np.array([np.log(0.1), np.log(0.5), np.log(1e-8)]),
        np.array([np.log(1.0), np.log(2.0), np.log(1e-4)]),
        np.array([np.log(2.0), np.log(5.0), np.log(1e-6)]),
        np.array([np.log(0.3), np.log(0.8), np.log(1e-10)]),
    ]
    bounds = [(np.log(0.01), np.log(10.0)), (np.log(0.01), np.log(100.0)), (np.log(1e-10), np.log(1.0))]

    best_theta = starts[0]
    best_val = np.inf
    for start in starts:
        try:
            res = minimize(log_marginal_likelihood, start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x
        except Exception:
            continue

    l_opt = np.exp(best_theta[0])
    s2_opt = np.exp(best_theta[1])
    noise_opt = np.exp(best_theta[2])

    K = kernel(Xc, Xc, l_opt, s2_opt) + (noise_opt + 1e-10) * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))

    def predict(X_test):
        Xt = np.asarray(X_test, dtype=np.float64)
        if Xt.ndim == 1:
            Xt = Xt.reshape(-1, 1)
        Xt_flat = Xt.ravel()
        Xtc = (Xt_flat - x_mean) / x_std
        K_s = kernel(Xtc, Xc, l_opt, s2_opt)  # (m, n)
        y_pred_scaled = K_s @ alpha
        y_pred = y_pred_scaled * y_std + y_mean
        return y_pred.reshape(Xt.shape[0])

    return predict
# EVOLVE-BLOCK-END
