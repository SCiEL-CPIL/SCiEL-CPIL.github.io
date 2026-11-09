"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def fit(X_train, y_train):
    """Fit using Gaussian Process regression for 1D data,
    with a polynomial fallback for multi-dimensional inputs."""
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    if d == 1:
        x = X[:, 0].copy()
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # GP with RBF kernel, noiseless data
        # Hyperparameters: log(length_scale), log(signal_variance)
        # For noiseless data, we can use a small noise floor to keep Cholesky stable

        def rbf_kernel(x1, x2, length_scale):
            sq_dists = (x1[:, None] - x2[None, :])**2
            return np.exp(-sq_dists / (2 * length_scale**2))

        def neg_log_likelihood(params):
            log_ls, log_sv, log_noise = params
            ls = np.exp(log_ls)
            sv = np.exp(log_sv)
            noise = np.exp(log_noise)
            
            K = rbf_kernel(x_sorted, x_sorted, ls) * sv
            K += noise * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
            except:
                return 1e10
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_sorted))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            nll = 0.5 * (y_sorted @ alpha + log_det + n * np.log(2 * np.pi))
            return nll

        # Initial guesses
        x_range = x_sorted[-1] - x_sorted[0]
        if x_range < 1e-10:
            x_range = 1.0
        init_ls = x_range / 3.0
        init_sv = np.var(y_sorted) if np.var(y_sorted) > 1e-10 else 1.0
        init_noise = 1e-6

        x0 = [np.log(init_ls), np.log(init_sv), np.log(init_noise)]
        bounds = [
            (np.log(x_range / 100), np.log(x_range * 10)),
            (np.log(1e-4), np.log(1e4)),
            (np.log(1e-8), np.log(1e-2))
        ]

        # Optimize hyperparameters
        best_params = x0
        best_nll = neg_log_likelihood(x0)

        try:
            result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 100, 'maxfun': 200})
            if result.fun < best_nll:
                best_params = result.x
                best_nll = result.fun
        except:
            pass

        if time.time() > deadline:
            # Fallback to simple interpolation
            from scipy.interpolate import interp1d
            f_interp = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
            def predict_interp(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                return f_interp(X_test[:, 0])
            return predict_interp

        log_ls, log_sv, log_noise = best_params
        ls = np.exp(log_ls)
        sv = np.exp(log_sv)
        noise = np.exp(log_noise)

        # Precompute Cholesky factor for fast prediction
        K = rbf_kernel(x_sorted, x_sorted, ls) * sv
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_sorted))
        except:
            # Fallback to interpolation if Cholesky fails
            from scipy.interpolate import interp1d
            f_interp = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
            def predict_interp(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                return f_interp(X_test[:, 0])
            return predict_interp

        def predict_gp(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            xt = X_test[:, 0]
            
            # K_* : (n_test, n_train)
            K_star = rbf_kernel(xt, x_sorted, ls) * sv
            # Predictions: K_star @ alpha
            preds = K_star @ alpha
            return preds

        return predict_gp

    else:
        # Multi-dimensional fallback: polynomial regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)

        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)

        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, d)
            return features(X_test) @ w

        return predict_poly

# EVOLVE-BLOCK-END
