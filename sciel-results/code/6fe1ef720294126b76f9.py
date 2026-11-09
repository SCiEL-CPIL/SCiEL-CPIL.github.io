# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import differential_evolution
from scipy.linalg import solve, cho_factor, cho_solve
from scipy.stats import qnorm

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # 50s margin for 330s kill

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)

    def check_time():
        return time.time() < deadline

    # Normalize features for numerical stability
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X_train - X_min) / X_range

    # Normalize y
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf

    best_predict = None
    best_mse = np.inf

    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_mse:
            best_mse = mse
            best_predict = model

    # Strategy 1: Gaussian Process Regression with adaptive hyperparameters
    def gp_predict(X, y, kernel_fn, alpha):
        """Train GP and return predict callable"""
        K = kernel_fn(X, X) + alpha * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
            alpha_inv_y = np.linalg.solve(np.linalg.cholesky(K).T, y)
            alpha_inv_y = np.linalg.solve(np.linalg.cholesky(K), alpha_inv_y)
        except np.linalg.LinAlgError:
            # Add jitter and retry
            K += 1e-6 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
                alpha_inv_y = np.linalg.solve(np.linalg.cholesky(K).T, y)
                alpha_inv_y = np.linalg.solve(np.linalg.cholesky(K), alpha_inv_y)
            except Exception:
                return None

        def predict_gp(X_new):
            K_new = kernel_fn(X_new, X)
            try:
                mu = K_new @ alpha_inv_y
                return mu
            except Exception:
                return np.zeros(len(X_new))

        return predict_gp

    def make_rbf_kernel(length_scale, variance):
        def kernel(X1, X2):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            return variance * np.exp(-0.5 * dist_sq / (length_scale ** 2 + 1e-10))
        return kernel

    def make_linear_kernel(variance):
        def kernel(X1, X2):
            return variance * (X1 @ X2.T)
        return kernel

    def make_composite_kernel(rbf_ls, rbf_var, lin_var):
        rbf = make_rbf_kernel(rbf_ls, rbf_var)
        lin = make_linear_kernel(lin_var)
        def kernel(X1, X2):
            return rbf(X1, X2) + lin(X1, X2)
        return kernel

    # Try GP with different hyperparameters
    # Estimate characteristic length scale from data
    pairwise_dists = []
    if n > 10:
        idx = np.random.choice(n, min(50, n), replace=False)
        X_sample = X_norm[idx]
        for i in range(len(X_sample)):
            for j in range(i+1, len(X_sample)):
                pairwise_dists.append(np.linalg.norm(X_sample[i] - X_sample[j]))
    if pairwise_dists:
        median_dist = np.median(pairwise_dists)
    else:
        median_dist = 1.0

    for ls_mult in [0.3, 0.5, 1.0, 2.0, 3.0]:
        if not check_time():
            break
        for var_mult in [0.5, 1.0, 2.0]:
            if not check_time():
                break
            for alpha in [1e-6, 1e-5, 1e-4]:
                if not check_time():
                    break
                try:
                    ls = median_dist * ls_mult
                    var = var_mult
                    kernel_fn = make_rbf_kernel(ls, var)
                    model = gp_predict(X_norm, y_norm, kernel_fn, alpha)
                    if model is not None:
                        def predict_gp_wrapper(X, model=model, y_mean=y_mean, y_std=y_std):
                            Xn = (X - X_min) / X_range
                            return y_mean + y_std * model(Xn)
                        update_best(predict_gp_wrapper)
                except Exception:
                    pass

    # Composite kernel GP (RBF + Linear)
    for ls_mult in [0.5, 1.0, 2.0]:
        if not check_time():
            break
        for alpha in [1e-6, 1e-5]:
            if not check_time():
                break
                try:
                    ls = median_dist * ls_mult
                    kernel_fn = make_composite_kernel(ls, 1.0, 1.0)
                    model = gp_predict(X_norm, y_norm, kernel_fn, alpha)
                    if model is not None:
                        def predict_comp_wrapper(X, model=model, y_mean=y_mean, y_std=y_std):
                            Xn = (X - X_min) / X_range
                            return y_mean + y_std * model(Xn)
                        update_best(predict_comp_wrapper)
                except Exception:
                    pass

    # Strategy 2: Polynomial regression with various degrees
    def make_poly_features(X, degree):
        Xn = (X - X_min) / X_range
        cols = [np.ones(len(Xn))]
        for j in range(Xn.shape[1]):
            for k in range(1, degree + 1):
                cols.append(Xn[:, j] ** k)
        # Interaction terms
        for i in range(Xn.shape[1]):
            for j in range(i, Xn.shape[1]):
                if degree >= 2:
                    cols.append(Xn[:, i] * Xn[:, j])
                if degree >= 3:
                    for k in range(j, Xn.shape[1]):
                        cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k])
                if degree >= 4:
                    for k in range(j, Xn.shape[1]):
                        for l in range(k, Xn.shape[1]):
                            cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k] * Xn[:, l])
        return np.column_stack(cols)

    for deg in [1, 2, 3, 4]:
        if not check_time():
            break
        try:
            F = make_poly_features(X_train, deg)
            lam = 1e-5
            A = F.T @ F + lam * np.eye(F.shape[1])
            A[0, 0] -= lam
            w = solve(A, F.T @ y_norm)

            def predict_poly(X, F_fn, w, y_mean, y_std):
                F_test = F_fn(X)
                return y_mean + y_std * (F_test @ w)

            update_best(lambda X: predict_poly(X, make_poly_features, w, y_mean, y_std))
        except Exception:
            pass

    # Strategy 3: Fourier basis
    def make_fourier_features(X, max_freq):
        Xn = (X - X_min) / X_range
        cols = [np.ones(len(Xn))]
        for j in range(Xn.shape[1]):
            for k in range(1, max_freq + 1):
                cols.append(np.sin(k * 2 * np.pi * Xn[:, j]))
                cols.append(np.cos(k * 2 * np.pi * Xn[:, j]))
        # Cross terms
        if Xn.shape[1] >= 2:
            for i in range(Xn.shape[1]):
                for j in range(i + 1, Xn.shape[1]):
                    cols.append(np.sin(Xn[:, i] + Xn[:, j]))
                    cols.append(np.cos(Xn[:, i] + Xn[:, j]))
        return np.column_stack(cols)

    for freq in [2, 3, 4, 5]:
        if not check_time():
            break
        try:
            F = make_fourier_features(X_train, freq)
            lam = 1e-5
            A = F.T @ F + lam * np.eye(F.shape[1])
            A[0, 0] -= lam
            w = solve(A, F.T @ y_norm)

            def predict_fourier(X, F_fn, w, y_mean, y_std):
                F_test = F_fn(X)
                return y_mean + y_std * (F_test @ w)

            update_best(lambda X: predict_fourier(X, make_fourier_features, w, y_mean, y_std))
        except Exception:
            pass

    # Strategy 4: RBF with adaptive centers
    def make_rbf_model(gamma, n_centers):
        if n > n_centers:
            idx = np.random.choice(n, n_centers, replace=False)
            centers = X_train[idx]
        else:
            centers = X_train

        diffs = X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F = np.exp(-gamma * dists)

        lam = 1e-5
        A = F.T @ F + lam * np.eye(F.shape[1])
        A[0, 0] -= lam
        w = solve(A, F.T @ y_norm)

        def predict_rbf(X, centers, gamma, w, y_mean, y_std):
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            F_test = np.exp(-gamma * dists)
            return y_mean + y_std * (F_test @ w)

        return lambda X: predict_rbf(X, centers, gamma, w, y_mean, y_std)

    # Estimate scale for gamma
    X_range_mean = np.mean(X_range)
    base_gamma = 1.0 / (d * X_range_mean ** 2 + 1e-10)

    for g_mult in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if not check_time():
            break
        try:
            model = make_rbf_model(base_gamma * g_mult, min(50, n))
            update_best(model)
        except Exception:
            pass

    # Strategy 5: Neural network-like with ReLU activation
    def make_relu_model(n_hidden, alpha):
        # Simple feedforward with 1 hidden layer
        np.random.seed(42)
        W1 = np.random.randn(d, n_hidden) * alpha
        b1 = np.zeros(n_hidden)

        def forward(Xn):
            z1 = Xn @ W1 + b1
            a1 = np.maximum(0, z1)
            return a1

        F = forward((X_train - X_min) / X_range)
        F = np.column_stack([np.ones(len(F)), F])

        lam = 1e-5
        A = F.T @ F + lam * np.eye(F.shape[1])
        A[0, 0] -= lam
        w = solve(A, F.T @ y_norm)

        def predict_relu(X, W1, b1, w, y_mean, y_std):
            Xn = (X - X_min) / X_range
            z1 = Xn @ W1 + b1
            a1 = np.maximum(0, z1)
            F_test = np.column_stack([np.ones(len(a1)), a1])
            return y_mean + y_std * (F_test @ w)

        return lambda X: predict_relu(X, W1, b1, w, y_mean, y_std)

    for n_hid in [10, 20, 50]:
        if not check_time():
            break
        try:
            model = make_relu_model(n_hid, 1.0)
            update_best(model)
        except Exception:
            pass

    # Strategy 6: Try to detect simple functional forms
    # Check if y is a simple function of one or two variables
    for i in range(d):
        if not check_time():
            break
        # Try fitting y as f(X[:, i]) with polynomial
        try:
            x1 = X_train[:, i]
            # Fit polynomial of degree 5
            coeffs = np.polyfit(x1, y_norm, 5)
            def predict_1d(X, col, coeffs, y_mean, y_std):
                return y_mean + y_std * np.polyval(coeffs, X[:, col])
            update_best(lambda X, c=i, co=coeffs: predict_1d(X, c, co, y_mean, y_std))
        except Exception:
            pass

    # Check pairwise interactions
    for i in range(d):
        for j in range(i + 1, d):
            if not check_time():
                break
            try:
                x1 = X_train[:, i]
                x2 = X_train[:, j]
                F = np.column_stack([
                    np.ones(n),
                    x1, x2,
                    x1 * x2,
                    x1 ** 2, x2 ** 2,
                    x1 ** 3, x2 ** 3,
                    x1 ** 2 * x2, x1 * x2 ** 2
                ])
                lam = 1e-5
                A = F.T @ F + lam * np.eye(F.shape[1])
                A[0, 0] -= lam
                w = solve(A, F.T @ y_norm)

                def predict_pair(X, i, j, w, y_mean, y_std):
                    x1 = X[:, i]
                    x2 = X[:, j]
                    F_test = np.column_stack([
                        np.ones(len(x1)),
                        x1, x2,
                        x1 * x2,
                        x1 ** 2, x2 ** 2,
                        x1 ** 3, x2 ** 3,
                        x1 ** 2 * x2, x1 * x2 ** 2
                    ])
                    return y_mean + y_std * (F_test @ w)

                update_best(lambda X, i=i, j=j, w=w: predict_pair(X, i, j, w, y_mean, y_std))
            except Exception:
                pass

    # Fallback
    if best_predict is None:
        return lambda X: np.full(len(X), y_mean)

    return best_predict

# EVOLVE-BLOCK-END
