# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


def _rbf_kernel(X1, X2, gamma):
    X1_sq = np.sum(X1 ** 2, axis=1)
    X2_sq = np.sum(X2 ** 2, axis=1)
    sq_dists = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-gamma * sq_dists)


def _poly_kernel(X1, X2, degree):
    return (1 + X1 @ X2.T) ** degree


def _matern52_kernel(X1, X2, length_scale, variance):
    X1_sq = np.sum(X1 ** 2, axis=1)
    X2_sq = np.sum(X2 ** 2, axis=1)
    sq_dists = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    dists = np.sqrt(sq_dists)
    r = dists / length_scale
    return variance * (1 + r + r ** 2 / 3) * np.exp(-r)


def _loocv_ridge(X, y, kernel_fn, alpha_grid):
    n = len(y)
    K = kernel_fn(X, X)
    best_mse = np.inf
    best_alpha = None
    for alpha in alpha_grid:
        K_reg = K + alpha * np.eye(n)
        try:
            L = np.linalg.cholesky(K_reg)
            alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y))
            # LOO: yhat_i = y_i - K_ii / h_ii where h_ii = ((K+alpha I)^{-1})_{ii}
            # h_ii = 1 / (L^T L)_{ii}... actually h_ii = 1/(L L^T)_{ii} is not right
            # (K+alpha I)^{-1} = (L L^T)^{-1} = (L^T)^{-1} L^{-1}
            # diag of (K+alpha I)^{-1} = diag of (L^T)^{-1} L^{-1}
            # Let w = L^{-1} 1 (solving L w = 1), then (L^T)^{-1} w = (L^T L)^{-1} 1... no
            # Actually: if L L^T = K_reg, then (K_reg)^{-1} = (L^T)^{-1} L^{-1}
            # The diagonal of (K_reg)^{-1}: let v_i be the i-th column of L^{-1}, then
            # ((K_reg)^{-1})_{ii} = v_i^T (L^T)^{-T} ... this is getting circular.
            # Simpler: solve (K_reg) z = e_i for each i? No, too slow.
            # Use: (K_reg)^{-1} = (L L^T)^{-1} = (L^T)^{-1} L^{-1}
            # Let A = L^{-1} (so L A = I). Then (K_reg)^{-1} = A^T A... no, that's (L^T L)^{-1}.
            # Wait: K_reg = L L^T. (K_reg)^{-1} = (L^T)^{-1} L^{-1}.
            # Let B = L^{-1}. Then (K_reg)^{-1} = B^T B? No: (L^T)^{-1} = (L^{-1})^T = B^T.
            # So (K_reg)^{-1} = B^T B. But B^T B is (L^{-1})^T L^{-1} = (L L^T)^{-1}. Yes!
            # So diag((K_reg)^{-1}) = diag(B^T B) where B = L^{-1}.
            # B^T B has diagonal: sum of squares of each row of B.
            B = np.linalg.solve(L, np.eye(n))
            diag_inv = np.sum(B ** 2, axis=0)
            yhat_loo = y - np.diag(K) * diag_inv
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except np.linalg.LinAlgError:
            continue
    return best_mse, best_alpha, K


def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    # Degree 1
    for j in range(d):
        cols.append(X[:, j])
    # Degree 2
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Degree 3
    if degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)


def _loocv_poly_ridge(X, y, degree, alpha_grid):
    n = len(y)
    B = _poly_features(X, degree)
    m = B.shape[1]
    best_mse = np.inf
    best_alpha = None
    for alpha in alpha_grid:
        A = B.T @ B + alpha * np.eye(m)
        try:
            w = np.linalg.solve(A, B.T @ y)
            yhat = B @ w
            # LOO for ridge: yhat_loo_i = y_i - H_ii/(1-H_ii) * (y_i - yhat_i)
            # H = B (B^T B + alpha I)^{-1} B^T
            # Instead of computing full H, use: H_ii = x_i^T (B^T B + alpha I)^{-1} x_i
            # Let's compute (B^T B + alpha I)^{-1} once
            A_inv = np.linalg.inv(A)
            H_diag = np.einsum('ij,ji->i', B, A_inv @ B.T)
            # Actually H = B A^{-1} B^T, so H_ii = B_i^T A^{-1} B_i
            H_diag = np.einsum('ij,jk,ki->i', B, A_inv, B)
            mask = np.abs(1 - H_diag) > 1e-10
            yhat_loo = y.copy()
            yhat_loo[mask] = y[mask] - (H_diag[mask] / (1 - H_diag[mask])) * (y[mask] - yhat[mask])
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except np.linalg.LinAlgError:
            continue
    return best_mse, best_alpha, B


def _loocv_gp(X, y, length_scale_grid, variance_grid, noise_grid):
    n = len(y)
    best_mse = np.inf
    best_params = None
    for ls in length_scale_grid:
        for var in variance_grid:
            for noise in noise_grid:
                K_prior = _matern52_kernel(X, X, ls, var)
                K_obs = K_prior + noise * np.eye(n)
                try:
                    L = np.linalg.cholesky(K_obs)
                    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    # LOO for GP: yhat_i = y_i - K_prior_ii * (K_obs^{-1})_{ii}
                    # (K_obs^{-1})_{ii}: K_obs = L L^T, so (K_obs)^{-1} = (L^T)^{-1} L^{-1}
                    B = np.linalg.solve(L, np.eye(n))
                    diag_inv = np.sum(B ** 2, axis=0)
                    yhat_loo = y - var * diag_inv
                    mse = np.mean((y - yhat_loo) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_params = (ls, var, noise)
                except (np.linalg.LinAlgError, ValueError):
                    continue
    return best_mse, best_params


def fit(X_train, y_train):
    X = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std

    models = []
    cv_scores = []

    # Model 1: RBF Kernel Ridge Regression
    gamma_grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    alpha_grid_krr = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_mse1 = np.inf
    best_gamma1 = None
    best_alpha1 = None
    best_w1 = None
    for gamma in gamma_grid:
        mse_g, alpha_g, K_g = _loocv_ridge(X_norm, y, lambda a, b, g=gamma: _rbf_kernel(a, b, g), alpha_grid_krr)
        if mse_g < best_mse1:
            best_mse1 = mse_g
            best_gamma1 = gamma
            best_alpha1 = alpha_g
            K_reg = K_g + alpha_g * np.eye(n)
            best_w1 = np.linalg.solve(K_reg, y)

    if best_mse1 < np.inf and best_w1 is not None:
        def rbf_predict(X_test):
            X_test = (np.array(X_test, dtype=np.float64) - X_mean) / X_std
            K_test = _rbf_kernel(X_test, X_norm, best_gamma1)
            return K_test @ best_w1
        models.append(rbf_predict)
        cv_scores.append(best_mse1)

    # Model 2: Polynomial Kernel Ridge Regression (degree 3)
    best_mse2 = np.inf
    best_alpha2 = None
    best_w2 = None
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        mse_a, _, K_a = _loocv_ridge(X_norm, y, lambda a, b: _poly_kernel(a, b, 3), [alpha])
        if mse_a < best_mse2:
            best_mse2 = mse_a
            best_alpha2 = alpha
            K_reg = K_a + alpha * np.eye(n)
            best_w2 = np.linalg.solve(K_reg, y)

    if best_mse2 < np.inf and best_w2 is not None:
        def poly_krr_predict(X_test):
            X_test = (np.array(X_test, dtype=np.float64) - X_mean) / X_std
            K_test = _poly_kernel(X_test, X_norm, 3)
            return K_test @ best_w2
        models.append(poly_krr_predict)
        cv_scores.append(best_mse2)

    # Model 3: Polynomial feature expansion with ridge (degree 2)
    mse3, alpha3, B3 = _loocv_poly_ridge(X_norm, y, 2, [0.001, 0.01, 0.1, 1.0])
    if mse3 < np.inf and alpha3 is not None:
        A3 = B3.T @ B3 + alpha3 * np.eye(B3.shape[1])
        w3 = np.linalg.solve(A3, B3.T @ y)
        def poly2_predict(X_test):
            X_test = (np.array(X_test, dtype=np.float64) - X_mean) / X_std
            B_test = _poly_features(X_test, 2)
            return B_test @ w3
        models.append(poly2_predict)
        cv_scores.append(mse3)

    # Model 4: Polynomial feature expansion with ridge (degree 3)
    mse4, alpha4, B4 = _loocv_poly_ridge(X_norm, y, 3, [0.001, 0.01, 0.1, 1.0])
    if mse4 < np.inf and alpha4 is not None:
        A4 = B4.T @ B4 + alpha4 * np.eye(B4.shape[1])
        w4 = np.linalg.solve(A4, B4.T @ y)
        def poly3_predict(X_test):
            X_test = (np.array(X_test, dtype=np.float64) - X_mean) / X_std
            B_test = _poly_features(X_test, 3)
            return B_test @ w4
        models.append(poly3_predict)
        cv_scores.append(mse4)

    # Model 5: Matern-5/2 GP
    ls_grid = [0.1, 0.3, 0.5, 1.0, 2.0]
    var_grid = [0.5, 1.0, 2.0]
    noise_grid = [0.01, 0.1]
    mse5, gp_params = _loocv_gp(X_norm, y, ls_grid, var_grid, noise_grid)
    if mse5 < np.inf and gp_params is not None:
        ls, var, noise = gp_params
        K_prior = _matern52_kernel(X_norm, X_norm, ls, var)
        K_obs = K_prior + noise * np.eye(n)
        L = np.linalg.cholesky(K_obs)
        alpha_gp = np.linalg.solve(L.T, np.linalg.solve(L, y))

        def gp_predict(X_test):
            X_test = (np.array(X_test, dtype=np.float64) - X_mean) / X_std
            K_test = _matern52_kernel(X_test, X_norm, ls, var)
            return K_test @ alpha_gp
        models.append(gp_predict)
        cv_scores.append(mse5)

    # Model 6: Linear ridge regression
    B6 = np.column_stack([np.ones(n), X_norm])
    m6 = B6.shape[1]
    best_mse6 = np.inf
    best_alpha6 = None
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        A = B6.T @ B6 + alpha * np.eye(m6)
        w = np.linalg.solve(A, B6.T @ y)
        yhat = B6 @ w
        A_inv = np.linalg.inv(A)
        H_diag = np.einsum('ij,jk,ki->i', B6, A_inv, B6)
        mask = np.abs(1 - H_diag) > 1e-10
        yhat_loo = y.copy()
        yhat_loo[mask] = y[mask] - (H_diag[mask] / (1 - H_diag[mask])) * (y[mask] - yhat[mask])
        mse = np.mean((y - yhat_loo) ** 2)
        if mse < best_mse6:
            best_mse6 = mse
            best_alpha6 = alpha

    if best_mse6 < np.inf and best_alpha6 is not None:
        A6 = B6.T @ B6 + best_alpha6 * np.eye(m6)
        w6 = np.linalg.solve(A6, B6.T @ y)
        def linear_predict(X_test):
            X_test = (np.array(X_test, dtype=np.float64) - X_mean) / X_std
            B_test = np.column_stack([np.ones(len(X_test)), X_test])
            return B_test @ w6
        models.append(linear_predict)
        cv_scores.append(best_mse6)

    # Select best model based on LOO CV score
    if not models:
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)

    best_idx = int(np.argmin(cv_scores))
    best_model = models[best_idx]

    def predict(X_test):
        X_test = np.array(X_test, dtype=np.float64)
        try:
            preds = best_model(X_test)
            preds = np.asarray(preds, dtype=np.float64)
            preds = np.where(np.isfinite(preds), preds, np.mean(y))
            return preds
        except Exception:
            return np.full(len(X_test), np.mean(y))

    return predict
# EVOLVE-BLOCK-END
