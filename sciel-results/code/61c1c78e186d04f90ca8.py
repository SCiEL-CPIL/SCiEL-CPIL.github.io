"""Kernel ridge regression with composite kernels for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 270.0  # safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Center and scale X for stable kernel computation
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_scaled = (X - X_mean) / X_std

    def compute_kernel(Xs, length_scale, poly_degree, poly_gamma, periodic_period,
                       rbf_weight, poly_weight, periodic_weight, linear_weight):
        """Compute composite kernel matrix."""
        # Squared distances
        diff = Xs[:, None, :] - Xs[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)

        # RBF kernel
        K_rbf = np.exp(-sq_dist / (2.0 * length_scale ** 2))

        # Polynomial kernel
        K_poly = (1.0 + poly_gamma * sq_dist * 0.5) ** poly_degree

        # Periodic kernel
        K_periodic = np.exp(-2.0 * np.pi ** 2 * (1.0 - np.cos(2.0 * np.pi * sq_dist / (2.0 * periodic_period ** 2))) / (2.0 * length_scale ** 2))

        # Linear kernel
        K_linear = Xs @ Xs.T

        return (rbf_weight * K_rbf + poly_weight * K_poly +
                periodic_weight * K_periodic + linear_weight * K_linear)

    # Grid search over hyperparameters
    length_scales = [0.3, 0.5, 1.0, 2.0, 3.0]
    poly_degrees = [2, 3, 4]
    poly_gammas = [0.5, 1.0, 2.0]
    periodic_periods = [2.0, 3.0, 5.0, 10.0]
    rbf_weights = [0.5, 1.0, 2.0]
    poly_weights = [0.1, 0.5, 1.0]
    periodic_weights = [0.0, 0.1, 0.5]
    linear_weights = [0.1, 0.5, 1.0]
    ridge_alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]

    best_val = np.inf
    best_params = None

    # Use validation split for hyperparameter selection
    # Simple KFold implementation to avoid sklearn dependency
    def kfold_split(n, k, seed=42):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        folds = np.array_split(indices, k)
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            yield train_idx, val_idx
    
    kf_splits = list(kfold_split(n, 5))

    # To keep computation manageable, use a reduced grid
    # First pass: coarse grid
    for ls in length_scales:
        for pd_ in poly_degrees:
            for pg in poly_gammas:
                for pp in periodic_periods:
                    for rw in rbf_weights:
                        for pw in poly_weights:
                            for kpw in periodic_weights:
                                for lw in linear_weights:
                                    for alpha in ridge_alphas:
                                        if time.time() > deadline - 30:
                                            break
                                    if time.time() > deadline - 30:
                                        break
                                if time.time() > deadline - 30:
                                    break
                            if time.time() > deadline - 30:
                                break
                        if time.time() > deadline - 30:
                            break
                    if time.time() > deadline - 30:
                        break
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break

        if time.time() > deadline - 30:
            break

    # The full grid is too large; use a smarter approach
    # Evaluate a subset of hyperparameter combinations
    candidate_params = []
    for ls in [0.5, 1.0, 2.0]:
        for pd_ in [2, 3]:
            for pg in [1.0]:
                for pp in [3.0, 5.0]:
                    for rw in [1.0]:
                        for pw in [0.5]:
                            for kpw in [0.0, 0.1]:
                                for lw in [0.5]:
                                    for alpha in [1e-3, 1e-2, 0.1]:
                                        candidate_params.append((ls, pd_, pg, pp, rw, pw, kpw, lw, alpha))

    for params in candidate_params:
        ls, pd_, pg, pp, rw, pw, kpw, lw, alpha = params
        if time.time() > deadline - 20:
            break

        # Cross-validation
        cv_errors = []
        for train_idx, val_idx in kf_splits:
            X_tr = X_scaled[train_idx]
            y_tr = y[train_idx]
            X_val = X_scaled[val_idx]
            y_val = y[val_idx]

            K_tr = compute_kernel(X_tr, ls, pd_, pg, pp, rw, pw, kpw, lw)
            K_val = compute_kernel(X_val, ls, pd_, pg, pp, rw, pw, kpw, lw)

            try:
                alpha_mat = alpha * np.eye(len(X_tr))
                alpha_mat[0, 0] += 1e-10  # small regularization for numerical stability
                coeffs = np.linalg.solve(K_tr + alpha_mat, y_tr)
                pred_val = K_val @ coeffs
                mse = np.mean((pred_val - y_val) ** 2)
                var_y = np.var(y_val)
                if var_y < 1e-10:
                    nmse = mse
                else:
                    nmse = mse / var_y
                cv_errors.append(nmse)
            except np.linalg.LinAlgError:
                cv_errors.append(np.inf)

        mean_cv = np.mean(cv_errors)
        if mean_cv < best_val:
            best_val = mean_cv
            best_params = params

    # If no valid params found, use defaults
    if best_params is None:
        best_params = (1.0, 2, 1.0, 5.0, 1.0, 0.5, 0.0, 0.5, 1e-2)

    ls, pd_, pg, pp, rw, pw, kpw, lw, alpha = best_params

    # Fit on all data with best hyperparameters
    K_full = compute_kernel(X_scaled, ls, pd_, pg, pp, rw, pw, kpw, lw)
    alpha_mat = alpha * np.eye(n)
    alpha_mat[0, 0] += 1e-10
    try:
        coeffs = np.linalg.solve(K_full + alpha_mat, y)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        coeffs = np.linalg.lstsq(K_full + alpha_mat, y, rcond=None)[0]

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_scaled = (X_test - X_mean) / X_std
        K_test = compute_kernel(X_test_scaled, ls, pd_, pg, pp, rw, pw, kpw, lw)
        preds = K_test @ coeffs
        # Ensure finite predictions
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds

    return predict
# EVOLVE-BLOCK-END
