"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280  # safety margin before 300s limit

    X = X_train
    y = y_train
    n, d = X.shape

    # Normalize inputs to [0, 1] for better kernel behavior
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range < 1e-12] = 1.0

    def normalize(X_raw):
        return (X_raw - x_min) / x_range

    Xn = normalize(X)

    # Try multiple RBF bandwidths and ridge parameters via CV
    best_score = np.inf
    best_predict = None

    # Candidate bandwidths (gamma for RBF: exp(-gamma * ||xi-xj||^2))
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    # Candidate ridge alphas
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]

    # Also try polynomial kernel as an alternative
    poly_degrees = [2, 3, 4, 5, 6]
    poly_coefs = [0.1, 1.0, 10.0]

    # Cross-validation: 5-fold
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)

    for gamma in gammas:
        if time.time() > deadline:
            break
        # Precompute training kernel for this gamma
        K_train = np.exp(-gamma * np.sum((Xn[:, None, :] - Xn[None, :, :])**2, axis=2))

        for alpha in alphas:
            if time.time() > deadline:
                break
            # CV score
            fold_errors = []
            for fold in range(n_folds):
                val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
                trn_idx = np.delete(indices, val_idx)

                K_val = np.exp(-gamma * np.sum((Xn[val_idx][:, None, :] - Xn[trn_idx][None, :, :])**2, axis=2))
                K_trn = K_train[np.ix_(trn_idx, trn_idx)]

                A = K_trn + alpha * np.eye(len(trn_idx))
                try:
                    w = np.linalg.solve(A, y[trn_idx])
                    pred_val = K_val @ w
                    fold_errors.append(np.mean((pred_val - y[val_idx])**2))
                except np.linalg.LinAlgError:
                    fold_errors.append(np.inf)

            cv_mse = np.mean(fold_errors)
            if cv_mse < best_score:
                best_score = cv_mse
                # Fit on all data with best params
                A = K_train + alpha * np.eye(n)
                try:
                    w_all = np.linalg.solve(A, y)
                    best_predict = (lambda X_raw, Xn_raw=Xn, w=w_all, g=gamma:
                                    np.exp(-g * np.sum((normalize(X_raw)[:, None, :] - Xn_raw[None, :, :])**2, axis=2)) @ w)
                except np.linalg.LinAlgError:
                    pass

    # Try polynomial kernels
    for deg in poly_degrees:
        if time.time() > deadline:
            break
        for coef in poly_coefs:
            if time.time() > deadline:
                break
            for alpha in [1e-3, 1e-2, 0.1, 1.0]:
                if time.time() > deadline:
                    break
                # Polynomial kernel: (coef * xi.xj + 1)^deg
                K_train = (coef * Xn @ Xn.T + 1.0) ** deg

                fold_errors = []
                for fold in range(n_folds):
                    val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
                    trn_idx = np.delete(indices, val_idx)

                    K_val = (coef * Xn[val_idx] @ Xn[trn_idx].T + 1.0) ** deg
                    K_trn = K_train[np.ix_(trn_idx, trn_idx)]

                    A = K_trn + alpha * np.eye(len(trn_idx))
                    try:
                        w = np.linalg.solve(A, y[trn_idx])
                        pred_val = K_val @ w
                        fold_errors.append(np.mean((pred_val - y[val_idx])**2))
                    except np.linalg.LinAlgError:
                        fold_errors.append(np.inf)

                cv_mse = np.mean(fold_errors)
                if cv_mse < best_score:
                    best_score = cv_mse
                    A = K_train + alpha * np.eye(n)
                    try:
                        w_all = np.linalg.solve(A, y)
                        best_predict = (lambda X_raw, w=w_all, c=coef, dg=deg:
                                        (c * normalize(X_raw) @ Xn.T + 1.0) ** dg @ w)
                    except np.linalg.LinAlgError:
                        pass

    # Fallback: if no valid model found, use simple mean
    if best_predict is None:
        y_mean = np.mean(y)
        return lambda X: np.full(X.shape[0], y_mean)

    return best_predict
# EVOLVE-BLOCK-END
