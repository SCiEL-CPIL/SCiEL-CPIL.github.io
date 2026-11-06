"""Strategy: Regularized polynomial regression with degree selection via internal CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 10:
        return np.zeros(6)

    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Try degrees 1 through 5 with ridge regularization
    # Use internal k-fold cross-validation to select best degree and lambda
    best_score = np.inf
    best_coeff = np.zeros(6)

    # Cross-validation setup
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    fold_size = n // n_folds

    for degree in range(1, 6):
        if time.time() > deadline - 30:
            break

        # Try several lambda values
        lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]

        for lam in lambdas:
            if time.time() > deadline - 30:
                break

            cv_errors = []
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_folds - 1 else n
                train_mask = np.ones(n, dtype=bool)
                train_mask[val_start:val_end] = False

                x_train = x_norm[train_mask]
                y_train = y[train_mask]
                x_val = x_norm[val_start:val_end]
                y_val = y[val_start:val_end]

                # Build Vandermonde matrix (in normalized x)
                V_train = np.vander(x_train, N=degree + 1, increasing=True).T
                V_val = np.vander(x_val, N=degree + 1, increasing=True).T

                # Ridge regression: (V^T V + lam*I) c = V^T y
                A = V_train.T @ V_train + lam * np.eye(degree + 1)
                b = V_train.T @ y_train
                try:
                    c_norm = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue

                y_pred = V_val @ c_norm
                mse = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(mse)

            if len(cv_errors) == 0:
                continue

            avg_cv_error = np.mean(cv_errors)
            if avg_cv_error < best_score:
                best_score = avg_cv_error
                # Refit on all data with best parameters
                V_all = np.vander(x_norm, N=degree + 1, increasing=True).T
                A_all = V_all.T @ V_all + lam * np.eye(degree + 1)
                b_all = V_all.T @ y
                try:
                    c_norm_best = np.linalg.solve(A_all, b_all)
                except np.linalg.LinAlgError:
                    continue

                # Convert from normalized x to original x
                # x_norm = (x - x_mean) / x_std
                # We need coefficients for polynomial in x: sum(c_i * x^i)
                # c_norm gives coefficients for sum(d_j * x_norm^j)
                # x_norm^j = ((x - x_mean)/x_std)^j = sum_{k=0}^{j} binom(j,k) * (-x_mean/x_std)^{j-k} * x^k / x_std^k
                # So d_j * x_norm^j = d_j * sum_{k=0}^{j} binom(j,k) * (-x_mean/x_std)^{j-k} * x^k / x_std^k
                # Coefficient of x^k: sum_{j=k}^{degree} d_j * binom(j,k) * (-x_mean/x_std)^{j-k} / x_std^k

                c_orig = np.zeros(degree + 1)
                for k in range(degree + 1):
                    for j in range(k, degree + 1):
                        # binom(j, k)
                        from math import comb
                        c_orig[k] += c_norm_best[j] * comb(j, k) * ((-x_mean / x_std) ** (j - k)) / (x_std ** k)

                # Pad to length 6
                coeff_full = np.zeros(6)
                coeff_full[:degree + 1] = c_orig
                best_coeff = coeff_full

    return best_coeff
# EVOLVE-BLOCK-END
