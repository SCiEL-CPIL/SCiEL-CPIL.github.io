import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if n < 10:
        mean_y = np.mean(y) if n > 0 else 0.0
        return np.array([mean_y, 0, 0, 0, 0, 0], dtype=np.float64)

    # Scale x to [-1, 1] for better conditioning
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x_min - 0.5
        x_max = x_min + 0.5

    # Linear map: x_scaled = A * x + B
    A = 2.0 / x_range
    B = - (x_max + x_min) / x_range
    x_scaled = A * x + B

    # Model selection via 5-Fold Cross Validation
    # Use eigen-decomposition to batch all alpha ridge solves per fold
    degrees = [0, 1, 2, 3, 4, 5]
    alphas = np.logspace(-8, 4, 40)

    k_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // k_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(k_folds)]
    if len(folds[-1]) < fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    best_cv_mse = np.inf
    best_degree = 0
    best_alpha = 1.0

    for deg in degrees:
        if time.time() > deadline:
            break

        # Build Vandermonde matrix for all data
        V_all = np.vander(x_scaled, deg + 1, increasing=True)

        # Accumulate MSE for each alpha across folds
        alpha_mse_sums = np.zeros(len(alphas))

        for i in range(k_folds):
            if time.time() > deadline:
                break

            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            V_train = V_all[train_idx]
            y_train = y[train_idx]
            V_val = V_all[val_idx]
            y_val = y[val_idx]

            # Compute Gram matrix and cross-product once
            G = V_train.T @ V_train  # (d+1) x (d+1)
            g = V_train.T @ y_train  # (d+1)

            # Eigen-decompose G for batched ridge solves
            # G is symmetric PSD
            try:
                eigvals, eigvecs = np.linalg.eigh(G)
            except np.linalg.LinAlgError:
                continue

            # For ridge: (G + alpha*I)^{-1} g
            # = eigvecs @ diag(1/(eigvals+alpha)) @ (eigvecs.T @ g)
            eigvecs_T_g = eigvecs.T @ g  # (d+1)

            # For each alpha, theta = eigvecs @ (eigvecs_T_g / (eigvals + alpha))
            # Batch: shape (d+1, len(alphas))
            denom = eigvals[:, np.newaxis] + alphas[np.newaxis, :]  # (d+1, len(alphas))
            theta_batch = eigvecs @ (eigvecs_T_g[:, np.newaxis] / denom)  # (d+1, len(alphas))

            # Predictions for validation: V_val @ theta_batch
            # V_val is (n_val, d+1), theta_batch is (d+1, len(alphas))
            y_pred_batch = V_val @ theta_batch  # (n_val, len(alphas))

            # MSE for each alpha
            residuals = y_pred_batch - y_val[:, np.newaxis]
            mse_per_alpha = np.mean(residuals ** 2, axis=0)  # (len(alphas),)

            alpha_mse_sums += mse_per_alpha

        if time.time() > deadline:
            break

        min_idx = np.argmin(alpha_mse_sums)
        avg_cv_mse = alpha_mse_sums[min_idx] / k_folds

        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_degree = deg
            best_alpha = alphas[min_idx]

    # Final fit on all data with best_degree and best_alpha
    V_all = np.vander(x_scaled, best_degree + 1, increasing=True)
    reg_matrix = np.eye(best_degree + 1) * best_alpha
    A_mat = V_all.T @ V_all + reg_matrix
    b_vec = V_all.T @ y

    try:
        theta_scaled = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        theta_scaled = np.zeros(best_degree + 1)

    # Convert coefficients from scaled monomial basis to original monomial basis
    # P(x_scaled) = sum_{i=0}^{d} theta_scaled[i] * x_scaled^i
    # x_scaled = A * x + B
    # P(x) = sum_{i=0}^{d} theta_scaled[i] * (A * x + B)^i

    result_poly = np.zeros(6, dtype=np.float64)

    # (A*x + B) is represented by coeffs [B, A] in ascending order
    base_poly = np.array([B, A], dtype=np.float64)

    current_power = np.array([1.0], dtype=np.float64)  # (Ax+B)^0

    for i in range(best_degree + 1):
        # Add theta_scaled[i] * current_power to result_poly
        if len(current_power) < 6:
            cp_padded = np.pad(current_power, (0, 6 - len(current_power)), mode='constant')
        else:
            cp_padded = current_power[:6]

        result_poly += theta_scaled[i] * cp_padded

        # Update current_power for next iteration
        if i < best_degree:
            current_power = np.polynomial.polynomial.polymul(current_power, base_poly)
            if len(current_power) > 6:
                current_power = current_power[:6]

    # Ensure finite
    if not np.all(np.isfinite(result_poly)):
        result_poly = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    return result_poly
