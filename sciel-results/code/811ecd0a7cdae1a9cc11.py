import numpy as np
import time

def _fista_lasso(A_mat, b_vec, lam, max_iter=3000, tol=1e-7):
    """FISTA (Fast Iterative Shrinkage-Thresholding) for Lasso: min 0.5*||y - A@theta||^2 + lam*||theta||_1"""
    n_features = A_mat.shape[1]
    theta = np.zeros(n_features)
    theta_prev = np.zeros(n_features)
    
    # Compute Lipschitz constant
    try:
        L = np.linalg.norm(A_mat, 2) ** 2
    except:
        L = np.trace(A_mat.T @ A_mat) / n_features
    L = max(L, 1e-10)
    step = 1.0 / L
    
    t = 1.0
    for it in range(max_iter):
        # Gradient step
        grad = A_mat.T @ (A_mat @ theta - b_vec)
        z = theta - step * grad
        
        # Proximal (soft-thresholding)
        theta_new = np.sign(z) * np.maximum(np.abs(z) - lam * step, 0)
        
        # FISTA update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        if it > 0:
            theta = theta_new + ((t - 1.0) / t_new) * (theta_new - theta_prev)
        
        # Convergence check
        if np.linalg.norm(theta - theta_prev) < tol * max(1.0, np.linalg.norm(theta)):
            break
        
        theta_prev = theta_new
        t = t_new
    
    return theta

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
    A_scale = 2.0 / x_range
    B_scale = - (x_max + x_min) / x_range
    x_scaled = A_scale * x + B_scale

    # Set up cross-validation folds
    k_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // k_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(k_folds)]
    if len(folds[-1]) < fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    # Pre-compute Vandermonde matrix for degree 5
    V_all = np.vander(x_scaled, 6, increasing=True)  # degree 5, 6 terms

    # Approach 1: Ridge polynomial with degree and alpha selection
    degrees = [0, 1, 2, 3, 4, 5]
    alphas_ridge = np.logspace(-6, 3, 30)

    best_ridge_cv_mse = np.inf
    best_ridge_degree = 0
    best_ridge_alpha = 1.0

    for deg in degrees:
        if time.time() > deadline:
            break

        V_deg = V_all[:, :deg + 1]
        alpha_mse_sums = np.zeros(len(alphas_ridge))

        for i in range(k_folds):
            if time.time() > deadline:
                break

            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            V_train = V_deg[train_idx]
            y_train = y[train_idx]
            V_val = V_deg[val_idx]
            y_val = y[val_idx]

            G = V_train.T @ V_train
            g = V_train.T @ y_train

            try:
                eigvals, eigvecs = np.linalg.eigh(G)
            except np.linalg.LinAlgError:
                continue

            eigvecs_T_g = eigvecs.T @ g
            denom = eigvals[:, np.newaxis] + alphas_ridge[np.newaxis, :]
            theta_batch = eigvecs @ (eigvecs_T_g[:, np.newaxis] / denom)
            y_pred_batch = V_val @ theta_batch
            residuals = y_pred_batch - y_val[:, np.newaxis]
            mse_per_alpha = np.mean(residuals ** 2, axis=0)
            alpha_mse_sums += mse_per_alpha

        if time.time() > deadline:
            break

        min_idx = np.argmin(alpha_mse_sums)
        avg_cv_mse = alpha_mse_sums[min_idx] / k_folds

        if avg_cv_mse < best_ridge_cv_mse:
            best_ridge_cv_mse = avg_cv_mse
            best_ridge_degree = deg
            best_ridge_alpha = alphas_ridge[min_idx]

    # Approach 2: Lasso on full degree-5 polynomial basis (sparse selection)
    lams = np.logspace(-4, 2, 25)

    best_lasso_cv_mse = np.inf
    best_lasso_lam = 1.0

    for lam in lams:
        if time.time() > deadline:
            break

        lam_mse_sum = 0.0
        for i in range(k_folds):
            if time.time() > deadline:
                break

            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            A_train = V_all[train_idx]
            y_train = y[train_idx]
            A_val = V_all[val_idx]
            y_val = y[val_idx]

            # Solve Lasso via FISTA
            theta_lasso = _fista_lasso(A_train, y_train, lam, max_iter=2000, tol=1e-6)

            y_pred = A_val @ theta_lasso
            mse = np.mean((y_pred - y_val) ** 2)
            lam_mse_sum += mse

        if time.time() > deadline:
            break

        avg_cv_mse = lam_mse_sum / k_folds
        if avg_cv_mse < best_lasso_cv_mse:
            best_lasso_cv_mse = avg_cv_mse
            best_lasso_lam = lam

    # Choose the better approach
    if best_lasso_cv_mse < best_ridge_cv_mse:
        # Use Lasso solution
        A_mat = V_all
        b_vec = y
        theta_scaled = _fista_lasso(A_mat, b_vec, best_lasso_lam, max_iter=5000, tol=1e-7)
        best_degree = 5
    else:
        # Use Ridge solution
        V_deg = V_all[:, :best_ridge_degree + 1]
        reg_matrix = np.eye(best_ridge_degree + 1) * best_ridge_alpha
        A_mat = V_deg.T @ V_deg + reg_matrix
        b_vec = V_deg.T @ y

        try:
            theta_scaled = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            theta_scaled = np.zeros(best_ridge_degree + 1)
        best_degree = best_ridge_degree

    # Convert coefficients from scaled monomial basis to original monomial basis
    result_poly = np.zeros(6, dtype=np.float64)

    # (A*x + B) is represented by coeffs [B, A] in ascending order
    base_poly = np.array([B_scale, A_scale], dtype=np.float64)

    current_power = np.array([1.0], dtype=np.float64)  # (Ax+B)^0

    for i in range(best_degree + 1):
        if len(current_power) < 6:
            cp_padded = np.pad(current_power, (0, 6 - len(current_power)), mode='constant')
        else:
            cp_padded = current_power[:6]

        result_poly += theta_scaled[i] * cp_padded

        if i < best_degree:
            current_power = np.polynomial.polynomial.polymul(current_power, base_poly)
            if len(current_power) > 6:
                current_power = current_power[:6]

    # Ensure finite
    if not np.all(np.isfinite(result_poly)):
        result_poly = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    return result_poly
