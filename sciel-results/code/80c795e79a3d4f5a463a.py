"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale
    forcing of a two-scale Lorenz-96 model.
    
    Returns 6 finite real polynomial coefficients in ASCENDING powers of x.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    # Convert inputs to numpy arrays and flatten
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Check time
    if time.time() > deadline:
        return np.zeros(6, dtype=np.float64)

    # Strategy: Use Ridge regularization with cross-validation to select
    # both degree and regularization strength. This helps generalization
    # to unseen forcing regimes (F=10) by preventing overfitting.

    # For efficiency with large n, subsample if needed
    max_samples = 5000
    if n > max_samples:
        rng_sub = np.random.default_rng(0)
        idx_sub = rng_sub.choice(n, size=max_samples, replace=False)
        x_sub = x[idx_sub]
        y_sub = y[idx_sub]
    else:
        x_sub = x
        y_sub = y
        n_sub = n
    n_sub = x_sub.size

    # Define regularization strengths to try (log-spaced)
    alphas = np.logspace(-4, 4, 17)  # 17 values from 1e-4 to 1e4
    
    # K-fold cross-validation parameters
    k_folds = 5
    rng_cv = np.random.default_rng(42)
    
    # Prepare CV folds
    indices = rng_cv.permutation(n_sub)
    fold_sizes = np.full(k_folds, n_sub // k_folds)
    fold_sizes[:n_sub % k_folds] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start:start + size])
        start += size

    best_coeff = np.zeros(6, dtype=np.float64)
    best_cv_error = np.inf

    # Try degrees 0 through 5
    for degree in range(0, 6):
        if time.time() > deadline:
            break
            
        if n_sub <= degree + 1:
            continue
        
        for alpha in alphas:
            if time.time() > deadline:
                break
                
            cv_errors = []
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                x_tr = x_sub[train_idx]
                y_tr = y_sub[train_idx]
                x_va = x_sub[val_idx]
                y_va = y_sub[val_idx]
                
                # Build Vandermonde matrix
                V_tr = np.vander(x_tr, N=degree + 1, increasing=True)
                
                # Ridge regression: (V^T V + alpha*I) c = V^T y
                # Don't regularize the intercept (first column)
                A = V_tr.T @ V_tr
                A[0, 0] += 0  # No regularization on intercept
                A += alpha * np.eye(degree + 1)
                A[0, 0] -= alpha  # Remove regularization from intercept
                A[0, 0] += 1  # Ensure at least 1 on diagonal for intercept
                
                b = V_tr.T @ y_tr
                
                try:
                    coeff_cv = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    try:
                        coeff_cv = np.linalg.lstsq(A, b, rcond=None)[0]
                    except Exception:
                        continue
                
                if not np.all(np.isfinite(coeff_cv)):
                    continue
                
                # Evaluate on validation fold
                V_va = np.vander(x_va, N=degree + 1, increasing=True)
                y_pred = V_va @ coeff_cv
                
                if not np.all(np.isfinite(y_pred)):
                    continue
                
                mse = np.mean((y_va - y_pred) ** 2)
                cv_errors.append(mse)
            
            if len(cv_errors) == k_folds:
                mean_cv_error = np.mean(cv_errors)
                if mean_cv_error < best_cv_error:
                    best_cv_error = mean_cv_error
                    # Refit on all data with best alpha and degree
                    V_all = np.vander(x_sub, N=degree + 1, increasing=True)
                    A_all = V_all.T @ V_all
                    A_all += alpha * np.eye(degree + 1)
                    A_all[0, 0] -= alpha
                    A_all[0, 0] += 1
                    
                    b_all = V_all.T @ y_sub
                    
                    try:
                        best_coeff = np.linalg.solve(A_all, b_all)
                    except np.linalg.LinAlgError:
                        try:
                            best_coeff = np.linalg.lstsq(A_all, b_all, rcond=None)[0]
                        except Exception:
                            best_coeff = np.zeros(degree + 1)
                    
                    # Pad to length 6
                    if len(best_coeff) < 6:
                        padded = np.zeros(6, dtype=np.float64)
                        padded[:len(best_coeff)] = best_coeff
                        best_coeff = padded

    # Ensure final coefficients are finite
    if not np.all(np.isfinite(best_coeff)) or len(best_coeff) != 6:
        # Fallback: simple OLS degree 5
        try:
            V = np.vander(x_sub, N=6, increasing=True)
            coeff_ols, _, _, _ = np.linalg.lstsq(V, y_sub, rcond=None)
            if np.all(np.isfinite(coeff_ols)):
                best_coeff = coeff_ols
            else:
                best_coeff = np.zeros(6, dtype=np.float64)
        except Exception:
            best_coeff = np.zeros(6, dtype=np.float64)

    return best_coeff
# EVOLVE-BLOCK-END
