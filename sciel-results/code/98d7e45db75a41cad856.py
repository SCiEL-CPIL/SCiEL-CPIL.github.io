"""Improved program for the suite task; uses Tikhonov-regularized polynomial fit with internal CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    Uses Tikhonov regularization (excluding intercept) with k-fold CV to select lambda.
    """
    start_time = time.time()
    deadline = 150.0  # seconds, safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Build Vandermonde matrix (ascending powers)
    V = np.vander(x, N=6, increasing=True)

    # Regularization matrix: exclude intercept (index 0) from penalty
    R = np.eye(6)
    R[0, 0] = 0.0

    def _fit_regularized(lam):
        """Solve (V^T V + lam*R) c = V^T y for coefficients c."""
        A = V.T @ V + lam * R
        b = V.T @ y
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
        return c

    def _mse(coeffs):
        """Mean squared error on training data."""
        pred = V @ coeffs
        return np.mean((y - pred) ** 2)

    # --- K-fold cross-validation to select lambda ---
    # Use a modest grid of lambda values
    y_var = np.var(y)
    if y_var < 1e-12:
        return np.zeros(6)

    # Scale lambda relative to data magnitude
    scale = np.mean(V.T @ V) / max(1, n)
    lam_grid = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]) * scale

    k = 5
    # Shuffle indices for CV
    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    best_lam = 0.0
    best_cv_score = np.inf

    for lam in lam_grid:
        fold_errors = []
        for i in range(k):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            if len(tr_idx) < 6:
                continue

            V_tr = V[tr_idx]
            y_tr = y[tr_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]

            # Regularized fit on training fold
            A = V_tr.T @ V_tr + lam * R
            b = V_tr.T @ y_tr
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, b, rcond=None)[0]

            pred_val = V_val @ c
            fold_errors.append(np.mean((y_val - pred_val) ** 2))

        if len(fold_errors) > 0:
            cv_score = np.mean(fold_errors)
            if cv_score < best_cv_score:
                best_cv_score = cv_score
                best_lam = lam

        # Deadline check
        if time.time() > deadline - 30:
            break

    # Final fit with best lambda on all data
    coeffs = _fit_regularized(best_lam)

    # --- Optional: try a few additional refinement steps if time permits ---
    if time.time() < deadline - 15:
        # Try a finer grid around the best lambda
        if best_lam > 0:
            fine_grid = np.linspace(best_lam * 0.5, best_lam * 2.0, 11)
        else:
            fine_grid = np.array([0.0, 1e-7, 1e-6, 1e-5]) * scale

        best_fine_score = best_cv_score
        best_fine_lam = best_lam

        for lam in fine_grid:
            if time.time() > deadline - 15:
                break
            fold_errors = []
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                if len(tr_idx) < 6:
                    continue
                V_tr = V[tr_idx]
                y_tr = y[tr_idx]
                V_val = V[val_idx]
                y_val = y[val_idx]
                A = V_tr.T @ V_tr + lam * R
                b = V_tr.T @ y_tr
                try:
                    c = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    c = np.linalg.lstsq(A, b, rcond=None)[0]
                pred_val = V_val @ c
                fold_errors.append(np.mean((y_val - pred_val) ** 2))
            if len(fold_errors) > 0:
                cv_score = np.mean(fold_errors)
                if cv_score < best_fine_score:
                    best_fine_score = cv_score
                    best_fine_lam = lam

        if best_fine_lam != best_lam:
            coeffs = _fit_regularized(best_fine_lam)

    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.zeros(6)

    return coeffs
# EVOLVE-BLOCK-END
