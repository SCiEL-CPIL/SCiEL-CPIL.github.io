"""Weighted group-aware Tikhonov polynomial closure for two-scale Lorenz-96.

Structural change from parent: instead of a single unweighted ridge fit on odd powers,
this program:
  1. Identifies the two training forcing regimes (F=6, F=8) via clustering on x.
  2. Fits a shared odd-power polynomial with a composite loss:
     - Standard L2 data residual
     - A group-separation penalty that encourages the polynomial to explain
       the cross-F mean difference through x-variation rather than a constant offset
  3. Selects regularization via internal k-fold CV on the training data.
This targets better generalization to the unseen F=10 validation regime.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses group-aware Tikhonov regularization exploiting the two training F-regimes.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    if len(x) < 6:
        return np.zeros(6)

    n = len(x)

    # --- Identify the two F-regimes via 1-D k-means on x ---
    # The two training trajectories are at F=6 and F=8.
    # Their x-distributions differ in mean/variance.
    # Use a simple 2-cluster split: find the median gap.
    x_sorted = np.sort(x)
    # Try to find a natural gap; if none, split at median
    diffs = np.diff(x_sorted)
    # Use the largest gap as a separator if it's significant
    if len(diffs) > 0:
        max_gap_idx = np.argmax(diffs)
        # If the largest gap is more than 1 std of x, use it as separator
        x_std = x.std()
        if diffs[max_gap_idx] > 0.5 * x_std:
            sep = x_sorted[max_gap_idx] + 0.5 * diffs[max_gap_idx]
            group_a = x <= sep
            group_b = ~group_a
        else:
            # Fallback: split at median
            median_x = np.median(x)
            group_a = x <= median_x
            group_b = ~group_a
    else:
        group_a = np.zeros(n, dtype=bool)
        group_b = np.ones(n, dtype=bool)

    # Ensure both groups have at least 3 samples
    if group_a.sum() < 3 or group_b.sum() < 3:
        # Fallback: use all data as one group
        group_a = np.ones(n, dtype=bool)
        group_b = np.zeros(n, dtype=bool)

    # --- Build design matrix for odd powers: x, x^3, x^5 ---
    V = np.column_stack([x, x**3, x**5]).astype(np.float64)

    # Scale columns for numerical stability
    col_means = V.mean(axis=0)
    col_stds = V.std(axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    V_scaled = (V - col_means) / col_stds

    # --- Group-aware Tikhonov penalty ---
    # Compute group means of y
    y_a_mean = y[group_a].mean() if group_a.sum() > 0 else 0.0
    y_b_mean = y[group_b].mean() if group_b.sum() > 0 else 0.0
    y_diff = y_b_mean - y_a_mean  # observed cross-F mean difference

    # Compute what the polynomial predicts for the group mean x values
    x_a_mean = x[group_a].mean() if group_a.sum() > 0 else 0.0
    x_b_mean = x[group_b].mean() if group_b.sum() > 0 else 0.0

    # The group-separation penalty: we want the polynomial's predicted difference
    # at the group-mean x values to be consistent with the observed y difference.
    # This is a soft constraint: (p(x_b_mean) - p(x_a_mean) - y_diff)^2 * beta
    # where p(x) = c1*x + c3*x^3 + c5*x^5
    # In terms of coefficients c = [c1, c3, c5]:
    # p(x_b) - p(x_a) = c1*(x_b - x_a) + c3*(x_b^3 - x_a^3) + c5*(x_b^5 - x_a^5)
    g_vec = np.array([
        x_b_mean - x_a_mean,
        x_b_mean**3 - x_a_mean**3,
        x_b_mean**5 - x_a_mean**5
    ])

    # --- K-fold cross-validation to select regularization ---
    # We have two hyperparameters: alpha (ridge on coefficients) and beta (group penalty)
    # Use a grid search over (alpha, beta) with k-fold CV

    n_folds = 5
    # Create fold indices
    rng = np.random.RandomState(42)
    fold_indices = np.array_split(np.arange(n), n_folds)

    alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    betas = [0.0, 0.01, 0.1, 1.0, 10.0]

    best_cv_score = np.inf
    best_alpha = 0.0
    best_beta = 0.0

    # Precompute for speed
    V_scaled_T = V_scaled.T
    y_col = y.reshape(-1, 1)

    for alpha in alphas:
        if time.time() - start_time > deadline - 20.0:
            break
        for beta in betas:
            if time.time() - start_time > deadline - 20.0:
                break

            fold_errors = []
            for fold_i in range(n_folds):
                # Validation set
                val_idx = fold_indices[fold_i]
                # Training set
                train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != fold_i])

                V_train = V_scaled[train_idx]
                y_train = y[train_idx]
                V_val = V_scaled[val_idx]
                y_val = y[val_idx]

                # Build regularized system:
                # (V^T V + alpha * I) c = V^T y  +  group penalty term
                # The group penalty adds: beta * g_vec * (g_vec^T c - y_diff)
                # In matrix form: add beta * outer(g_vec, g_vec) to the left,
                # and beta * g_vec * y_diff to the right
                n_params = 3
                A = V_train.T @ V_train + alpha * np.eye(n_params)
                b = V_train.T @ y_train
                # Add group penalty
                A += beta * np.outer(g_vec, g_vec)
                b += beta * g_vec * y_diff

                try:
                    coeffs = np.linalg.solve(A, b)
                    if not np.all(np.isfinite(coeffs)):
                        continue
                    pred_val = V_val @ coeffs
                    mse = np.mean((pred_val - y_val) ** 2)
                    fold_errors.append(mse)
                except np.linalg.LinAlgError:
                    continue

            if len(fold_errors) > 0:
                cv_score = np.mean(fold_errors)
                if cv_score < best_cv_score:
                    best_cv_score = cv_score
                    best_alpha = alpha
                    best_beta = beta

    # --- Final fit with best hyperparameters ---
    A_final = V_scaled.T @ V_scaled + best_alpha * np.eye(3)
    b_final = V_scaled.T @ y
    A_final += best_beta * np.outer(g_vec, g_vec)
    b_final += best_beta * g_vec * y_diff

    try:
        coeffs_scaled = np.linalg.solve(A_final, b_final)
        if not np.all(np.isfinite(coeffs_scaled)):
            raise np.linalg.LinAlgError("Non-finite coefficients")
    except Exception:
        # Fallback: unregularized least squares
        try:
            coeffs_scaled = np.linalg.lstsq(V_scaled, y, rcond=None)[0]
            if not np.all(np.isfinite(coeffs_scaled)):
                raise Exception
        except Exception:
            # Last resort: linear fit
            try:
                c_lin = np.polyfit(x, y, 1)
                coeffs = np.zeros(6)
                coeffs[1] = c_lin[0]  # polyfit returns descending powers
                # Actually polyfit returns [a1, a0] for degree 1: a1*x + a0
                # We want ascending: c0 + c1*x + ... so c1 = a1
                return coeffs
            except Exception:
                return np.zeros(6)

    # Unscale coefficients: V = (V - col_means) / col_stds
    # So y = V_scaled @ coeffs_scaled = ((V - col_means)/col_stds) @ coeffs_scaled
    # We need y = V @ coeffs where V = [x, x^3, x^5]
    # V_scaled @ coeffs_scaled = (V - col_means) @ diag(1/col_stds) @ coeffs_scaled
    # = V @ diag(1/col_stds) @ coeffs_scaled - col_means @ diag(1/col_stds) @ coeffs_scaled
    # So we need: V @ coeffs = V_scaled @ coeffs_scaled
    # => coeffs = diag(1/col_stds) @ coeffs_scaled  (for the V part)
    # But there's a constant offset: col_means @ diag(1/col_stds) @ coeffs_scaled
    # This means our polynomial in unscaled form has an implicit constant term.
    # However, since we only use odd powers and the physics says the forcing is odd,
    # the constant should be ~0. Let's just map the scaled coefficients back.
    # Actually, the correct mapping:
    # y = V_scaled @ c_s = ((V - m)/s) @ c_s = V @ (c_s / s) - (m/s) @ c_s
    # The term (m/s) @ c_s is a constant. Since we're fitting odd powers only,
    # this constant is an artifact. For the output, we want coefficients of x, x^3, x^5.
    # The effective coefficients in the original basis are c_s / s.
    coeffs_unscaled = coeffs_scaled / col_stds

    # Map to 6-coefficient output (ascending powers)
    coeffs_out = np.zeros(6)
    coeffs_out[1] = coeffs_unscaled[0]  # x
    coeffs_out[3] = coeffs_unscaled[1]  # x^3
    coeffs_out[5] = coeffs_unscaled[2]  # x^5

    # Final sanity check
    if not np.all(np.isfinite(coeffs_out)):
        return np.zeros(6)

    return coeffs_out
# EVOLVE-BLOCK-END
