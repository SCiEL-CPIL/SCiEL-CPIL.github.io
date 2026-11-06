"""Odd-power polynomial closure with Legendre-basis regularized fit and CV polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 10:
        # Not enough data; return zero coefficients
        return np.zeros(6)

    n = x.size

    # The subgrid tendency h*c*mean(Y) is approximately an odd function of X
    # due to the symmetry of the Lorenz-96 fast dynamics.
    # Fit using only odd powers: x, x^3, x^5
    # This reduces parameters from 6 to 3, improving generalization.

    # Scale x to [-1, 1] for numerical stability with Legendre polynomials
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-15:
        # Degenerate case: all x are the same
        c1 = np.mean(y) / x[0] if abs(x[0]) > 1e-15 else 0.0
        coeff = np.zeros(6)
        coeff[1] = c1
        return coeff

    # Map x to [-1, 1]
    x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)

    # Build Legendre polynomials of degree 1, 3, 5 evaluated at x_scaled
    # P_1(t) = t
    # P_3(t) = (5t^3 - 3t) / 2
    # P_5(t) = (63t^5 - 70t^3 + 15t) / 8
    L1 = x_scaled
    L3 = (5.0 * x_scaled**3 - 3.0 * x_scaled) / 2.0
    L5 = (63.0 * x_scaled**5 - 70.0 * x_scaled**3 + 15.0 * x_scaled) / 8.0

    # Design matrix in Legendre basis
    A_leg = np.column_stack([L1, L3, L5])

    # Determine number of folds for cross-validation
    if n >= 50:
        n_folds = 5
    elif n >= 20:
        n_folds = 4
    else:
        n_folds = 3

    # In the Legendre basis, the columns are approximately orthogonal,
    # so the Gram matrix is close to diagonal. Regularization in this basis
    # is more well-conditioned than in the monomial basis.
    # Use a lambda grid scaled relative to the diagonal of A_leg^T A_leg
    gram_full = A_leg.T @ A_leg
    diag_gram = np.diag(gram_full)
    # Scale lambda relative to the mean diagonal element
    mean_diag = np.mean(diag_gram)
    if mean_diag > 0 and np.isfinite(mean_diag):
        base_lam = mean_diag
        rel_factors = np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
        lams = rel_factors * base_lam
    else:
        lams = np.array([1e-6, 1e-4, 1e-2, 1.0, 100.0])

    # Ensure at least a few lambda values
    if lams.size < 3:
        lams = np.array([1e-4, 1e-2, 1.0])

    # K-fold cross-validation to select best lambda
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)

    # Precompute fold masks
    fold_sizes = np.array([n // n_folds] * n_folds)
    fold_sizes[:n % n_folds] += 1
    fold_starts = np.concatenate([[0], np.cumsum(fold_sizes)])

    best_cv_mse = np.inf
    best_lam = lams[0]

    for i_lam, lam in enumerate(lams):
        # Check time budget
        if time.time() > deadline - 10.0:
            break

        cv_mse_sum = 0.0
        cv_count = 0

        for k in range(n_folds):
            test_idx = indices[fold_starts[k]:fold_starts[k+1]]
            train_idx = np.concatenate([indices[:fold_starts[k]], indices[fold_starts[k+1]:]])

            A_train = A_leg[train_idx]
            y_train = y[train_idx]
            A_test = A_leg[test_idx]
            y_test = y[test_idx]

            # Solve regularized least squares in Legendre basis
            try:
                gram = A_train.T @ A_train
                rhs = A_train.T @ y_train
                coeff_leg_k = np.linalg.solve(gram + lam * np.eye(3), rhs)
            except np.linalg.LinAlgError:
                try:
                    coeff_leg_k = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
                except:
                    continue

            # Compute MSE on test fold
            pred = A_test @ coeff_leg_k
            mse = np.mean((pred - y_test)**2)
            cv_mse_sum += mse
            cv_count += 1

        if cv_count > 0:
            cv_mse = cv_mse_sum / cv_count
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_lam = lam

    # Refit on full dataset with best lambda
    try:
        gram = A_leg.T @ A_leg
        rhs = A_leg.T @ y
        coeff_leg = np.linalg.solve(gram + best_lam * np.eye(3), rhs)
    except np.linalg.LinAlgError:
        try:
            coeff_leg = np.linalg.lstsq(A_leg, y, rcond=None)[0]
        except:
            # Ultimate fallback: unregularized least squares on linear term only
            coeff_leg = np.zeros(3)
            denom = np.dot(L1, L1)
            if denom > 1e-15:
                coeff_leg[0] = np.dot(L1, y) / denom

    # Convert from Legendre basis to monomial basis
    # P_1(t) = t
    # P_3(t) = (5t^3 - 3t) / 2
    # P_5(t) = (63t^5 - 70t^3 + 15t) / 8
    #
    # We have: y ≈ a1*P_1(t) + a3*P_3(t) + a5*P_5(t)
    # where t = (x - x_mid) / x_half, x_mid = (x_min+x_max)/2, x_half = x_range/2
    #
    # Expand in powers of t:
    # a1*t + a3*(5t^3-3t)/2 + a5*(63t^5-70t^3+15t)/8
    # = (a1 - 3a3/2 + 15a5/8)*t + (5a3/2 - 70a5/8)*t^3 + (63a5/8)*t^5
    #
    # Now substitute t = (x - x_mid) / x_half and expand in powers of x.
    # This is algebraically intensive, so let's use a numerical approach:
    # Evaluate the Legendre-based polynomial at a set of points and fit
    # a degree-5 polynomial in x to those values.

    # Generate evaluation points in the training range
    n_eval = 200
    x_eval = np.linspace(x_min, x_max, n_eval)
    t_eval = (x_eval - (x_min + x_max) / 2.0) / (x_range / 2.0)

    # Evaluate the fitted Legendre polynomial at these points
    L1_eval = t_eval
    L3_eval = (5.0 * t_eval**3 - 3.0 * t_eval) / 2.0
    L5_eval = (63.0 * t_eval**5 - 70.0 * t_eval**3 + 15.0 * t_eval) / 8.0
    A_leg_eval = np.column_stack([L1_eval, L3_eval, L5_eval])
    y_eval = A_leg_eval @ coeff_leg

    # Fit a degree-5 monomial polynomial to (x_eval, y_eval)
    # Using only odd powers for consistency
    A_monomial_eval = np.column_stack([x_eval, x_eval**3, x_eval**5])
    try:
        coeff_odd, _, _, _ = np.linalg.lstsq(A_monomial_eval, y_eval, rcond=None)
    except:
        # Fallback: use the direct algebraic conversion
        x_mid = (x_min + x_max) / 2.0
        x_half = x_range / 2.0
        a1, a3, a5 = coeff_leg

        # Coefficients in powers of t
        c1_t = a1 - 3.0 * a3 / 2.0 + 15.0 * a5 / 8.0
        c3_t = 5.0 * a3 / 2.0 - 70.0 * a5 / 8.0
        c5_t = 63.0 * a5 / 8.0

        # Now expand (x - x_mid)/x_half in powers of x
        # t = (x - x_mid) / x_half
        # t^1 = x/x_half - x_mid/x_half
        # t^3 = (x - x_mid)^3 / x_half^3
        # t^5 = (x - x_mid)^5 / x_half^5
        #
        # This is complex, so let's just use the lstsq result above.
        # If that failed, fall back to linear only.
        coeff_odd = np.zeros(3)
        denom = np.dot(x, x)
        if denom > 1e-15:
            coeff_odd[0] = np.dot(x, y) / denom

    # Map odd coefficients to full degree-5 polynomial
    coeff = np.zeros(6)
    coeff[1] = coeff_odd[0]  # x^1
    coeff[3] = coeff_odd[1]  # x^3
    coeff[5] = coeff_odd[2]  # x^5

    # Verify finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to simple linear
        try:
            c1 = np.dot(x, y) / (np.dot(x, x) + 1e-15)
            coeff = np.zeros(6)
            coeff[1] = c1
        except:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
