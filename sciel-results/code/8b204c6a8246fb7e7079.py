"""Rewritten program: physics-informed sparse basis + ridge + optional L1 selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_physics_basis(x, degree=5):
    """
    Build a sparse, physics-informed basis matrix of width degree+1.
    Basis vectors (ascending in x):
      0: 1
      1: x
      2: x^2
      3: x^3
      4: x * mean(x)  -- mean-field interaction
      5: x^5          -- highest odd term (for degree=5)
    Returns (B, active_cols) where B is (n, degree+1) and active_cols maps
    output index -> basis column index.
    """
    n = len(x)
    mx = np.mean(x)
    B = np.zeros((n, degree + 1))
    B[:, 0] = 1.0
    B[:, 1] = x
    B[:, 2] = x ** 2
    B[:, 3] = x ** 3
    B[:, 4] = x * mx
    if degree >= 5:
        B[:, 5] = x ** 5
    # For degrees < 5, we still fill all columns but we'll only use up to degree
    active_cols = list(range(degree + 1))
    return B, active_cols

def _ridge_solve(B, y, alpha):
    """Solve ridge regression: min ||B c - y||^2 + alpha ||c||^2."""
    A = B.T @ B
    b = B.T @ y
    reg = alpha * np.eye(A.shape[0])
    reg[0, 0] += 1e-12
    try:
        c = np.linalg.solve(A + reg, b)
    except np.linalg.LinAlgError:
        c, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    return c

def _polyval_basis(x, c, degree=5):
    """Evaluate using the physics basis."""
    B, _ = _build_physics_basis(x, degree)
    return B @ c

def _convert_to_standard_poly(c_basis, degree=5):
    """
    Convert coefficients from the physics basis to standard ascending polynomial
    coefficients [c0, c1, c2, c3, c4, c5].
    Basis: [1, x, x^2, x^3, x*mean(x), x^5]
    We need to express x*mean(x) as mean(x)*x, which is (mean(x))*x, i.e. it
    contributes to the x^1 term. So:
      c0_std = c0_basis
      c1_std = c1_basis + c4_basis * mean(x)  -- but mean(x) is data-dependent!
    
    This is a problem: the physics basis is data-dependent. The task requires
    a single polynomial valid for all x. So we CANNOT use x*mean(x) as a basis
    term in the final output. 
    
    Instead, let's use a different approach: use only pure monomials but select
    which ones to keep via L1. The "physics-informed" part is that we bias the
    initial search toward odd powers and lower degrees.
    """
    # Since x*mean(x) is not a pure monomial, we can't directly convert.
    # Let's just return the coefficients for the pure monomial terms.
    std = np.zeros(degree + 1)
    for i in range(min(len(c_basis), degree + 1)):
        if i <= 3:
            std[i] = c_basis[i]
        elif i == 5:
            std[5] = c_basis[5]
        # i==4 (x*mean(x)) is dropped since it's not a pure monomial
    return std

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()

    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.zeros(6)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 6:
        if len(x) >= 2:
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except:
                coeff[0] = np.mean(y)
            return coeff
        elif len(x) >= 1:
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)
            return coeff
        else:
            return np.zeros(6)

    n = len(x)

    # Default fallback: linear fit
    try:
        default_coeff = np.zeros(6)
        default_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    except:
        default_coeff = np.zeros(6)
        default_coeff[0] = np.mean(y)

    if n <= 50:
        # Small sample: use ridge with moderate alpha on degree-5 polynomial
        try:
            xc = x - np.mean(x)
            sigma = np.std(x)
            if sigma < 1e-12:
                sigma = 1.0
            xc = xc / sigma
            V = np.column_stack([xc ** k for k in range(6)])
            A = V.T @ V
            b = V.T @ y
            alpha = 1e-2
            reg = alpha * np.eye(6)
            reg[0, 0] += 1e-12
            c_centered = np.linalg.solve(A + reg, b)
            
            # Convert back to original basis
            coeff_orig = np.zeros(6)
            mu = np.mean(x)
            for k in range(6):
                if abs(c_centered[k]) < 1e-15:
                    continue
                a = 1.0 / sigma
                b_const = -mu / sigma
                for i in range(k + 1):
                    # binom(k, i)
                    from math import comb
                    binom_k_i = comb(k, i)
                    term_coeff = binom_k_i * (a ** i) * (b_const ** (k - i))
                    coeff_orig[i] += c_centered[k] * term_coeff
            if np.all(np.isfinite(coeff_orig)):
                return coeff_orig[:6]
        except:
            pass
        return default_coeff

    # Larger sample: cross-validate degree and alpha
    # Use pure monomial basis with centering/scaling for numerical stability
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    n_folds = 5
    fold_size = n // n_folds
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]

    candidates = []
    for deg in [5, 4, 3, 2]:
        for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            candidates.append((deg, alpha))

    best_cv_score = np.inf
    best_params = (5, 1e-4)

    for deg, alpha in candidates:
        if time.time() > deadline - 15.0:
            break

        cv_errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]

            try:
                mu = np.mean(x_train)
                sigma = np.std(x_train)
                if sigma < 1e-12:
                    sigma = 1.0
                xc_train = (x_train - mu) / sigma
                xc_val = (x_val - mu) / sigma

                V_train = np.column_stack([xc_train ** k for k in range(deg + 1)])
                V_val = np.column_stack([xc_val ** k for k in range(deg + 1)])

                A = V_train.T @ V_train
                b = V_train.T @ y_train
                reg = alpha * np.eye(deg + 1)
                reg[0, 0] += 1e-12
                try:
                    c_centered = np.linalg.solve(A + reg, b)
                except np.linalg.LinAlgError:
                    c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)

                y_pred = V_val @ c_centered
                mse = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(mse)
            except Exception:
                cv_errors.append(np.inf)

        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_score:
            best_cv_score = mean_cv_error
            best_params = (deg, alpha)

    deg, alpha = best_params

    # Final fit on all data
    try:
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma < 1e-12:
            sigma = 1.0
        xc = (x - mu) / sigma
        V = np.column_stack([xc ** k for k in range(deg + 1)])

        A = V.T @ V
        b = V.T @ y
        reg = alpha * np.eye(deg + 1)
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)

        # Convert from centered/scaled to original basis
        coeff_orig = np.zeros(6)
        for k in range(deg + 1):
            if abs(c_centered[k]) < 1e-15:
                continue
            a = 1.0 / sigma
            b_const = -mu / sigma
            for i in range(k + 1):
                from math import comb
                binom_k_i = comb(k, i)
                term_coeff = binom_k_i * (a ** i) * (b_const ** (k - i))
                coeff_orig[i] += c_centered[k] * term_coeff

        best_coeff = coeff_orig
    except Exception:
        best_coeff = default_coeff

    # Optional: L1 sparsity selection via soft-thresholding in coefficient space
    # Try to zero out small coefficients to reduce overfitting
    if time.time() < deadline - 25.0:
        try:
            # Evaluate training MSE with current coefficients
            def _eval_mse(coeff):
                return np.mean((y - np.polynomial.polynomial.polyval(x, coeff)) ** 2)

            base_mse = _eval_mse(best_coeff)

            # Try L1 soft-thresholding at various levels
            best_sparse_coeff = best_coeff.copy()
            best_sparse_mse = base_mse
            for threshold in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1]:
                sparse = best_coeff.copy()
                sparse[np.abs(sparse) < threshold] = 0.0
                mse_sp = _eval_mse(sparse)
                # Accept if MSE doesn't increase much (within 5%)
                if mse_sp < best_sparse_mse * 1.05:
                    best_sparse_coeff = sparse
                    best_sparse_mse = mse_sp

            if best_sparse_mse < base_mse * 1.02:
                best_coeff = best_sparse_coeff
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = default_coeff

    return best_coeff[:6]
# EVOLVE-BLOCK-END
