"""Rewritten program with improved L1 solver, scaled basis, and expanded search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fista_lasso(X, y, lam, max_iter=2000, tol=1e-8):
    """
    Fit Lasso (L1 regularized) regression using FISTA.
    X: (n, d) design matrix
    y: (n,) target
    lam: L1 penalty parameter
    Returns coefficients (d,)
    """
    n, d = X.shape
    if n < d:
        # Fallback to least squares if underdetermined
        try:
            return np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            return np.zeros(d)

    # Precompute X^T X and X^T y for speed? No, FISTA uses matrix-vector products.
    # Lipschitz constant L for gradient step
    # L = lambda_max(X^T X / n)
    try:
        # Use power iteration or eigenvalues for small d
        XtX = X.T @ X
        L = np.max(np.abs(np.linalg.eigvalsh(XtX))) / n
        if L < 1e-12:
            L = 1.0
    except:
        L = 1.0

    step = 1.0 / L
    
    # Initialize
    beta = np.zeros(d)
    beta_old = np.zeros(d)
    t_k = 1.0
    
    for k in range(max_iter):
        # Gradient step
        grad = (X.T @ (X @ beta - y)) / n
        beta_new = beta - step * grad
        
        # Soft thresholding (proximal operator)
        beta_new = np.sign(beta_new) * np.maximum(np.abs(beta_new) - step * lam, 0.0)
        
        # Check convergence
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
            
        # Update momentum
        t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        beta = beta_new + ((t_k - 1) / t_next) * (beta_new - beta_old)
        beta_old = beta_new
        t_k = t_next
        
    return beta


def _fit_poly_scaled(x, y, deg, lam_l1=0.0, lam_l2=0.0):
    """
    Fit polynomial of degree 'deg' to x, y.
    Uses scaled x for numerical stability.
    Returns 6-coeff array in original x basis.
    """
    if len(x) < deg + 1:
        return None

    # Scale x to [0, 1] or [-1, 1] for stability
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-12:
        scale = 1.0
        offset = 0.0
    else:
        scale = (x_max - x_min) / 2.0
        offset = (x_max + x_min) / 2.0
        
    x_scaled = (x - offset) / scale
    
    V = np.polynomial.polynomial.polyvander(x_scaled, deg)
    
    if lam_l1 > 0.0:
        # L1 fit
        c_scaled = _fista_lasso(V, y, lam_l1)
    elif lam_l2 > 0.0:
        # Ridge fit
        R = lam_l2 * np.eye(deg + 1)
        # Don't regularize the intercept as much? Standard ridge regularizes all.
        A = V.T @ V + R
        b = V.T @ y
        try:
            c_scaled = np.linalg.solve(A, b)
        except:
            c_scaled = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        # Least Squares
        try:
            c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]
        except:
            return None

    if not np.all(np.isfinite(c_scaled)):
        return None

    # Transform coefficients back to original x basis
    # We want c_orig such that sum(c_orig[i] * x**i) == sum(c_scaled[i] * ((x-offset)/scale)**i)
    # This is a change of basis.
    # Let z = (x - offset) / scale => x = scale * z + offset
    # We have P(z) = sum c_scaled[i] z^i
    # We want Q(x) = P((x - offset)/scale) = sum c_orig[j] x^j
    
    # We can compute this by evaluating the polynomial in the new basis?
    # Or use polynomial composition.
    # Easier: Create a Vandermonde matrix for the transformation?
    # Actually, np.polynomial.polynomial.polyfromroots or similar doesn't help directly with affine transform of variable.
    
    # Let's use the property that if we have coefficients in z, we can convert to x.
    # z = (x - offset)/scale
    # z^k = (1/scale^k) * (x - offset)^k
    # (x - offset)^k = sum_{j=0}^k binom(k, j) x^j (-offset)^(k-j)
    
    c_orig = np.zeros(deg + 1)
    for i in range(deg + 1):
        if c_scaled[i] == 0:
            continue
        # Coefficients of (x - offset)^i
        # Binomial expansion
        for j in range(i + 1):
            binom_coeff = 1.0
            # Calculate binom(i, j)
            # i is small (<=5), so simple loop
            num = 1
            den = 1
            for k in range(j):
                num *= (i - k)
                den *= (k + 1)
            binom_coeff = num / den
            
            term = c_scaled[i] * (1.0 / (scale ** i)) * binom_coeff * ((-offset) ** (i - j))
            c_orig[j] += term
            
    # Pad to 6
    out = np.zeros(6)
    out[:deg + 1] = c_orig
    
    if not np.all(np.isfinite(out)):
        return None
        
    return out


def _cv_mse(x, y, deg, lam_l1=0.0, lam_l2=0.0, n_folds=5, seed=42):
    """K-fold CV MSE."""
    n = len(x)
    if n < 2 * n_folds:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        c = _fit_poly_scaled(x, y, deg, lam_l1, lam_l2)
        if c is None:
            return np.inf
        pred = np.polynomial.polynomial.polyval(x, c)
        return np.mean((pred - y) ** 2)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < deg + 1:
            continue

        c = _fit_poly_scaled(x[train_idx], y[train_idx], deg, lam_l1, lam_l2)
        if c is None:
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 6:
        return np.zeros(6)

    best_coeff = None
    best_score = np.inf

    # Candidate configurations: (deg, lam_l1, lam_l2)
    # We test a range of degrees and regularization strengths.
    # L1 is good for sparsity, L2 for stability.
    candidates = [
        # Unregularized / Light Ridge
        (5, 0.0, 0.0),
        (5, 0.0, 1e-5),
        (5, 0.0, 1e-4),
        (5, 0.0, 1e-3),
        (4, 0.0, 1e-4),
        (4, 0.0, 1e-3),
        (3, 0.0, 1e-3),
        # L1 Regularization (Lasso)
        (5, 1e-4, 0.0),
        (5, 1e-3, 0.0),
        (5, 1e-2, 0.0),
        (5, 1e-1, 0.0),
        (4, 1e-3, 0.0),
        (4, 1e-2, 0.0),
        (3, 1e-2, 0.0),
        (2, 1e-2, 0.0),
        # Elastic Net (Both)
        (5, 1e-3, 1e-3),
        (5, 1e-2, 1e-2),
    ]

    # Sort candidates by expected simplicity? Or just iterate.
    # To save time, we can stop early if we find a very good score, but CV is expensive.
    # We will iterate through all unless time runs out.

    for deg, l1, l2 in candidates:
        if time.time() > deadline - 5.0:
            break

        score = _cv_mse(x, y, deg, l1, l2, n_folds=5, seed=42)
        
        if score < best_score:
            best_score = score
            # Refit on full training set
            c = _fit_poly_scaled(x, y, deg, l1, l2)
            if c is not None and np.all(np.isfinite(c)):
                best_coeff = c

    if best_coeff is not None and np.all(np.isfinite(best_coeff)):
        return best_coeff

    # Fallbacks
    try:
        c = np.polynomial.polynomial.polyfit(x, y, 5)
        if np.all(np.isfinite(c)):
            out = np.zeros(6)
            out[:6] = c
            return out
    except:
        pass

    try:
        c = np.polynomial.polynomial.polyfit(x, y, 3)
        if np.all(np.isfinite(c)):
            out = np.zeros(6)
            out[:4] = c
            return out
    except:
        pass

    return np.zeros(6)
# EVOLVE-BLOCK-END
