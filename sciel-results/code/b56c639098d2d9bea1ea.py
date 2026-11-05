"""Model-and-optimize: Regularized Chebyshev polynomial closure for Lorenz-96 subgrid tendency."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial import chebyshev as C


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial using Chebyshev basis with optimized Tikhonov
    regularization. Returns 6 coefficients in ascending powers of x.
    
    Strategy:
    1. Center and scale x to [-1, 1] for Chebyshev evaluation
    2. Build Chebyshev design matrix for degree 5
    3. Optimize regularization lambda via internal k-fold cross-validation
    4. Solve regularized least squares in Chebyshev basis
    5. Convert back to power basis for output
    """
    t_start = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)
    
    if n < 10:
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    # Determine scaling: min-max to map data range to [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_center = x_min
        x_scale = 1.0
        x_norm = np.zeros_like(x)
    else:
        x_center = (x_min + x_max) / 2.0
        x_scale = x_range / 2.0
        x_norm = (x - x_center) / x_scale
    
    # No clipping needed if we handle the domain properly, 
    # but Chebyshev is stable outside [-1,1] as long as we don't have extreme outliers 
    # that cause overflow in T_k. With degree 5, it's safe.
    
    # Build Chebyshev design matrix for degree 5
    # T_k(x) for k=0..5
    degree = 5
    # Evaluate Chebyshev polynomials
    # C.chebvander(x, deg) returns columns [T_0, T_1, ..., T_deg]
    try:
        V = C.chebvander(x_norm, degree)
    except Exception:
        # Fallback: use power basis with scaling
        V = np.vander(x_norm, N=degree+1, increasing=True)
    
    # Design matrix shape: (n, 6)
    # Columns: [T_0(x), T_1(x), ..., T_5(x)]
    
    # Tikhonov regularization: minimize ||V*c - y||^2 + lambda * ||c||^2
    # But we want to regularize higher degrees more (decay)
    # Use a diagonal regularization matrix that penalizes higher degrees more
    
    def solve_regularized(V, y, lam_diag):
        """Solve (V^T V + lam_diag) c = V^T y"""
        VtV = V.T @ V
        Vty = V.T @ y
        A = VtV + lam_diag
        # Solve using np.linalg.solve or lstsq
        try:
            c = np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, Vty, rcond=None)[0]
        return c
    
    def compute_mse(c, x_norm_test, y_test):
        """Compute MSE for given coefficients on test data"""
        try:
            V_test = C.chebvander(x_norm_test, degree)
        except Exception:
            V_test = np.vander(x_norm_test, N=degree+1, increasing=True)
        pred = V_test @ c
        return np.mean((pred - y_test) ** 2)
    
    # Try a grid of regularization parameters
    # Lambda scales the identity regularization per coefficient
    # We'll use a decaying weight: reg_k = lambda * (k+1)^2 for degree k
    lambda_candidates = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # Internal cross-validation to pick lambda
    # Use k-fold CV
    n_folds = 5
    n_samples = len(x_norm)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    if len(folds) < n_folds:
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(len(folds))]
    
    best_lambda = 0.0
    best_cv_score = np.inf
    best_coeff_cheb = None
    
    for lam in lambda_candidates:
        cv_scores = []
        for fold in folds:
            test_idx = fold
            train_idx = np.setdiff1d(indices, test_idx)
            
            if len(train_idx) < degree + 2:
                continue
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            
            # Build regularization matrix: diagonal with decaying weights
            # reg_k = lam * (k+1)^2 to penalize higher degrees more
            reg_weights = np.array([(k+1)**2 for k in range(degree+1)], dtype=np.float64) * lam
            lam_diag = np.diag(reg_weights)
            
            c = solve_regularized(V_train, y_train, lam_diag)
            
            # Evaluate on test fold
            mse = compute_mse(c, x_norm[test_idx], y[test_idx])
            cv_scores.append(mse)
        
        if cv_scores:
            avg_cv = np.mean(cv_scores)
            if avg_cv < best_cv_score:
                best_cv_score = avg_cv
                best_lambda = lam
                # Solve on full data with this lambda
                reg_weights = np.array([(k+1)**2 for k in range(degree+1)], dtype=np.float64) * best_lambda
                lam_diag = np.diag(reg_weights)
                best_coeff_cheb = solve_regularized(V, y, lam_diag)
        
        # Time check
        if time.time() > DEADLINE:
            break
    
    # If no valid lambda was found, use unregularized
    if best_coeff_cheb is None:
        best_coeff_cheb = solve_regularized(V, y, np.zeros((degree+1, degree+1)))
    
    # Check time
    if time.time() > DEADLINE - 5:
        # Convert and return quickly
        pass
    
    # Convert Chebyshev coefficients to power basis
    # best_coeff_cheb are coefficients for T_k(x_norm) where x_norm in [-1, 1]
    # We want the polynomial in x.
    # Using numpy's Chebyshev class with the correct domain handles the affine transform.
    
    try:
        # Create Chebyshev object with domain [x_min, x_max]
        # The coefficients are for the window [-1, 1], which maps to [x_min, x_max]
        cheb_obj = C.chebyshev.Chebyshev(best_coeff_cheb, domain=[x_min, x_max])
        # Convert to power basis in x
        poly_obj = cheb_obj.convert(kind='poly')
        power_coeff = poly_obj.coef
        
        # Ensure length is 6 (degree 5)
        if len(power_coeff) < 6:
            power_coeff = np.pad(power_coeff, (0, 6 - len(power_coeff)))
        else:
            power_coeff = power_coeff[:6]
            
    except Exception:
        # Fallback to grid fit if numpy conversion fails (unlikely)
        x_norm_grid = np.linspace(-1.0, 1.0, 200)
        y_grid = np.zeros_like(x_norm_grid)
        for k in range(degree+1):
            y_grid += best_coeff_cheb[k] * C.chebval(x_norm_grid, k)
        x_grid = x_norm_grid * x_scale + x_center
        try:
            power_coeff = np.polynomial.polynomial.polyfit(x_grid, y_grid, degree)
        except Exception:
            power_coeff = np.zeros(6)
            if n >= 2:
                power_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)

    # Ensure finite
    if not np.all(np.isfinite(power_coeff)):
        power_coeff = np.zeros(6)
        if n >= 2:
            power_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    
    return power_coeff
# EVOLVE-BLOCK-END
