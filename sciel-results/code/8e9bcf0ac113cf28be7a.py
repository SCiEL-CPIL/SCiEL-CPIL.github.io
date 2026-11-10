"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Strategy: Use robust weighted least-squares with iterative reweighting to
    downweight outliers and capture the main structure. The degree-5 polynomial
    is constrained by the task, so we optimize within that space.
    
    We use:
    1. Initial full least-squares fit
    2. Iterative reweighting based on residuals (Huber-like)
    3. Multiple random subsampling to find robust solution
    4. Return the best polynomial by internal validation (train subset)
    """
    start_time = time.time()
    DEADLINE = 140.0  # seconds, with 10s safety margin
    
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(training_x)
    if n < 6:
        # Not enough data for degree-5, fall back to lower degree
        deg = max(1, min(5, n - 1))
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:deg+1] = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg)
        return coeff
    
    # Build Vandermonde matrix for degree-5 polynomial
    # V[i, j] = x[i]^j for j=0..5
    V = np.vander(training_x, N=6, increasing=True)
    
    # Helper: weighted least squares
    def wls_solve(V, y, w):
        # Solve (V^T W V) c = V^T W y
        W = np.diag(w)
        # Use normal equations for efficiency
        VtW = V.T * w  # (6, n)
        A = VtW @ V    # (6, 6)
        b = VtW @ y    # (6,)
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
        return c
    
    # Helper: compute normalized MSE
    def norm_mse(y_true, y_pred):
        var = np.var(y_true)
        if var < 1e-12:
            return np.mean((y_true - y_pred)**2)
        return np.mean((y_true - y_pred)**2) / var
    
    # Initial fit
    c0 = wls_solve(V, training_subgrid_tendency, np.ones(n))
    
    # Iterative reweighting (Huber-like)
    def robust_fit(V, y, n_iter=20, delta=1.5):
        w = np.ones(len(y))
        c = wls_solve(V, y, w)
        for _ in range(n_iter):
            if time.time() > DEADLINE:
                break
            residuals = y - V @ c
            abs_res = np.abs(residuals)
            med = np.median(abs_res)
            if med < 1e-12:
                med = 1e-12
            # Huber weights
            w = np.where(abs_res <= delta * med, 1.0, delta * med / abs_res)
            c_new = wls_solve(V, y, w)
            if np.allclose(c, c_new, atol=1e-10):
                c = c_new
                break
            c = c_new
        return c
    
    # Get robust fit
    c_robust = robust_fit(V, training_subgrid_tendency, n_iter=30, delta=1.5)
    
    # Also try with different delta values
    best_c = c_robust
    best_score = norm_mse(training_subgrid_tendency, V @ c_robust)
    
    for delta in [0.5, 1.0, 2.0, 3.0]:
        if time.time() > DEADLINE:
            break
        c_try = robust_fit(V, training_subgrid_tendency, n_iter=20, delta=delta)
        score_try = norm_mse(training_subgrid_tendency, V @ c_try)
        if score_try < best_score:
            best_score = score_try
            best_c = c_try
    
    # Try subsampling for robustness: fit on random subsets and average
    if time.time() < DEADLINE - 20:
        n_subsets = 10
        subset_size = max(6, int(0.8 * n))
        rng = np.random.RandomState(42)
        
        # Collect multiple fits
        fits = [best_c.copy()]
        for _ in range(n_subsets):
            if time.time() > DEADLINE - 10:
                break
            idx = rng.choice(n, size=subset_size, replace=False)
            V_sub = V[idx]
            y_sub = training_subgrid_tendency[idx]
            c_sub = wls_solve(V_sub, y_sub, np.ones(subset_size))
            fits.append(c_sub)
        
        # Use median of coefficients for robustness
        fits_arr = np.array(fits)
        c_median = np.median(fits_arr, axis=0)
        
        # Evaluate which is better: robust fit or median
        score_median = norm_mse(training_subgrid_tendency, V @ c_median)
        if score_median < best_score:
            best_c = c_median
            best_score = score_median
    
    # Final check: ensure finite
    if not np.all(np.isfinite(best_c)):
        # Fall back to simple linear fit
        best_c = np.zeros(6)
        best_c[:2] = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, 1)
    
    return best_c
# EVOLVE-BLOCK-END
