"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Strategy: Combine robust weighted least-squares with a local weighted
    polynomial regression approach. The key insight is that the fast-scale
    forcing may have regime-dependent structure that a single global polynomial
    cannot capture well. By using a Gaussian kernel to weight samples based on
    their proximity in x-space, we can create an effective local fit that is
    then represented by the global polynomial coefficients.
    
    We use:
    1. Initial full least-squares fit
    2. Iterative reweighting based on residuals (Huber-like)
    3. Local weighted regression with Gaussian kernel to capture local structure
    4. Multiple bandwidths to find the best local-global compromise
    5. Return the best polynomial by internal validation (train subset)
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
    
    # Local weighted polynomial regression with Gaussian kernel
    # This captures regime-dependent structure by weighting nearby points more heavily
    if time.time() < DEADLINE - 30:
        # Try different bandwidths for the Gaussian kernel
        x_range = np.ptp(training_x)
        if x_range < 1e-10:
            x_range = 1.0
        
        bandwidths = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        
        for bw_factor in bandwidths:
            if time.time() > DEADLINE - 10:
                break
                
            bandwidth = bw_factor * x_range
            
            # Create local weights for each point based on its x-value
            # We'll do a "local" fit by weighting each sample's contribution
            # based on a Gaussian kernel centered at that sample's x-value
            # Then average the resulting polynomial coefficients
            
            n_local = min(50, n)  # Limit number of local fits for speed
            if n_local < 10:
                n_local = n
                
            # Sample indices for local fits
            if n_local < n:
                rng = np.random.RandomState(42)
                local_indices = rng.choice(n, size=n_local, replace=False)
            else:
                local_indices = np.arange(n)
            
            local_fits = []
            for idx in local_indices:
                if time.time() > DEADLINE - 5:
                    break
                    
                x_center = training_x[idx]
                
                # Gaussian kernel weights
                dist = np.abs(training_x - x_center)
                w_local = np.exp(-0.5 * (dist / bandwidth)**2)
                
                # Ensure at least some weight
                w_local = np.maximum(w_local, 1e-10)
                
                c_local = wls_solve(V, training_subgrid_tendency, w_local)
                if np.all(np.isfinite(c_local)):
                    local_fits.append(c_local)
            
            if len(local_fits) > 0:
                # Average the local fits
                c_avg = np.mean(local_fits, axis=0)
                score_avg = norm_mse(training_subgrid_tendency, V @ c_avg)
                
                if score_avg < best_score:
                    best_score = score_avg
                    best_c = c_avg
    
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
