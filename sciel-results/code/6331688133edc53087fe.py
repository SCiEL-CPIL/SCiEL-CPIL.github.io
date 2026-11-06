import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Filter out any NaNs or infinities just in case
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n_samples = len(x)
    
    if n_samples < 10:
        # Fallback to linear if data is too sparse
        coeff = np.zeros(6)
        if n_samples >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-10:
        coeff = np.zeros(6)
        coeff[0] = np.mean(y)
        return coeff
    
    # Nadaraya-Watson smoothing with Gaussian kernel
    # This is a proper nonparametric smoother that handles variable density better than binning
    
    def nadaraya_watson(x, y, bandwidth):
        """
        Nadaraya-Watson estimator with Gaussian kernel.
        Returns smoothed y values at x locations.
        """
        # Use a grid for evaluation to be efficient
        # For large N, we can subsample or use a grid
        n = len(x)
        if n > 5000:
            # Subsample for speed
            idx = np.random.choice(n, 5000, replace=False)
            x_sub = x[idx]
            y_sub = y[idx]
        else:
            x_sub = x
            y_sub = y
        
        # Evaluate on a grid for efficiency
        grid_size = min(2000, len(x_sub))
        grid = np.linspace(x_min - bandwidth, x_max + bandwidth, grid_size)
        
        # Compute weights for each grid point
        y_smoothed_grid = np.zeros(grid_size)
        for i, xi in enumerate(grid):
            dist = x_sub - xi
            weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
            sum_w = np.sum(weights)
            if sum_w > 1e-10:
                y_smoothed_grid[i] = np.sum(weights * y_sub) / sum_w
            else:
                y_smoothed_grid[i] = np.mean(y_sub)
        
        # Interpolate back to original x locations
        y_smoothed = np.interp(x, grid, y_smoothed_grid)
        return y_smoothed
    
    # Try multiple bandwidths and select based on cross-validation or training MSE
    # Bandwidth should be relative to the range of x
    x_range = x_max - x_min
    candidate_bandwidths = [0.05 * x_range, 0.1 * x_range, 0.15 * x_range, 0.2 * x_range, 0.3 * x_range]
    
    best_coeff = None
    best_score = np.inf
    
    # Also try direct polynomial fit as a baseline
    try:
        coeff_direct = np.polynomial.polynomial.polyfit(x, y, 5)
        y_pred_direct = np.polynomial.polynomial.polyval(x, coeff_direct)
        mse_direct = np.mean((y - y_pred_direct) ** 2)
        best_coeff = coeff_direct
        best_score = mse_direct
    except Exception:
        pass
    
    # Try NW smoothing with different bandwidths
    for bw in candidate_bandwidths:
        if time.time() > deadline:
            break
            
        try:
            y_smoothed = nadaraya_watson(x, y, bw)
            
            # Fit polynomial to smoothed data
            coeff = np.polynomial.polynomial.polyfit(x, y_smoothed, 5)
            
            # Evaluate on original data
            y_pred = np.polynomial.polynomial.polyval(x, coeff)
            mse = np.mean((y - y_pred) ** 2)
            
            if mse < best_score:
                best_score = mse
                best_coeff = coeff
        except Exception:
            continue
    
    # If NW didn't improve, try a ridge-regularized polynomial fit
    if time.time() < deadline - 10:
        # Add ridge regularization to the polynomial fit
        # This helps with overfitting to noise
        from numpy.polynomial import polynomial as P
        
        # Create Vandermonde matrix
        vander = np.vander(x, 6, increasing=True)
        
        # Try different lambda values for ridge regularization
        for lambda_val in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
            if time.time() > deadline:
                break
            try:
                # Ridge regression: (X^T X + lambda I)^{-1} X^T y
                XtX = vander.T @ vander
                Xty = vander.T @ y
                ridge_coeff = np.linalg.solve(XtX + lambda_val * np.eye(6), Xty)
                
                y_pred = P.polyval(x, ridge_coeff)
                mse = np.mean((y - y_pred) ** 2)
                
                if mse < best_score:
                    best_score = mse
                    best_coeff = ridge_coeff
            except Exception:
                continue
    
    if best_coeff is None:
        # Final fallback
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            best_coeff = np.zeros(6)
            if n_samples >= 2:
                best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            else:
                best_coeff[0] = np.mean(y) if n_samples > 0 else 0.0
    
    # Ensure finite coefficients
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)
        if n_samples >= 2:
            try:
                best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                best_coeff[0] = np.mean(y)
    
    return best_coeff
