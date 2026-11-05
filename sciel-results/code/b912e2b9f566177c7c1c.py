import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Strategy:
    1. Normalize x to improve conditioning.
    2. Construct a smooth, non-parametric surrogate of the data using a robust
       local weighted least-squares (LOESS) approach. This captures the
       oscillatory, non-polynomial structure of the Lorenz-96 subgrid tendency
       better than a direct global polynomial fit.
    3. Fit the degree-5 polynomial to this surrogate using Ridge Regression
       with a small regularization parameter to ensure stability.
    4. Transform the optimal normalized coefficients back to the original x scale 
       using binomial expansion.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin from 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        return np.zeros(6)
    
    # 1. Normalize x
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Sort for efficient LOESS
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_sorted = y[sort_idx]
    
    # 2. Generate a smooth surrogate using LOESS
    # We want to estimate the underlying function f(x) which is oscillatory.
    # A global polynomial of degree 5 cannot capture this well.
    # We create a grid of points and fit a local polynomial to each, then use
    # these local fits to create a dense, smooth surrogate.
    
    # Define a grid for the surrogate. Use a range slightly wider than the data
    # to ensure the polynomial fit covers the domain well.
    x_min = x_sorted[0] - 0.1
    x_max = x_sorted[-1] + 0.1
    n_grid = 1000
    x_grid = np.linspace(x_min, x_max, n_grid)
    
    y_surrogate = np.zeros(n_grid)
    
    # LOESS parameters
    # Bandwidth: fraction of data points to use in local fit
    # A smaller bandwidth captures more detail but is noisier.
    # A larger bandwidth is smoother but may miss oscillations.
    # For Lorenz-96, the fast scales are somewhat chaotic, so a moderate
    # bandwidth is appropriate.
    bandwidth_frac = 0.1  # Use 10% of points in each local fit
    n_neighbors = max(10, int(bandwidth_frac * len(x_sorted)))
    
    # Local polynomial degree
    local_deg = 3  # Cubic is usually sufficient for local smoothness
    
    # Precompute design matrix for local fits? No, it changes with center.
    # But we can vectorize the distance calculation.
    
    # For each grid point, find the n_neighbors closest points in x_sorted
    # and fit a local polynomial.
    
    # Optimization: Use searchsorted to find the start index of the window
    # This is O(N log N) for the whole grid.
    
    for i, xg in enumerate(x_grid):
        if time.time() - start_time > DEADLINE - 15:
            # Fallback: if we run out of time, just use the mean or a simple fit
            # We can break early and fill the rest with the last value or 0
            # But let's try to be faster.
            pass
            
        # Find the index of the closest point
        idx = np.searchsorted(x_sorted, xg)
        
        # Define the window
        start = max(0, idx - n_neighbors // 2)
        end = min(len(x_sorted), start + n_neighbors)
        
        # Adjust if we are at the boundaries
        if start == 0:
            end = min(len(x_sorted), n_neighbors)
        elif end == len(x_sorted):
            start = max(0, len(x_sorted) - n_neighbors)
            
        x_local = x_sorted[start:end]
        y_local = y_sorted[start:end]
        
        if len(x_local) < local_deg + 1:
            # Not enough points, use mean
            y_surrogate[i] = np.mean(y_local)
            continue
            
        # Fit local polynomial
        # Weights: Triangular or Gaussian? Triangular is standard for LOESS.
        # w_j = (1 - |x_j - xg| / h)^2 if |x_j - xg| < h, else 0
        # Here h is the distance to the farthest neighbor.
        h = max(np.abs(x_local[0] - xg), np.abs(x_local[-1] - xg))
        if h < 1e-12:
            h = 1.0
            
        dist = np.abs(x_local - xg) / h
        weights = np.where(dist < 1.0, (1 - dist)**2, 0.0)
        
        # Solve weighted least squares
        V = np.ones((len(x_local), local_deg + 1))
        for k in range(1, local_deg + 1):
            V[:, k] = V[:, k-1] * x_local
            
        # Weighted design matrix
        Vw = V * np.sqrt(weights)[:, np.newaxis]
        yw = y_local * np.sqrt(weights)
        
        try:
            # Regularize slightly for stability
            reg = 1e-8
            A = Vw.T @ Vw + reg * np.eye(local_deg + 1)
            B = Vw.T @ yw
            c_local = np.linalg.solve(A, B)
            
            # Evaluate at xg
            y_surrogate[i] = np.sum(c_local * xg**np.arange(local_deg + 1))
        except np.linalg.LinAlgError:
            y_surrogate[i] = np.mean(y_local)
            
    # 3. Fit the global degree-5 polynomial to the surrogate
    # This step forces the final output to be a degree-5 polynomial,
    # but it is fitted to a target that captures the local structure.
    
    # Use Ridge Regression with a small lambda to avoid overfitting to noise
    # in the surrogate (which is already smooth, so lambda can be small).
    
    def make_design_matrix(x_vals):
        x_vals = np.asarray(x_vals)
        V = np.ones((len(x_vals), 6))
        for i in range(1, 6):
            V[:, i] = V[:, i-1] * x_vals
        return V
        
    V_grid = make_design_matrix(x_grid)
    
    # We can use the grid points as the training data for the global fit.
    # This is effectively a projection of the surrogate onto the polynomial subspace.
    
    # Choose a small lambda. Since the surrogate is smooth, we don't need
    # heavy regularization.
    lambda_reg = 1e-4
    
    try:
        A_full = V_grid.T @ V_grid + lambda_reg * np.eye(6)
        B_full = V_grid.T @ y_surrogate
        coef_norm = np.linalg.solve(A_full, B_full)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        coef_norm, _, _, _ = np.linalg.lstsq(V_grid, y_surrogate, rcond=None)
        
    if not np.all(np.isfinite(coef_norm)):
        return np.zeros(6)
        
    # 4. Transform normalized coefficients to original x scale
    # y = sum_{i=0}^5 c_norm[i] * ( (x - x_mean)/x_std )^i
    # Let z = (x - x_mean)/x_std = a*x + b
    # where a = 1/x_std, b = -x_mean/x_std
    # (a*x + b)^i = sum_{k=0}^i C(i,k) * (a*x)^k * b^(i-k)
    #              = sum_{k=0}^i C(i,k) * a^k * b^(i-k) * x^k
    # So, coeff_orig[k] = sum_{i=k}^5 c_norm[i] * C(i,k) * a^k * b^(i-k)
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    coeff_orig = np.zeros(6)
    for i in range(6):
        ci = coef_norm[i]
        for k in range(i + 1):
            coeff_orig[k] += ci * comb(i, k) * (a ** k) * (b ** (i - k))
            
    # 5. Final sanity check
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback: simple mean
        coeff_orig = np.zeros(6)
        coeff_orig[0] = np.mean(y)
        
    return coeff_orig
