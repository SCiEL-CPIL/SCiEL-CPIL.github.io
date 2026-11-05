"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.interpolate as interp

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit
    
    # 1. Prepare data
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Remove any NaNs or infs if present (though inputs are usually clean)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        return np.zeros(6)
    
    # 2. Fit a piecewise cubic spline
    # Use quantile-based knots to better capture local features
    # Number of knots: scale with data size, but cap to avoid overfitting/instability
    n_knots = min(len(x) // 5, 50)
    if n_knots < 5:
        n_knots = 5
        
    # Generate quantile knots
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
    knots = np.quantile(x, quantiles)
    
    try:
        # Fit cubic spline with specified interior knots
        # s=0 for exact interpolation, or small s for smoothing
        # Using s=0 is usually safe if data is noisy-free enough, but let's try a small smoothing
        # Actually, for "unresolved tendency" which is deterministic in the model, exact fit is good
        # But we want to generalize to validation. Smoothing helps.
        # Let's use a simple smoothing parameter based on noise level estimate or just a small constant
        # Given the "oscillatory" nature, we want to preserve the shape.
        cs = interp.CubicSpline(x, y, knots=knots, bc_type='not-a-knot')
    except Exception:
        # Fallback to polynomial fit if spline fails
        try:
            coeffs = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeffs
        except:
            return np.zeros(6)
    
    # 3. Project the spline onto the degree-5 monomial basis
    # We sample the spline at many points within the domain of x
    # to solve the least squares problem: P(x) ~ Spline(x)
    
    # Define evaluation points
    # Use points spread across the range of x, perhaps weighted by data density
    # Or just uniform in the min/max range of x
    x_min, x_max = np.min(x), np.max(x)
    
    # If range is zero, handle edge case
    if x_max == x_min:
        return np.array([0, 0, 0, 0, 0, np.mean(y)])
        
    # Number of sampling points for projection
    n_samples = min(1000, len(x) * 2)
    x_proj = np.linspace(x_min, x_max, n_samples)
    
    # Evaluate spline at these points
    y_proj = cs(x_proj)
    
    # Build Vandermonde matrix for degree 5 polynomial
    # Columns: 1, x, x^2, x^3, x^4, x^5
    A = np.vander(x_proj, N=6, increasing=True)
    
    # Solve least squares
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, y_proj, rcond=None)
        
        # Check time
        if time.time() > deadline - 5:
            return coeffs
            
        return coeffs
    except Exception:
        # Fallback to direct polyfit if projection fails
        try:
            coeffs = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeffs
        except:
            return np.zeros(6)

# EVOLVE-BLOCK-END
