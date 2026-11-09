"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Strategy:
    # 1. Estimate the conditional expectation E[Y|X] using a Gaussian Process (GPR)
    #    with an RBF kernel. This acts as a powerful non-parametric denoiser that
    #    adapts its smoothness to the data.
    # 2. Fit a degree-5 polynomial to these denoised pseudo-labels.
    #    This projects the smooth non-parametric estimate onto the required
    #    parametric form, avoiding the bias of fitting directly to noisy data
    #    or the fixed bandwidth limitations of simple kernel regression.
    
    # Define a grid for evaluating the conditional mean
    # Cover the range of x with some buffer for potential extrapolation/edge effects
    x_min, x_max = np.min(x), np.max(x)
    buffer = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
    x_min -= buffer
    x_max += buffer
    
    num_grid = 200
    x_grid = np.linspace(x_min, x_max, num_grid)
    
    # GPR Implementation
    # Kernel: RBF k(x, x') = sigma_f^2 * exp(-0.5 * (x - x')^2 / l^2)
    # We need to choose sigma_f (signal std) and l (length scale).
    # Heuristics:
    # sigma_f ~ std(y)
    # l ~ range(x) / sqrt(n) * factor, or just a fraction of range.
    
    sigma_f = np.std(y)
    if sigma_f < 1e-12:
        sigma_f = 1.0
        
    # Length scale heuristic: typical distance between points
    # If n is large, l should be small. If n is small, l should be large.
    # Let's try a few length scales and pick the one that minimizes a simple
    # validation-like metric? No, we don't have validation data.
    # Use a robust default: l = (x_max - x_min) / 10.0
    # Or better, estimate from data spread.
    # Let's use a few candidate length scales and average the predictions?
    # Or just pick a good one.
    
    # Let's use a length scale that is roughly the 10th percentile of pairwise distances?
    # That's expensive O(n^2).
    # Simple heuristic: l = (x_max - x_min) / 15.0
    l = (x_max - x_min) / 15.0
    if l < 1e-6:
        l = 0.1
    
    # Noise level: assume small noise relative to signal
    # In Lorenz-96, the fast scale is chaotic, so noise is significant.
    # Let's estimate noise variance as 1% of signal variance? Or use a small epsilon.
    # For GPR to be stable, we need noise variance > 0.
    # Let's try sigma_n^2 = 0.01 * sigma_f^2
    sigma_n = 0.1 * sigma_f
    
    # Build Kernel Matrix K (n x n)
    # To save memory and time for large n, we might need to subsample if n is huge.
    # But typically n is manageable (thousands).
    # K[i,j] = sigma_f^2 * exp(-0.5 * (x[i]-x[j])^2 / l^2)
    
    # Check time
    if time.time() - t_start > deadline - 10.0:
         # Fallback to simple linear fit if GPR setup is too slow
         try:
             A_lin = np.vander(x, 2, increasing=True)
             coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
             out = np.zeros(6)
             out[:2] = coeff_lin
             return out
         except Exception:
             return np.zeros(6, dtype=np.float64)
    
    # Compute K
    # Vectorized computation of distances
    # x is (n,), x_grid is (num_grid,)
    # We need K for training points to predict at x_grid.
    # K_train: (n, n)
    # K_test: (num_grid, n)
    
    # Memory check: if n > 5000, n^2 is 25M floats ~ 200MB. Might be okay.
    # If n > 10000, 100M floats ~ 800MB. Risky.
    # Let's subsample training points if n > 5000 to ensure speed/memory.
    max_train_pts = 5000
    if n > max_train_pts:
        idx = np.random.choice(n, max_train_pts, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y
        max_train_pts = n
        
    # Recalculate l based on subsampled points? No, keep same l.
    
    # Compute K_train
    # dist2 = (x_sub[:, None] - x_sub[None, :])**2
    # This creates a (n_sub, n_sub) matrix.
    
    if max_train_pts > 10000: # Extra safety
        # Just do simple regression if too large
        A = np.vander(x, 6, increasing=True)
        coeff_asc, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coeff_asc

    dist2_train = (x_sub[:, None] - x_sub[None, :])**2
    K_train = sigma_f**2 * np.exp(-0.5 * dist2_train / l**2)
    
    # Add noise to diagonal
    K_train += (sigma_n**2) * np.eye(max_train_pts)
    
    # Solve for alpha: K_train * alpha = y_sub
    try:
        alpha = np.linalg.solve(K_train, y_sub)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        try:
            alpha, _, _, _ = np.linalg.lstsq(K_train, y_sub, rcond=None)
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    # Check time
    if time.time() - t_start > deadline - 5.0:
         # Fallback to simple regression if solving was slow
         A = np.vander(x, 6, increasing=True)
         coeff_asc, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
         return coeff_asc

    # Compute K_test: (num_grid, n_sub)
    dist2_test = (x_grid[:, None] - x_sub[None, :])**2
    K_test = sigma_f**2 * np.exp(-0.5 * dist2_test / l**2)
    
    # Predict mean
    y_grid = K_test @ alpha
    
    # Check time
    if time.time() - t_start > deadline - 2.0:
         A = np.vander(x_grid, 6, increasing=True)
         coeff_asc, _, _, _ = np.linalg.lstsq(A, y_grid, rcond=None)
         return coeff_asc

    # Fit degree-5 polynomial to y_grid
    A = np.vander(x_grid, 6, increasing=True)
    try:
        coeff_asc, residuals, rank, sv = np.linalg.lstsq(A, y_grid, rcond=None)
        
        if not np.all(np.isfinite(coeff_asc)):
             A_lin = np.vander(x_grid, 2, increasing=True)
             coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y_grid, rcond=None)
             coeff_asc = np.zeros(6)
             coeff_asc[:2] = coeff_lin
             
    except Exception:
        try:
            A_lin = np.vander(x, 2, increasing=True)
            coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            out = np.zeros(6)
            out[:2] = coeff_lin
            return out
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    return coeff_asc
# EVOLVE-BLOCK-END
