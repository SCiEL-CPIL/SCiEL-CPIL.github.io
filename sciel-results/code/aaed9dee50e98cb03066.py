# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _mc_volume_fast(base_x, base_y, N=8, n_samples=10000, seed=42):
    """
    Fast Monte Carlo estimate of the volume.
    Used for optimization.
    """
    rng = np.random.default_rng(seed)
    n_tubes = N * N
    
    # Determine bounds
    # x_k + c*i/N + 1/N
    # Max x is max(base_x) + 1/N (since i/N <= 1)
    # Min x is min(base_x)
    X0 = np.min(base_x)
    X1 = np.max(base_x) + 1.0 / N
    Y0 = np.min(base_y)
    Y1 = np.max(base_y) + 1.0 / N
    
    # If box is degenerate, return large value
    if X1 - X0 < 1e-9 or Y1 - Y0 < 1e-9:
        return 1e10
        
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a,b,c) is in any tube
    # Tube k (i,j):
    # x in [base_x[k] + c*i/N, base_x[k] + 1/N + c*i/N]
    # y in [base_y[k] + c*j/N, base_y[k] + 1/N + c*j/N]
    
    i_idx = np.arange(n_tubes) // N
    j_idx = np.arange(n_tubes) % N
    
    # Vectorize over tubes and samples
    # a: (n_samples,), base_x: (n_tubes,)
    # We want to check for each sample if ANY tube contains it.
    
    # Shifts
    x_shifts = c[:, None] * i_idx[None, :] / N  # (n_samples, n_tubes)
    y_shifts = c[:, None] * j_idx[None, :] / N
    
    x_lefts = base_x[None, :] + x_shifts
    x_rights = x_lefts + 1.0 / N
    y_bottoms = base_y[None, :] + y_shifts
    y_tops = y_bottoms + 1.0 / N
    
    # Check containment
    # a[:, None] >= x_lefts AND a[:, None] <= x_rights
    in_x = (a[:, None] >= x_lefts) & (a[:, None] <= x_rights)
    in_y = (b[:, None] >= y_bottoms) & (b[:, None] <= y_tops)
    
    in_tube = in_x & in_y
    
    # Any tube?
    covered = np.any(in_tube, axis=1)
    
    fraction = np.mean(covered)
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with N = 8.
    Optimizes the 2D base positions directly to minimize the Monte Carlo union volume.
    """
    N = 8
    n_tubes = N * N
    
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    
    # Initial guess: 1D folding
    def _get_1d_fold(N):
        base = np.zeros(N)
        for i in range(N):
            base[i] = 1.0 / (2 * N) - i / N + (1.0 / (2 * N)) * (i % 2)
        base -= base.min()
        return base
        
    x_1d = _get_1d_fold(N)
    y_1d = _get_1d_fold(N)
    
    base_x_init = x_1d[i_flat]
    base_y_init = y_1d[j_flat]
    
    x0 = np.concatenate([base_x_init, base_y_init])
    
    def objective(params):
        bx = params[:n_tubes]
        by = params[n_tubes:]
        # Shift to remove translation symmetry
        bx = bx - np.min(bx)
        by = by - np.min(by)
        return _mc_volume_fast(bx, by, N, n_samples=5000, seed=123)
        
    # Use Nelder-Mead as it is robust and doesn't need gradients.
    # 128 dimensions is high for Nelder-Mead, but we can try.
    # Alternatively, use L-BFGS-B.
    
    # Let's try L-BFGS-B with the fast MC objective.
    # It should be much faster than exact volume.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-6})
        if res.success:
            best_params = res.x
        else:
            # If it fails, try Nelder-Mead with fewer iterations
            res2 = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            best_params = res2.x
    except Exception:
        best_params = x0
        
    bx = best_params[:n_tubes]
    by = best_params[n_tubes:]
    
    bx = bx - np.min(bx)
    by = by - np.min(by)
    
    positions = np.stack([bx, by], axis=1)
    return positions

# EVOLVE-BLOCK-END
