import numpy as np
import time
from scipy.optimize import minimize

def _compute_volume(positions, N=8, n_samples=500000, seed=12345):
    """
    Compute the Monte Carlo volume estimate for given positions.
    positions: (64, 2) array.
    """
    rng = np.random.default_rng(seed)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x_starts = positions[:, 0]
    x_ends = positions[:, 0] + (i_flat + 1) / N
    y_starts = positions[:, 1]
    y_ends = positions[:, 1] + (j_flat + 1) / N
    
    X0 = np.min(x_starts)
    X1 = np.max(x_ends)
    Y0 = np.min(y_starts)
    Y1 = np.max(y_ends)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(N * N):
        i_k = i_flat[k]
        j_k = j_flat[k]
        
        x_lo = positions[k, 0] + c * i_k / N
        x_hi = x_lo + 1.0 / N
        y_lo = positions[k, 1] + c * j_k / N
        y_hi = y_lo + 1.0 / N
        
        inside |= (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
        
    frac = np.mean(inside)
    return frac * (X1 - X0) * (Y1 - Y0)

def _keich_1d(N=8):
    """Compute Keich binary construction for 1D positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    # Initial guess: Keich construction
    base_x_1d = _keich_1d(N)
    base_y_1d = _keich_1d(N)
    
    # The 16 parameters are the 8 x-positions and 8 y-positions
    x0 = base_x_1d.copy()
    y0 = base_y_1d.copy()
    initial_params = np.concatenate([x0, y0])
    
    def objective(params):
        x_1d = params[:N]
        y_1d = params[N:]
        
        # Construct full positions array
        x_grid = np.tile(x_1d[:, np.newaxis], (1, N))
        y_grid = np.tile(y_1d[np.newaxis, :], (N, 1))
        
        positions = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
        
        V = _compute_volume(positions, N=N, n_samples=500000)
        return V
    
    # Run optimization
    # We use Nelder-Mead as it doesn't require gradients and is robust.
    # We set maxiter high enough to converge within time limits.
    # Each iteration is ~0.1s. 2000 iterations is ~200s.
    # We have ~590s.
    
    best_params = initial_params
    best_vol = objective(initial_params)
    
    try:
        res = minimize(objective, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_vol:
            best_vol = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # If we have time, try a second phase with Powell or a tighter Nelder-Mead
    if time.time() < deadline - 30:
        try:
            res2 = minimize(objective, best_params, method='Powell', 
                           options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10})
            if res2.fun < best_vol:
                best_vol = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Construct final positions
    x_1d = best_params[:N]
    y_1d = best_params[N:]
    x_grid = np.tile(x_1d[:, np.newaxis], (1, N))
    y_grid = np.tile(y_1d[np.newaxis, :], (N, 1))
    final_positions = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
    
    return final_positions
