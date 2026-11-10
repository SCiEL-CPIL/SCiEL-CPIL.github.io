import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_volume_exact(positions, N=8, n_samples=100000):
    """
    Compute union volume using Monte Carlo estimation.
    """
    if positions.shape != (N * N, 2):
        return 1e10
    
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    i_vals = np.repeat(np.arange(N), N)
    j_vals = np.tile(np.arange(N), N)
    
    X0 = np.min(x_coords)
    X1 = np.max(x_coords + (i_vals + 1) / N)
    Y0 = np.min(y_coords)
    Y1 = np.max(y_coords + (j_vals + 1) / N)
    
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return 1e10
    
    rng = np.random.default_rng(42)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(N * N):
        i_k = k // N
        j_k = k % N
        x_k = x_coords[k]
        y_k = y_coords[k]
        
        a_min = x_k + c * i_k / N
        a_max = a_min + 1.0 / N
        b_min = y_k + c * j_k / N
        b_max = b_min + 1.0 / N
        
        mask = (a >= a_min) & (a <= a_max) & (b >= b_min) & (b <= b_max)
        inside |= mask
    
    fraction = np.mean(inside)
    volume = fraction * area
    return volume

def _volume_from_X(X, N=8, n_samples=100000):
    """
    Compute volume from 1D array X of length N.
    positions[k] = (X[i], X[j]) where k = i*N + j
    """
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X[i_flat]
    base_y = X[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return _compute_volume_exact(positions, N=N, n_samples=n_samples)

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    N = 8
    
    # Generate diverse initial guesses
    inits = []
    
    # Zero
    inits.append(np.zeros(N))
    
    # Linear
    inits.append(np.linspace(0, 0.5, N))
    inits.append(-np.linspace(0, 0.5, N))
    
    # Symmetric
    inits.append(np.array([0.25, 0.15, 0.05, 0.0, -0.0, -0.05, -0.15, -0.25]))
    
    # Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        inits.append(rng.uniform(-0.3, 0.3, N))
    
    # Structured guesses based on overlap optimization
    # Try to make tubes overlap as much as possible
    # At c=0, all tubes are at their base positions
    # At c=1, tube (i,j) is at (x_k + i/8, y_k + j/8)
    # For maximum overlap, we want the spread to be minimal
    
    # Try: X[i] = -i/(2*N) so that at c=0.5, all x-positions are centered
    inits.append(-np.arange(N) / (2 * N))
    inits.append(np.arange(N) / (2 * N) - 0.5)
    
    best_X = None
    best_vol = float('inf')
    
    # First pass: quick Nelder-Mead from multiple starts
    for idx, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(_volume_from_X, init, args=(N, 50000), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_vol:
                best_vol = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    if best_X is None:
        best_X = np.zeros(N)
    
    # Second pass: polish with more samples and tighter tolerance
    if time.time() < deadline - 30:
        try:
            res2 = minimize(_volume_from_X, best_X, args=(N, 100000), method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < best_vol:
                best_vol = res2.fun
                best_X = res2.x.copy()
        except Exception:
            pass
    
    # Third pass: try differential evolution if time permits
    if time.time() < deadline - 90:
        try:
            bounds = [(-1, 1)] * N
            de_res = differential_evolution(_volume_from_X, bounds, args=(N, 30000),
                                            maxiter=50, popsize=15, seed=42,
                                            tol=1e-8, polish=False)
            if de_res.fun < best_vol:
                best_vol = de_res.fun
                best_X = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 10:
        try:
            res_final = minimize(_volume_from_X, best_X, args=(N, 150000), method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_final.fun < best_vol:
                best_X = res_final.x.copy()
        except Exception:
            pass
    
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_X[i_flat]
    base_y = best_X[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
