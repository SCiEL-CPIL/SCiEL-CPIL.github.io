# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimize 64 tube base positions to minimize the 3D union volume.
    
    The volume of the union of the 64 sheared tubes is estimated by Monte Carlo.
    We use scipy's L-BFGS-B to directly minimize the MC-estimated volume,
    starting from the Keich binary construction.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s kill

    # Keich binary construction for initial guess
    pos = int(np.log2(N))  # 3
    def keich_offsets(N, pos):
        j_terms = np.arange(1, pos + 1)
        ks = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            ks[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return ks

    keich_base = keich_offsets(N, pos)

    # Build initial positions from Keich construction
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    i_grid, j_grid = np.meshgrid(i_vals, j_vals, indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    bx_init = keich_base[i_flat]
    by_init = keich_base[j_flat]
    pos_init = np.stack([bx_init, by_init], axis=1)

    def volume_estimate(positions, rng):
        """
        Compute the MC-estimated volume of the union of tubes.
        positions: (64, 2) array of (x_k, y_k)
        """
        if not np.all(np.isfinite(positions)):
            return np.inf
        
        xk = positions[:, 0]
        yk = positions[:, 1]
        
        # Compute box bounds
        # For tube k: i = k//8, j = k%8
        # x range: [x_k, x_k + (i+1)/8]
        # y range: [y_k, y_k + (j+1)/8]
        i_arr = np.arange(64) // 8
        j_arr = np.arange(64) % 8
        
        X0 = np.min(xk)
        X1 = np.max(xk + (i_arr + 1) / 8.0)
        Y0 = np.min(yk)
        Y1 = np.max(yk + (j_arr + 1) / 8.0)
        
        if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
            return np.inf
        
        # Monte Carlo sampling
        a = rng.uniform(X0, X1, 500000)
        b = rng.uniform(Y0, Y1, 500000)
        c = rng.uniform(0, 1, 500000)
        
        # Check if each point is inside at least one tube
        # For tube k: x in [x_k + c*i_k/8, x_k + c*i_k/8 + 1/8]
        #              y in [y_k + c*j_k/8, y_k + c*j_k/8 + 1/8]
        
        # Vectorized check: for each tube, check if point is inside
        # This is expensive, so we do it in chunks
        inside_any = np.zeros(500000, dtype=bool)
        
        for k in range(64):
            i_k = k // 8
            j_k = k % 8
            x_lo = xk[k] + c * i_k / 8.0
            x_hi = x_lo + 1.0 / 8.0
            y_lo = yk[k] + c * j_k / 8.0
            y_hi = y_lo + 1.0 / 8.0
            
            in_x = (a >= x_lo) & (a <= x_hi)
            in_y = (b >= y_lo) & (b <= y_hi)
            inside_any |= (in_x & in_y)
            
            if time.time() > deadline - 5:
                break
        
        fraction = np.mean(inside_any)
        V = fraction * (X1 - X0) * (Y1 - Y0)
        return V

    # Use a fixed rng for reproducibility in the objective
    rng_obj = np.random.default_rng(12345)
    
    def objective_flat(flat_pos):
        positions = flat_pos.reshape(64, 2)
        return volume_estimate(positions, rng_obj)
    
    # Initial evaluation
    best_V = volume_estimate(pos_init, np.random.default_rng(12345))
    best_pos = pos_init.copy()
    
    # Run L-BFGS-B optimization
    try:
        result = minimize(
            objective_flat,
            pos_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 200, 'disp': False}
        )
        
        if time.time() < deadline - 10:
            if result.fun < best_V and np.all(np.isfinite(result.x)):
                best_V = result.fun
                best_pos = result.x.reshape(64, 2)
        
    except Exception as e:
        pass
    
    # Try a few more random restarts if time permits
    rng_restart = np.random.default_rng(999)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Perturb the current best
        noise_scale = 0.05 / (restart + 1)
        perturbed = best_pos + rng_restart.normal(0, noise_scale, size=best_pos.shape)
        
        try:
            result2 = minimize(
                objective_flat,
                perturbed.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100, 'disp': False}
            )
            
            if time.time() < deadline - 10:
                if result2.fun < best_V and np.all(np.isfinite(result2.x)):
                    best_V = result2.fun
                    best_pos = result2.x.reshape(64, 2)
        except Exception:
            break
    
    # Final validation
    if not np.all(np.isfinite(best_pos)):
        # Fallback to initial
        best_pos = pos_init
    
    assert best_pos.shape == (64, 2)
    assert np.all(np.isfinite(best_pos))
    
    return best_pos

# EVOLVE-BLOCK-END
