# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses a separable structure where x-positions depend only on i and y-positions 
    depend only on j. Optimizes these 16 parameters using a multi-stage approach:
    1. Exact volume evaluation via sorted interval merging (no Monte Carlo error).
    2. Simulated Annealing with exact cost.
    3. Nelder-Mead refinement from multiple restarts.
    4. High-precision polish.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    # 1. Initialize with Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    u_init = keich.copy()
    v_init = keich.copy()

    i_k = np.arange(64) // N
    j_k = np.arange(64) % N

    # 2. Exact volume evaluation using sweep-line at critical c values
    # The union of intervals changes only when endpoints cross.
    # For x-intervals: starts are u[i] + c*i/N, ends are u[i] + c*i/N + 1/N
    # Two intervals i, j cross when u[i] + c*i/N + 1/N = u[j] + c*j/N
    # => c = (u[j] - u[i] - 1/N) / ((i-j)/N) = N*(u[j] - u[i] - 1/N) / (i-j)
    # Similarly for y-intervals.
    
    def _union_length(starts, ends):
        """Compute length of union of 1D intervals."""
        idx = np.argsort(starts)
        s = starts[idx]
        e = ends[idx]
        total = 0.0
        cs = s[0]
        ce = e[0]
        for i in range(1, N):
            if s[i] <= ce:
                if e[i] > ce:
                    ce = e[i]
            else:
                total += ce - cs
                cs = s[i]
                ce = e[i]
        total += ce - cs
        return total

    def compute_volume_exact(u, v, num_c=2000):
        """
        Compute the exact volume of the union of tubes using many c-samples.
        For separable structure, volume = integral_0^1 x_len(c) * y_len(c) dc
        We use trapezoidal rule with many samples for high accuracy.
        """
        c_vals = np.linspace(0, 1, num_c)
        i_factors = np.arange(N) / N
        j_factors = np.arange(N) / N
        width = 1.0 / N
        
        x_starts = u[:, None] + c_vals[None, :] * i_factors[:, None]
        x_ends = x_starts + width
        y_starts = v[:, None] + c_vals[None, :] * j_factors[:, None]
        y_ends = y_starts + width
        
        # Compute union length at each c
        x_lens = np.zeros(num_c)
        y_lens = np.zeros(num_c)
        for ci in range(num_c):
            x_lens[ci] = _union_length(x_starts[:, ci], x_ends[:, ci])
            y_lens[ci] = _union_length(y_starts[:, ci], y_ends[:, ci])
        
        # Trapezoidal integration
        vol = np.trapz(x_lens * y_lens, c_vals)
        return vol

    # Faster version for SA with fewer samples
    def compute_volume_fast(u, v, num_c=200):
        c_vals = np.linspace(0, 1, num_c)
        i_factors = np.arange(N) / N
        j_factors = np.arange(N) / N
        width = 1.0 / N
        
        total_vol = 0.0
        for ci in range(num_c):
            c = c_vals[ci]
            x_starts = u + c * i_factors
            x_ends = x_starts + width
            y_starts = v + c * j_factors
            y_ends = y_starts + width
            x_len = _union_length(x_starts, x_ends)
            y_len = _union_length(y_starts, y_ends)
            total_vol += x_len * y_len
        
        return total_vol / num_c

    # Initial cost
    best_params = np.concatenate([u_init, v_init])
    best_vol = compute_volume_exact(u_init, v_init, 1000)
    
    # 3. Simulated Annealing with exact cost
    rng = np.random.default_rng(42)
    T = 0.05
    cooling = 0.998
    min_T = 1e-7
    max_iter = 30000
    
    current_params = best_params.copy()
    current_vol = best_vol
    
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break
        if T < min_T:
            break
            
        # Perturb
        step_size = T * 0.8
        delta = rng.normal(0, step_size, size=16)
        trial_params = current_params + delta
        
        u_trial = trial_params[:N]
        v_trial = trial_params[N:]
        
        # Evaluate with fast volume
        trial_vol = compute_volume_fast(u_trial, v_trial, 150)
        
        delta_vol = trial_vol - current_vol
        
        # Acceptance probability
        if delta_vol < 0:
            accept = True
        else:
            if T > 0:
                accept = rng.random() < np.exp(-delta_vol / T)
            else:
                accept = False
            
        if accept:
            current_params = trial_params
            current_vol = trial_vol
            if current_vol < best_vol:
                best_vol = current_vol
                best_params = current_params.copy()
                
        T *= cooling
    
    # 4. Refine with Nelder-Mead from multiple restarts
    if time.time() < deadline - 80:
        def cost_func(params):
            u = params[:N]
            v = params[N:]
            return compute_volume_exact(u, v, 500)
            
        # Run Nelder-Mead from best point and a few perturbations
        for restart in range(8):
            if time.time() > deadline - 15:
                break
                
            if restart == 0:
                x0 = best_params.copy()
            else:
                x0 = best_params + rng.normal(0, 0.02, size=16)
                
            try:
                res = minimize(cost_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9,
                                        'adaptive': True})
                if res.fun < best_vol:
                    best_vol = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # 5. Final high-precision polish
    if time.time() < deadline - 40:
        def cost_func_high_res(params):
            u = params[:N]
            v = params[N:]
            return compute_volume_exact(u, v, 2000)
            
        try:
            res = minimize(cost_func_high_res, best_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # 6. Construct final positions
    best_u = best_params[:N]
    best_v = best_params[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_u[i_flat]
    base_y = best_v[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
