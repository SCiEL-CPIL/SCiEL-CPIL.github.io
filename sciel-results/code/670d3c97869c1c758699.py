import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Strategy: Nelder-Mead simplex optimization (gradient-free) to handle the
    non-smooth, piecewise-linear nature of the union area metric.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)

    # ---- Keich seed generation ----
    def keich_seed():
        seed_positions = np.zeros(n, dtype=np.float64)
        for i in range(n):
            # Convert i to binary string of length k
            bits = np.array(list(str(bin(i))[2:]), dtype=np.int8)
            bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
            # Calculate position based on bit weights
            # The specific formula from the Keich construction for n=2^k
            seed_positions[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)
        return seed_positions

    seed_positions = keich_seed()
    a = np.arange(1, n + 1) / n  # shape (64,)

    # Precompute y-grid for the objective function
    # Using a moderate number of points for speed during optimization,
    # then refining with a finer grid at the end.
    num_h_opt = 1000
    ys = (np.arange(num_h_opt) + 0.5) / num_h_opt
    ws = (1.0 - ys) / n

    def union_area_fast(x):
        """
        Vectorized union-area over the optimization y-grid.
        x: shape (64,)
        Returns scalar area estimate.
        """
        Ls = x[None, :] + a[None, :] * ys[:, None]  # (num_h_opt, 64)
        sorted_Ls = np.sort(Ls, axis=1)             # (num_h_opt, 64)
        diffs = np.diff(sorted_Ls, axis=1)          # (num_h_opt, 63)
        extensions = np.minimum(ws[:, None], diffs) # (num_h_opt, 63)
        union_lens = ws + extensions.sum(axis=1)    # (num_h_opt,)
        return union_lens.mean()

    def union_area_fine(x, num_h_fine=5000):
        """Higher-resolution area for final evaluation."""
        ys_f = (np.arange(num_h_fine) + 0.5) / num_h_fine
        ws_f = (1.0 - ys_f) / n
        Ls = x[None, :] + a[None, :] * ys_f[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        extensions = np.minimum(ws_f[:, None], diffs)
        union_lens = ws_f + extensions.sum(axis=1)
        return union_lens.mean()

    # ---- Optimization with Nelder-Mead ----
    start_time = time.time()
    max_time = 620.0  # Safety margin

    best_x = seed_positions.copy()
    best_area = union_area_fast(best_x)
    
    # We track the best area on the fine grid for the final return,
    # but use the fast grid for the optimizer.
    best_area_fine = union_area_fine(best_x)

    np.random.seed(123)

    # Restart list: Keich seed + perturbations
    restarts = [seed_positions.copy()]
    for scale in [0.1, 0.05, 0.01]:
        restarts.append(seed_positions + np.random.normal(0, scale, n))
    # Random starts in a reasonable range
    for _ in range(5):
        restarts.append(np.random.uniform(-0.2, 1.2, n))

    for idx, x0 in enumerate(restarts):
        if time.time() - start_time > max_time - 30:
            break
        
        try:
            # Nelder-Mead is robust to non-smoothness
            # maxfev controls the number of function evaluations
            # We need to be careful with maxfev to not run out of time
            # Each eval is ~1ms for n=64, H=1000. 10000 evals = 10s.
            res = minimize(union_area_fast, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'maxfev': 10000, 
                                    'xatol': 1e-8, 'fatol': 1e-10, 
                                    'disp': False})
            
            if res.fun < best_area:
                best_area = res.fun
                best_x = res.x
                best_area_fine = union_area_fine(best_x)
                
        except Exception:
            pass

    # ---- Final Polish with L-BFGS-B on a Smoothed Surrogate or Fine Grid ----
    # Nelder-Mead might stop near a kink. L-BFGS-B can do a final polish
    # if we use a slightly smoothed version of the objective, or just
    # try it on the fine grid directly.
    if time.time() - start_time < max_time - 10:
        def objective_fine(x):
            return union_area_fine(x, num_h_fine=5000)

        try:
            # Use L-BFGS-B with a small step to polish
            res_fine = minimize(objective_fine, best_x, method='L-BFGS-B',
                                options={'maxiter': 50, 'ftol': 1e-12, 
                                         'gtol': 1e-6, 'disp': False})
            if res_fine.fun < best_area_fine:
                best_area_fine = res_fine.fun
                best_x = res_fine.x
        except Exception:
            pass

    # Ensure the result is valid and finite
    if not np.all(np.isfinite(best_x)):
        return seed_positions
        
    return np.array(best_x, dtype=np.float64)
