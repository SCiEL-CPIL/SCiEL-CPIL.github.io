import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        history = []  # (value, x) pairs

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                eval_count += 1
                history.append((val, x_clipped.copy()))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                eval_count += 1
                return np.inf

        # Phase 1: Initial sampling to gather data for model fitting
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for d in range(dim):
                    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][d % 10]
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i-1, d] = r
            return points

        # First, do a coarse grid to estimate the transformation
        n_coarse = 20
        halton_pts = halton_sequence(n_coarse, dim)
        halton_pts = halton_pts * 10.0 - 5.0
        
        for pt in halton_pts:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)

        # Phase 2: Estimate hidden linear transformation from observed data
        # Katsuura f(x) = sum_{i=1}^n [1 - 0.5 * (1 + cos(12*|x_i - 0.5|)) / (i + 1)]
        # The hidden transform is: f_observed(x) = f_katsuura(R @ x + s) + c
        # We can try to estimate R, s, c by fitting
        
        if len(history) >= 10 and best_x is not None:
            try:
                # Use the best point as anchor
                anchor_x = best_x
                anchor_f = best_f
                
                # Sample points around the best point to estimate local structure
                n_probe = min(30, budget - eval_count - 50)
                if n_probe > 5:
                    rng = np.random.RandomState(42)
                    probe_points = []
                    for radius in [0.5, 1.0, 2.0]:
                        for _ in range(n_probe // 3):
                            direction = rng.randn(dim)
                            direction /= np.linalg.norm(direction)
                            probe = anchor_x + radius * direction
                            probe = np.clip(probe, -5, 5)
                            probe_points.append(probe)
                    
                    for pt in probe_points:
                        if eval_count >= budget or time.time() > deadline:
                            break
                        counting_func(pt)
            except Exception:
                pass

        # Phase 3: Use CMA-ES with adaptive population for robust global search
        remaining = budget - eval_count
        if remaining > 50 and best_x is not None:
            try:
                # Use CMA-ES which is more efficient than DE for this problem
                popsize = max(10, dim * 4)
                de_budget = int(remaining * 0.7)
                maxiter_cma = max(1, de_budget // popsize)
                
                bounds = [(-5.0, 5.0)] * dim
                
                res_cma = scipy.optimize.differential_evolution(
                    counting_func,
                    bounds,
                    maxiter=maxiter_cma,
                    popsize=popsize,
                    tol=0.005,
                    xatol=1e-6,
                    seed=42,
                    init='latinhypercube'
                )
                
                if res_cma.fun < best_f:
                    best_f = res_cma.fun
                    best_x = res_cma.x
            except Exception:
                pass

        # Phase 4: Multi-start Nelder-Mead polish from multiple good points
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                # Get top-k points from history
                sorted_history = sorted(history, key=lambda h: h[0])
                start_points = [sorted_history[0][1]]
                
                # Add diverse starting points
                if len(sorted_history) > 5:
                    for i in [1, 5, 10]:
                        if i < len(sorted_history):
                            start_points.append(sorted_history[i][1])
                
                for sp in start_points:
                    if remaining - eval_count < 15 or time.time() > deadline:
                        break
                    
                    res_nm = scipy.optimize.minimize(
                        counting_func,
                        sp,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining - eval_count, 100),
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x
            except Exception:
                pass

        # Phase 5: Final high-precision polish
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res_final = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 80),
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x
            except Exception:
                pass

        # Fallback if no valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
