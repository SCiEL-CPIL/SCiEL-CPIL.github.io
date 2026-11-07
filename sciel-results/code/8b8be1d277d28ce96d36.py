"""Combined optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer combining:
    - Halton quasi-random scan for initial basin coverage
    - Multi-start Simulated Annealing for global exploration (exploits multimodality)
    - Nelder-Mead + coordinate descent for local refinement
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def halton_sequence(n, base):
            """Generate n Halton sequence points in [0,1)."""
            result = np.zeros(n)
            for idx in range(1, n + 1):
                i = idx
                f = 1.0
                while i > 0:
                    i, remainder = divmod(i, base)
                    f /= base
                    result[idx - 1] += f * remainder
            return result

        # Phase 1: Halton quasi-random scan for diverse initial coverage
        n_scan = min(50, budget // 10)
        halton_points = []
        for d in range(dim):
            halton_points.append(halton_sequence(n_scan, d + 2))
        
        for i in range(n_scan):
            if calls >= budget or time.time() > deadline:
                break
            x = np.array([halton_points[d][i] for d in range(dim)])
            x = x * 10.0 - 5.0  # Map [0,1] to [-5,5]
            eval_and_track(x)

        # Also evaluate origin and a few structured points
        structured_points = [
            np.zeros(dim),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
        ]
        for sp in structured_points:
            if calls >= budget or time.time() > deadline:
                break
            eval_and_track(sp)

        # Phase 2: Multi-start Simulated Annealing for global exploration
        # This exploits the multimodality of f22 better than CMA-ES by using
        # stochastic global exploration to find deeper basins
        if calls < budget - 50 and time.time() < deadline:
            rng = np.random.default_rng(42)
            
            # Allocate 40% of remaining budget to SA
            remaining = budget - calls
            sa_budget = int(remaining * 0.4)
            
            if sa_budget > 20:
                # Multiple SA restarts from different starting points
                n_restarts = min(5, sa_budget // 10)
                restarts_per_sa = sa_budget // n_restarts
                
                # Starting points: best from scan + random points
                start_points = [best_x.copy()]
                for _ in range(n_restarts - 1):
                    start_points.append(rng.uniform(-4, 4, dim))
                
                for start_pt in start_points:
                    if calls >= budget - 30 or time.time() > deadline:
                        break
                    
                    # Simulated Annealing parameters
                    T_initial = 5.0
                    T_final = 1e-4
                    n_iter = min(restarts_per_sa, budget - calls - 20)
                    
                    if n_iter <= 0:
                        continue
                    
                    # Temperature schedule: geometric cooling
                    cooling_rate = (T_final / T_initial) ** (1.0 / n_iter)
                    
                    x_curr = start_pt.copy()
                    x_curr = np.clip(x_curr, -5.0, 5.0)
                    f_curr = eval_and_track(x_curr)
                    
                    T = T_initial
                    
                    for _ in range(n_iter):
                        if calls >= budget - 20 or time.time() > deadline:
                            break
                        
                        # Generate neighbor
                        step_size = 0.5 * T / T_initial + 0.1
                        x_new = x_curr + rng.normal(0, step_size, dim)
                        x_new = np.clip(x_new, -5.0, 5.0)
                        
                        f_new = eval_and_track(x_new)
                        
                        # Accept or reject
                        if f_new < f_curr:
                            x_curr = x_new
                            f_curr = f_new
                        else:
                            # Accept with probability exp(-(f_new - f_curr) / T)
                            if rng.random() < np.exp(-(f_new - f_curr) / T):
                                x_curr = x_new
                                f_curr = f_new
                        
                        T *= cooling_rate
                    
                    # Track best from SA
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()

        # Phase 3: Local refinement with Nelder-Mead
        if calls < budget - 30 and time.time() < deadline:
            # Use Nelder-Mead starting from the best point
            x0 = best_x.copy()
            
            def wrapper(x):
                nonlocal calls
                if calls >= budget or time.time() > deadline:
                    return best_f
                x_clipped = np.clip(x, -5.0, 5.0)
                calls += 1
                val = func(x_clipped)
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            
            # Nelder-Mead with reasonable initial simplex
            options = {
                'maxiter': min(100, (budget - calls) // 3),
                'xatol': 1e-6,
                'fatol': 1e-8,
            }
            
            try:
                result = minimize(wrapper, x0, method='Nelder-Mead', options=options)
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 4: Coordinate descent fine-tuning
        if calls < budget - 15 and time.time() < deadline:
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Multiple rounds of coordinate descent
            for round_idx in range(3):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                
                improved = False
                for d in range(dim):
                    if calls >= budget - 10 or time.time() > deadline:
                        break
                    
                    # Golden-section search along coordinate d
                    lo = max(-5.0, x_curr[d] - 1.0)
                    hi = min(5.0, x_curr[d] + 1.0)
                    
                    if lo >= hi:
                        continue
                    
                    gr = (np.sqrt(5) - 1) / 2
                    a, b = lo, hi
                    c = b - gr * (b - a)
                    d_val = a + gr * (b - a)
                    
                    def func_1d(val, d=d, x=x_curr):
                        nonlocal calls
                        if calls >= budget or time.time() > deadline:
                            return f_curr
                        x_temp = x.copy()
                        x_temp[d] = val
                        x_temp = np.clip(x_temp, -5.0, 5.0)
                        calls += 1
                        val = func(x_temp)
                        if val < best_f:
                            best_f = val
                            best_x = x_temp.copy()
                        return val
                    
                    fc = func_1d(c)
                    fd = func_1d(d_val)
                    
                    for _ in range(15):
                        if calls >= budget - 5 or time.time() > deadline:
                            break
                        if fc < fd:
                            b = d_val
                            d_val = c
                            fd = fc
                            c = b - gr * (b - a)
                            if abs(c - d_val) < 1e-10:
                                break
                            fc = func_1d(c)
                        else:
                            a = c
                            c = d_val
                            fc = fd
                            d_val = a + gr * (b - a)
                            if abs(c - d_val) < 1e-10:
                                break
                            fd = func_1d(d_val)
                    
                    opt_val = c if fc < fd else d_val
                    opt_f = min(fc, fd)
                    
                    if opt_f < f_curr:
                        x_curr[d] = opt_val
                        f_curr = opt_f
                        improved = True
                
                if not improved:
                    break
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()

        # Phase 5: If budget remains, do a few more random restarts with local refinement
        rng_final = np.random.default_rng(12345)
        while calls < budget - 30 and time.time() < deadline:
            # Random restart
            x_start = rng_final.uniform(-4, 4, dim)
            eval_and_track(x_start)
            
            # Quick local refinement from this point
            if calls < budget - 15 and time.time() < deadline:
                x_local = x_start.copy()
                for d in range(dim):
                    if calls >= budget - 10 or time.time() > deadline:
                        break
                    # Simple 3-point search along coordinate d
                    center = x_local[d]
                    step = 0.5
                    candidates = [
                        np.clip(center - step, -5, 5),
                        center,
                        np.clip(center + step, -5, 5),
                    ]
                    best_val = float('inf')
                    best_cand = center
                    for cand in candidates:
                        if calls >= budget - 5 or time.time() > deadline:
                            break
                        x_temp = x_local.copy()
                        x_temp[d] = cand
                        x_temp = np.clip(x_temp, -5.0, 5.0)
                        val = eval_and_track(x_temp)
                        if val < best_val:
                            best_val = val
                            best_cand = cand
                    x_local[d] = best_cand
                
                if best_val < best_f:
                    best_f = best_val
                    best_x = x_local.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
