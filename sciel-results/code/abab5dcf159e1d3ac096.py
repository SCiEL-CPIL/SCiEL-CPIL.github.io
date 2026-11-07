"""Multi-restart Adaptive Coordinate Descent with Nelder-Mead polish for Schwefel f20."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart Adaptive Coordinate Descent with Nelder-Mead polish on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 295.0  # safety margin before 330s kill
        calls = 0

        def F(x):
            nonlocal calls
            if calls >= budget:
                return best_f  # Return current best to avoid wasted calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline - 3.0

        def adaptive_cd(x0, f0, max_calls, step_scale=1.0):
            """
            Adaptive Coordinate Descent with step size increase on improvement.
            """
            nonlocal best_x, best_f, calls
            n = dim
            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # Initial step sizes scaled by step_scale
            steps = np.ones(n) * 0.5 * step_scale
            min_step = 1e-8
            max_step = 4.0

            local_calls = 0
            no_improve_count = 0

            while local_calls < max_calls and time_ok() and calls < budget:
                improved_this_round = False
                for d in range(n):
                    if local_calls >= max_calls or not time_ok() or calls >= budget:
                        break
                    
                    # Try positive step
                    y = x.copy()
                    y[d] += steps[d]
                    y = np.clip(y, -5, 5)
                    fy = F(y)
                    local_calls += 1
                    
                    if fy < fx:
                        x = y.copy()
                        fx = fy
                        improved_this_round = True
                        # Increase step size to accelerate convergence
                        steps[d] = min(steps[d] * 1.5, max_step)
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        # Try negative step if positive failed
                        y = x.copy()
                        y[d] -= steps[d]
                        y = np.clip(y, -5, 5)
                        fy = F(y)
                        local_calls += 1
                        if fy < fx:
                            x = y.copy()
                            fx = fy
                            improved_this_round = True
                            # Increase step size to accelerate convergence
                            steps[d] = min(steps[d] * 1.5, max_step)
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                        else:
                            # Both directions failed, reduce step size for this dimension
                            steps[d] *= 0.5
                            if steps[d] < min_step:
                                steps[d] = min_step

                if not improved_this_round:
                    no_improve_count += 1
                    # Try a random kick to escape local minima
                    kick_scale = max(0.05, 0.2 * (1.0 / (1.0 + no_improve_count)))
                    x = x + rng.standard_normal(n) * kick_scale
                    x = np.clip(x, -5, 5)
                    fx = F(x)
                    local_calls += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    
                    # Reset no_improve if kick helped
                    if fx < best_f + 1e-6:
                        no_improve_count = 0
                    
                    if steps.min() < min_step * 10 and no_improve_count > 5:
                        break
                else:
                    no_improve_count = 0

            return x, fx

        def nelder_mead_polish(x0, f0):
            """
            Use SciPy's Nelder-Mead for final high-precision polish.
            """
            nonlocal best_x, best_f, calls
            
            try:
                res = minimize(
                    F, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 
                             'adaptive': True}
                )
                if hasattr(res, 'fun') and np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            return best_x, best_f

        # Phase 1: Multi-restart Adaptive Coordinate Descent
        # Budget allocation: ~350 calls for CD, ~150 for NM
        calls_reserved_for_nm = 150
        cd_budget = budget - calls_reserved_for_nm
        
        # Number of restarts based on available CD budget
        # Each restart gets ~70 calls for CD
        calls_per_restart = 70
        n_restarts = max(1, min(6, cd_budget // calls_per_restart))
        
        # Generate diverse starting points
        start_points = []
        start_points.append(np.zeros(dim))
        # Add points spread across the domain
        for i in range(n_restarts - 1):
            # Use stratified sampling for better coverage
            pts = rng.uniform(-4, 4, dim)
            start_points.append(pts)
        
        # Shuffle to randomize order
        rng.shuffle(start_points)
        
        for sp in start_points[:n_restarts]:
            if calls >= budget - calls_reserved_for_nm or not time_ok():
                break
            
            # Evaluate start point
            f_sp = F(sp)
            if f_sp < best_f:
                best_f, best_x = f_sp, sp.copy()
            
            # Run adaptive CD from this point
            available = budget - calls - calls_reserved_for_nm
            if available < 15:
                break
            
            # Use larger step scale for earlier restarts (coarse to fine)
            step_scale = 1.0 + 0.5 * (1.0 - start_points.index(sp) / max(1, n_restarts - 1))
            
            adaptive_cd(sp, f_sp, min(calls_per_restart, available), step_scale)

        # Phase 2: Nelder-Mead Polish
        if calls < budget - 5 and time_ok():
            nelder_mead_polish(best_x.copy(), best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
