"""Combined optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead + (1+1)-ES with multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal best_x, best_f, calls_made
            fx = func(np.clip(x, -5, 5))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        def nelder_mead(x0, f0, max_calls, max_time):
            """Run Nelder-Mead simplex from x0 with given call/time budget."""
            nonlocal best_x, best_f, calls_made
            t_start = time.time()
            
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            values = np.zeros(n + 1)
            
            simplex[0] = x0.copy()
            values[0] = f0
            # calls_made already incremented by caller for x0
            
            step = 0.5
            for i in range(1, n + 1):
                if calls_made >= max_calls or time.time() > t_start + max_time:
                    break
                simplex[i] = simplex[0].copy()
                simplex[i][i - 1] += step
                values[i] = call_func(simplex[i])
            
            # Sort simplex
            order = np.argsort(values)
            simplex = simplex[order]
            values = values[order]
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            max_iter = max_calls - (calls_made - (calls_made - (n + 1)))  # rough
            # Actually, let's just loop until budget or time
            while calls_made < max_calls and time.time() < t_start + max_time:
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = call_func(x_ref)
                
                if f_ref < values[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = call_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        values[-1] = f_ref
                elif f_ref < values[-2]:
                    simplex[-1] = x_ref
                    values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < values[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (centroid - simplex[-1])
                    f_con = call_func(x_con)
                    if f_con <= f_ref:
                        simplex[-1] = x_con
                        values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            values[i] = call_func(simplex[i])
                
                # Re-sort
                order = np.argsort(values)
                simplex = simplex[order]
                values = values[order]
                
                # Check for convergence/stagnation
                if values[-1] - values[0] < 1e-10:
                    break
            
            return simplex[0], values[0]

        def es_1plus1(x0, f0, max_calls, max_time):
            """Run (1+1)-ES from x0 with given call/time budget."""
            nonlocal best_x, best_f, calls_made
            t_start = time.time()
            
            x = x0.copy()
            fx = f0
            sigma = 1.0
            sigma_min = 1e-5
            sigma_max = 8.0
            stuck_count = 0
            max_stuck = 50
            
            while calls_made < max_calls and time.time() < t_start + max_time:
                y = x + sigma * rng.standard_normal(dim)
                fy = call_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    stuck_count = 0
                    sigma = min(sigma * 1.3, sigma_max)
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                    stuck_count += 1
                
                if stuck_count >= max_stuck:
                    break
            
            return x, fx

        # Phase 1: Initial exploration
        n_initial = 12
        initial_budget = min(60, budget // 8)
        candidates = []
        
        for i in range(n_initial):
            if calls_made >= initial_budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = np.full(dim, 2.0)
            elif i == 2:
                x = np.full(dim, -2.0)
            elif i == 3:
                x = rng.uniform(-4, 4, dim)
            else:
                x = rng.uniform(-4, 4, dim)
            
            fx = call_func(x)
            candidates.append((fx, x.copy()))
        
        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Alternating NM and ES from top candidates
        remaining_budget = budget - calls_made
        n_restarts = min(6, len(candidates))
        
        for i in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Allocate budget per restart
            restart_budget = max(10, remaining_budget // (n_restarts - i))
            
            x_start = candidates[i][1]
            f_start = candidates[i][0]
            
            # Alternate between NM and ES
            if i % 2 == 0:
                # Nelder-Mead
                x_best, f_best = nelder_mead(x_start, f_start, calls_made + restart_budget, 280.0 - (time.time() - t0))
            else:
                # (1+1)-ES
                x_best, f_best = es_1plus1(x_start, f_start, calls_made + restart_budget, 280.0 - (time.time() - t0))
            
            # Add result to candidates if better
            if f_best < best_f:
                candidates.append((f_best, x_best.copy()))
                candidates.sort(key=lambda t: t[0])
            
            remaining_budget = budget - calls_made
        
        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = call_func(x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
