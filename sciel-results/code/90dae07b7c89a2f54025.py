import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        def nelder_mead(start_x, max_iter):
            """Nelder-Mead simplex optimization starting from start_x."""
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                return best_x, best_f
            
            # Initialize simplex
            alpha = 0.5
            simplex = [start_x.copy()]
            for i in range(d):
                v = start_x.copy()
                v[i] += alpha
                simplex.append(v)
            
            # Evaluate initial simplex
            try:
                f_simplex = [safe_func(x) for x in simplex]
            except StopIteration:
                return best_x, best_f
            
            # Nelder-Mead parameters
            alpha_nm = 1.0   # reflection
            gamma = 2.0      # expansion
            rho = 0.5        # contraction
            sigma_shrink = 0.5  # shrink
            
            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Reflection
                x_ref = centroid + alpha_nm * (centroid - worst)
                try:
                    f_ref = safe_func(x_ref)
                except StopIteration:
                    break
                
                if f_simplex[0] <= f_ref < f_worst:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                elif f_ref < f_simplex[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    try:
                        f_exp = safe_func(x_exp)
                    except StopIteration:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (worst - centroid)
                    try:
                        f_con = safe_func(x_con)
                    except StopIteration:
                        break
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            try:
                                f_simplex[i] = safe_func(simplex[i])
                            except StopIteration:
                                return best_x, best_f
            
            # Return best point found
            return best_x, best_f

        # Multi-start strategy
        # Use a mix of deterministic and random starting points
        start_points = []
        
        # Deterministic starts: corners and midpoints of the box
        start_points.append(np.zeros(d))
        start_points.append(np.full(d, -3.0))
        start_points.append(np.full(d, 3.0))
        start_points.append(np.full(d, -1.0))
        start_points.append(np.full(d, 1.0))
        
        # Add some random starts
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, d))
        
        # Allocate budget across starts
        # First start gets more budget, subsequent starts get less
        n_starts = len(start_points)
        base_budget_per_start = max(30, (budget - 10) // n_starts)
        
        for i, start_x in enumerate(start_points):
            if time.time() > deadline or calls >= budget:
                break
            
            # Give later starts less budget as we get closer to deadline
            remaining_time = deadline - time.time()
            if remaining_time < 5.0:
                break
            
            # Scale max_iter based on remaining budget
            remaining_calls = budget - calls
            if remaining_calls < 20:
                break
            
            max_iter = min(base_budget_per_start, remaining_calls - 5)
            if max_iter < 10:
                break
            
            nelder_mead(start_x, max_iter)
        
        # Final polish with (1+1)-ES if we have budget left
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.1
            while calls < budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
