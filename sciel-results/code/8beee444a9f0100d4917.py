"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        best_x_global = None
        best_f_global = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def nelder_mead(x0, f0):
            nonlocal calls, best_x_global, best_f_global
            
            # Initialize simplex
            # Start with a small perturbation in each dimension
            simplex = [x0.copy()]
            for i in range(self.dim):
                step = 0.1 * (1 + 0.1 * i) # Slightly increasing steps to avoid degeneracy
                x_new = x0.copy()
                x_new[i] += step
                simplex.append(x_new)
            
            # Evaluate simplex
            f_values = [f0]
            for i in range(1, self.dim + 1):
                if not time_ok() or calls >= self.budget - 5:
                    break
                fx = func(simplex[i])
                calls += 1
                f_values.append(fx)
                if fx < best_f_global:
                    best_f_global = fx
                    best_x_global = simplex[i].copy()
            
            # If we ran out of budget during init, return best found
            if len(f_values) < self.dim + 1:
                return best_x_global, best_f_global

            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink

            max_iter = 200
            for _ in range(max_iter):
                if not time_ok() or calls >= self.budget - 5:
                    break

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check convergence
                if f_values[-1] - f_values[0] < 1e-10:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                if f_r < best_f_global:
                    best_f_global = f_r
                    best_x_global = x_r.copy()

                if f_values[0] <= f_r < f_values[-2]:
                    # Replace worst with reflected
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f_global:
                        best_f_global = f_e
                        best_x_global = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f_global:
                        best_f_global = f_c
                        best_x_global = x_c.copy()

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_new = func(simplex[i])
                            calls += 1
                            f_values[i] = f_new
                            if f_new < best_f_global:
                                best_f_global = f_new
                                best_x_global = simplex[i].copy()
                            if calls >= self.budget - 5 or not time_ok():
                                break

            return best_x_global, best_f_global

        # Multi-start strategy
        # For Sharp Ridge, the optimum is often far from the origin.
        # We use a mix of random starts and a grid-like spread to ensure coverage.
        
        # 1. Origin
        starts = [np.zeros(self.dim)]
        
        # 2. Random points in [-4, 4]^5
        n_random = 8
        for _ in range(n_random):
            starts.append(rng.uniform(-4, 4, self.dim))
            
        # 3. Points along axes to catch ridge alignment if it's axis-aligned initially
        # (Though rotation is hidden, this helps if the shift is small)
        # Actually, random is better for hidden rotation. Let's stick to random + origin.
        
        # Sort starts by heuristic distance? No, just evaluate.
        # We need to be careful with budget.
        # 500 calls total.
        # Nelder-Mead uses ~5-10 calls per iteration.
        # If we do 5 starts, we have ~100 calls each.
        # This is a good balance.
        
        n_starts = 5
        # Pick 5 best from a larger pool? 
        # Let's just evaluate 5 random points first to pick the best start for NM?
        # Or run NM from each?
        # Running NM from each is expensive. 
        # Strategy: Evaluate 10 points, pick best 2, run NM from them.
        
        pool = []
        for x0 in starts[:10]: # We have 1 + 8 = 9 points. Add 1 more?
             pass
            
        # Let's generate 10 candidate starts
        candidate_starts = [np.zeros(self.dim)]
        for _ in range(9):
            candidate_starts.append(rng.uniform(-4, 4, self.dim))
            
        # Evaluate all candidates
        f_candidates = []
        for x0 in candidate_starts:
            if not time_ok() or calls >= self.budget - 10:
                break
            fx = func(x0)
            calls += 1
            f_candidates.append((fx, x0))
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x0.copy()
                
        if not f_candidates:
            return best_x_global, best_f_global

        # Sort by fitness
        f_candidates.sort(key=lambda x: x[0])
        
        # Run Nelder-Mead from the top 2 candidates
        # Remaining budget: 500 - calls
        # Allocate roughly half to each, or just let them run until budget/time out.
        # Since NM stops on convergence or budget, we just call it.
        
        for i in range(min(2, len(f_candidates))):
            if not time_ok() or calls >= self.budget - 5:
                break
            fx_init, x_init = f_candidates[i]
            nelder_mead(x_init, fx_init)

        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
