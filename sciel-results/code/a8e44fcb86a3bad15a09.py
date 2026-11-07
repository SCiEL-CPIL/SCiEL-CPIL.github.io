"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution with local refinement for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = float(func(np.asarray(x, dtype=np.float64)))
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = np.asarray(x, dtype=np.float64).copy()
            return fx

        def check_time():
            return calls_made < budget and time.time() < deadline - 5

        # Phase 1: Differential Evolution Global Search
        # Population size N. We want to use most of the budget for DE.
        # Let's reserve 50 calls for local refinement.
        de_budget = budget - 50
        N = 20  # Population size
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5.0, 5.0, (N, dim))
        fit = np.array([safe_func(ind) for ind in pop])
        
        # Keep track of best in population
        best_idx = np.argmin(fit)
        
        # DE Loop
        while check_time() and calls_made < de_budget:
            for i in range(N):
                if calls_made >= de_budget or not check_time():
                    break
                
                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice([j for j in range(N) if j != i], 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: create trial vector u
                j_rand = rng.randint(dim)
                u = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Clip to bounds
                u = np.clip(u, -5.0, 5.0)
                
                # Evaluation
                f_u = safe_func(u)
                
                # Selection
                if f_u <= fit[i]:
                    pop[i] = u
                    fit[i] = f_u
                    if f_u < fit[best_idx]:
                        best_idx = i

        # Phase 2: Local Refinement
        # Use the best point found by DE as the start for local optimization.
        # We might also try a few other good points from the population.
        # Sort population by fitness
        order = np.argsort(fit)
        top_k = min(5, N)
        local_starts = [pop[order[i]].copy() for i in range(top_k)]
        
        # Ensure best_x is in local_starts (it should be, but just in case)
        if best_x is not None:
             local_starts.append(best_x.copy())

        for x0 in local_starts:
            if not check_time() or calls_made >= budget:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            
            # Use scipy's Nelder-Mead
            # We need to be careful about the call budget
            # Nelder-Mead can make many calls. We limit maxiter.
            # With dim=5, a simplex has 6 points.
            # maxiter=50 is usually enough for local convergence if close.
            
            local_calls = [0]
            def local_func(x):
                local_calls[0] += 1
                if calls_made + local_calls[0] > budget:
                    return 1e6
                return safe_func(np.clip(x, -5.0, 5.0))
            
            try:
                # Tighter tolerances for Rastrigin
                result = minimize(local_func, x0, method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 3: Final Polish with Coordinate Descent if time permits
        if best_x is not None and check_time():
            x = best_x.copy()
            fx = best_f
            # Very fine steps
            for scale in [0.01, 0.001]:
                if not check_time() or calls_made >= budget:
                    break
                step = np.ones(dim) * scale
                for _ in range(2):
                    if not check_time() or calls_made >= budget:
                        break
                    improved = False
                    for d in range(dim):
                        if not check_time() or calls_made >= budget:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[d] = np.clip(x_trial[d] + direction * step[d], -5.0, 5.0)
                            if not np.allclose(x_trial, x, atol=1e-8):
                                f_trial = safe_func(x_trial)
                                if f_trial < fx:
                                    x = x_trial
                                    fx = f_trial
                                    improved = True
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                    break
                    if not improved:
                        break

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
