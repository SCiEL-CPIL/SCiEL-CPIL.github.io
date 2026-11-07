"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive Differential Evolution with Powell Local Refinement for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # Safety margin before 300s limit
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
            try:
                fx = float(func(np.asarray(x, dtype=np.float64)))
            except Exception:
                fx = 1e6
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = np.asarray(x, dtype=np.float64).copy()
            return fx

        def check_time():
            return calls_made < budget and time.time() < deadline - 5

        # Phase 1: Adaptive Differential Evolution Global Search
        # Smaller population, more generations, adaptive parameters
        de_budget = int(budget * 0.7)  # 70% for global
        N = 15  # Population size
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5.0, 5.0, (N, dim))
        fit = np.array([safe_func(ind) for ind in pop])
        
        # Keep track of best in population
        best_idx = np.argmin(fit)
        
        # Adaptive parameters history
        success_count = 0
        total_trials = 0
        
        # DE Loop
        while check_time() and calls_made < de_budget:
            # Adaptive update every 10 trials (approx)
            if total_trials > 10:
                success_rate = success_count / total_trials
                # Adjust F and CR
                F = np.clip(F + 0.1 * (success_rate - 0.5), 0.1, 1.0)
                CR = np.clip(CR + 0.1 * (success_rate - 0.5), 0.1, 1.0)
                success_count = 0
                total_trials = 0

            for i in range(N):
                if calls_made >= de_budget or not check_time():
                    break
                
                # Select 3 distinct random indices different from i
                # Use 'best/1/DE' strategy: x_best + F*(x_r1 - x_r2)
                # Occasionally use 'rand/1/DE'
                if rng.random() < 0.5:
                    # Best/1/DE
                    r1, r2 = rng.choice([j for j in range(N) if j != i], 2, replace=False)
                    v = pop[best_idx] + F * (pop[r1] - pop[r2])
                else:
                    # Rand/1/DE
                    r1, r2, r3 = rng.choice([j for j in range(N) if j != i], 3, replace=False)
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
                total_trials += 1
                
                # Selection
                if f_u <= fit[i]:
                    pop[i] = u
                    fit[i] = f_u
                    success_count += 1
                    if f_u < fit[best_idx]:
                        best_idx = i

        # Phase 2: Local Refinement using Powell's Method
        # Powell is often more robust than Nelder-Mead for Rastrigin in moderate dimensions
        # Select top 3 diverse points from the DE population
        order = np.argsort(fit)
        # Ensure diversity: pick best, then next best that is far from best
        local_starts = []
        best_p = pop[order[0]].copy()
        local_starts.append(best_p)
        
        for idx in order[1:]:
            if len(local_starts) >= 3:
                break
            p = pop[idx].copy()
            # Check distance from existing starts
            is_diverse = True
            for s in local_starts:
                if np.linalg.norm(p - s) < 0.5:
                    is_diverse = False
                    break
            if is_diverse:
                local_starts.append(p)
        
        # Add the global best found so far if not already in starts
        if best_x is not None:
            if all(np.linalg.norm(best_x - s) > 0.1 for s in local_starts):
                local_starts.append(best_x.copy())

        local_budget_remaining = budget - calls_made
        if local_budget_remaining <= 0:
            return best_x, best_f

        # Distribute local budget among starts
        calls_per_start = max(10, local_budget_remaining // len(local_starts))

        for x0 in local_starts:
            if not check_time() or calls_made >= budget:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            
            # Use scipy's Powell method
            # Powell is derivative-free and handles non-smooth functions better than BFGS
            # Nelder-Mead is also an option, but Powell is often faster for this shape
            local_calls = [0]
            def local_func(x):
                local_calls[0] += 1
                if calls_made + local_calls[0] > budget:
                    return 1e6
                return safe_func(np.clip(x, -5.0, 5.0))
            
            try:
                # Tighter tolerances
                result = minimize(local_func, x0, method='Powell',
                                 options={'maxiter': calls_per_start, 'xtol': 1e-8, 'ftol': 1e-8, 'disp': False})
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
            for scale in [0.01, 0.001, 0.0001]:
                if not check_time() or calls_made >= budget:
                    break
                step = np.ones(dim) * scale
                for _ in range(3):
                    if not check_time() or calls_made >= budget:
                        break
                    improved = False
                    for d in range(dim):
                        if not check_time() or calls_made >= budget:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[d] = np.clip(x_trial[d] + direction * step[d], -5.0, 5.0)
                            if not np.allclose(x_trial, x, atol=1e-9):
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
