"""Adaptive multi-start Nelder-Mead + differential evolution for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class Optimizer:
    """Adaptive multi-start Nelder-Mead + differential evolution for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = None
        best_f = float('inf')
        X_all = []
        y_all = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            eval_count += 1
            f_val = float(func(x))
            X_all.append(x.copy())
            y_all.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)

        # Phase 1: Initial space-filling sampling using Sobol-like sequence
        def sobol_like_sequence(n, dim, rng_seed=42):
            """Generate quasi-random points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    points[i, d] = r
            perm = rng.permutation(n)
            return points[perm]

        n_init = min(25, budget - 10)
        init_points = sobol_like_sequence(n_init, dim, rng_seed=42)
        init_points = init_points * 10.0 - 5.0

        for pt in init_points:
            if eval_count >= budget - 10 or time.time() > deadline:
                break
            wrapped_func(pt)

        # Phase 2: Multi-start Nelder-Mead from top candidates
        # Select diverse starting points based on function values
        if len(X_all) >= 5:
            X_arr = np.array(X_all)
            y_arr = np.array(y_all)
            # Sort by function value, take top diverse points
            sorted_idx = np.argsort(y_arr)
            # Take top 5 diverse points
            start_points = []
            for idx in sorted_idx[:min(10, len(sorted_idx))]:
                pt = X_arr[idx]
                # Check diversity
                if len(start_points) == 0:
                    start_points.append(pt)
                else:
                    min_dist = min(np.linalg.norm(pt - sp) for sp in start_points)
                    if min_dist > 1.0:
                        start_points.append(pt)
                if len(start_points) >= 5:
                    break

            # If not enough diverse points, add random ones
            while len(start_points) < 3:
                start_points.append(rng.uniform(-5.0, 5.0, dim))

            # Run Nelder-Mead from each start point
            for sp in start_points:
                if eval_count >= budget - 20 or time.time() > deadline:
                    break
                try:
                    res = minimize(
                        wrapped_func,
                        sp,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                except Exception:
                    pass

        # Phase 3: Differential Evolution for global search
        if eval_count < budget - 30 and time.time() < deadline:
            # DE parameters
            pop_size = min(20, budget - eval_count - 30)
            pop_size = max(pop_size, 10)
            
            # Initialize population
            population = []
            for i in range(pop_size):
                if i < len(X_all) and i < len(y_all):
                    # Use some existing points
                    pt = X_all[i].copy()
                else:
                    pt = rng.uniform(-5.0, 5.0, dim)
                population.append(pt)
            
            # Evaluate initial population
            for i, pt in enumerate(population):
                if eval_count >= budget - 20 or time.time() > deadline:
                    break
                if i < len(y_all) and np.linalg.norm(pt - X_all[i]) < 1e-6:
                    continue  # Already evaluated
                f_val = wrapped_func(pt)
            
            # DE evolution
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            max_gen = 50
            
            for gen in range(max_gen):
                if eval_count >= budget - 15 or time.time() > deadline:
                    break
                
                # Sort population by fitness
                pop_with_fit = [(pt, wrapped_func(pt) if not any(np.allclose(pt, x) for x in X_all[-10:]) else min(y_all[-10:])) for pt in population]
                pop_with_fit.sort(key=lambda x: x[1])
                population = [pt for pt, _ in pop_with_fit]
                
                # Generate new candidates
                new_population = []
                for i in range(len(population)):
                    if eval_count >= budget - 15 or time.time() > deadline:
                        break
                    
                    # Select three distinct individuals
                    a, b, c = rng.choice(len(population), 3, replace=False)
                    
                    # Mutation: v = x_a + F * (x_b - x_c)
                    v = population[a] + F * (population[b] - population[c])
                    v = np.clip(v, -5.0, 5.0)
                    
                    # Crossover
                    trial = population[i].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    
                    # Evaluate trial
                    trial_f = wrapped_func(trial)
                    
                    # Selection
                    if trial_f <= population[i][0] if isinstance(population[i], tuple) else trial_f <= wrapped_func(population[i]):
                        population[i] = trial
                    else:
                        population[i] = population[i] if isinstance(population[i], np.ndarray) else population[i]
        
        # Phase 4: Local refinement with L-BFGS-B from current best
        if eval_count < budget - 5 and time.time() < deadline and best_x is not None:
            try:
                minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxls': 15, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass

        # Phase 5: Fill remaining budget with evaluations near best
        while eval_count < budget and time.time() < deadline and best_x is not None:
            x = best_x + rng.normal(0, 0.05, dim)
            x = np.clip(x, -5.0, 5.0)
            wrapped_func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
