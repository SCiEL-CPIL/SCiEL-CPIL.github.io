"""Optimized black-box continuous optimizer using Differential Evolution."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Population-based optimizer using Differential Evolution (DE/rand/1)
    with adaptive mutation factor and a final Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Differential Evolution ---
        # Population size: Small to allow more generations
        pop_size = 10
        # Reserve some budget for polish
        polish_budget = max(20, budget // 5)
        de_budget = budget - polish_budget
        
        # Initialize population using Latin Hypercube Sampling for better coverage
        # LHS ensures uniform coverage in each dimension
        def lhs_sample(n, d, rng):
            # Divide each dimension into n equal intervals
            # Pick one random point in each interval
            points = np.zeros((n, d))
            for i in range(d):
                # Permutation of indices
                perm = rng.permutation(n)
                # Uniform random within each bin
                u = rng.uniform(size=n)
                points[:, i] = (perm + u) / n
            # Scale to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        population = lhs_sample(pop_size, dim, rng)
        fitness = np.array([evaluate(x) for x in population])
        
        # Track best
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # Adaptive F: Start high, decay
        F_start = 0.8
        F_end = 0.3
        CR = 0.5  # Crossover rate

        # DE Loop
        gen = 0
        max_generations = de_budget // pop_size
        stagnation_count = 0
        prev_best = np.inf

        while calls_made < de_budget and time.time() < deadline:
            gen += 1
            # Linear decay of F
            if max_generations > 0:
                progress = min(1.0, gen / max_generations)
                F = F_start + (F_end - F_start) * progress
            else:
                F = F_start

            # Sort population by fitness
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]

            # Update global best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = population[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # If stuck, maybe perturb best or restart part of pop?
            # For now, let DE handle it. If stuck for many gens, reduce CR?
            if stagnation_count > 5:
                CR = 0.2  # More exploitation
            else:
                CR = 0.5

            for i in range(pop_size):
                if calls_made >= de_budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                # Avoid i and the best (index 0) if possible, but standard DE allows any
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                # Ensure they are distinct from i? Standard DE/rand/1 usually picks distinct from each other, 
                # but can include i. Let's just pick 3 distinct indices.
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)

                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5.0, 5.0)

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                trial = np.clip(trial, -5.0, 5.0)

                f_trial = evaluate(trial)

                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

        # --- Phase 2: Nelder-Mead Polish ---
        # Use remaining budget for local refinement of the best point
        if calls_made < budget and best_x is not None:
            # We need to wrap func to count calls
            polish_calls = 0
            def polish_func(x):
                nonlocal polish_calls
                if polish_calls >= (budget - calls_made):
                    return np.inf
                x = np.clip(x, -5.0, 5.0)
                fx = float(func(x))
                polish_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # Nelder-Mead can be expensive. Limit iterations.
            # maxiter in scipy minimizes number of function evals roughly.
            # We have ~60 calls left.
            try:
                res = minimize(polish_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                # Update best if result is better (it should be tracked in polish_func)
            except Exception:
                pass

        return best_x, best_f
