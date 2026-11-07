"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES exploration + Rastrigin-aware coordinate refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: CMA-ES with adaptive population
        # CMA-ES handles the rotated Rastrigin landscape better than (1+1)-ES
        # because it adapts the covariance matrix to the rotation
        
        # Initial population: mix of center and random
        pop_size = 10
        # CMA-ES parameters
        sigma = 2.0  # initial step size
        mean = np.zeros(dim)
        
        # Initialize with diverse starting points
        population = [np.zeros(dim)]
        for _ in range(pop_size - 1):
            population.append(rng.uniform(-4, 4, dim))
        
        # Evaluate initial population
        pop_f = []
        for x in population:
            if calls >= budget or time.time() > deadline:
                break
            pop_f.append(eval_func(x))
        
        # CMA-ES evolution
        # We'll do a simplified CMA-ES-like evolution
        # Since implementing full CMA-ES is complex, we'll use a population-based
        # approach with adaptive sigma based on success rate
        
        n_generations = 8
        for gen in range(n_generations):
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Keep top half
            half = pop_size // 2
            survivors = population[:half]
            
            # Generate new half
            new_population = survivors.copy()
            for _ in range(pop_size - half):
                if calls >= budget or time.time() > deadline:
                    break
                # Pick a random survivor as parent
                parent_idx = rng.integers(0, half)
                parent = population[parent_idx]
                # Add noise
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                new_population.append(child)
            
            # Evaluate new individuals
            new_f = []
            for x in new_population[half:]:
                if calls >= budget or time.time() > deadline:
                    break
                new_f.append(eval_func(x))
            
            population = new_population
            pop_f = pop_f[:half] + new_f
            
            # Adapt sigma based on improvement
            if gen > 0:
                prev_best = min(pop_f[:half])  # best from previous gen's survivors
                curr_best = min(pop_f)
                if curr_best < prev_best - 0.1:
                    sigma *= 1.1  # good progress, increase step
                elif curr_best > prev_best + 0.1:
                    sigma *= 0.9  # bad progress, decrease step
            sigma = np.clip(sigma, 0.1, 4.0)

        # Phase 2: Coordinate-wise refinement with Rastrigin-aware snapping
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Strategy 1: Coordinate-wise refinement with decreasing step
            step = 0.3
            for _ in range(40):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
            
            # Strategy 2: Try snapping to nearby "integer-like" points
            # Rastrigin has minima at integer coordinates (after rotation/shift)
            if calls < budget and time.time() < deadline:
                candidates = []
                for d in range(dim):
                    base = x_local.copy()
                    for offset in [-1, 0, 1]:
                        y = base.copy()
                        y[d] = round(y[d]) + offset
                        y = np.clip(y, -5, 5)
                        candidates.append(y)
                candidates.append(np.clip(np.round(x_local), -5, 5))
                
                seen = set()
                for y in candidates:
                    if calls >= budget or time.time() > deadline:
                        break
                    key = tuple(np.round(y, 6))
                    if key not in seen:
                        seen.add(key)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
            
            # Strategy 3: Final fine-tuning with very small steps
            if calls < budget and time.time() < deadline:
                if best_f < f_local:
                    x_local = best_x.copy()
                    f_local = best_f
                step = 0.05
                for _ in range(30):
                    if calls >= budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            y = x_local.copy()
                            y[d] += sign * step
                            y = np.clip(y, -5, 5)
                            fy = eval_func(y)
                            if fy < f_local:
                                x_local, f_local = y, fy
                                improved = True
                                break
                    if not improved:
                        step *= 0.5
                        if step < 0.005:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
