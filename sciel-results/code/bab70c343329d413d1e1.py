"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES-like optimizer with Rastrigin-aware refinement."""

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

        # Phase 1: Population-based search with adaptive step
        # Use a population of 10, with selection and mutation
        pop_size = 10
        sigma = 1.5  # initial step size

        # Initialize population: mix of center, random, and structured points
        population = []
        # Center
        population.append(np.zeros(dim))
        # Random points
        for _ in range(pop_size - 1):
            population.append(rng.uniform(-4, 4, dim))

        # Evaluate initial population
        pop_f = []
        for x in population:
            if calls >= budget or time.time() > deadline:
                break
            pop_f.append(eval_func(x))

        # Main loop: evolve population
        max_generations = 45  # ~450 calls with pop_size 10
        for gen in range(max_generations):
            if calls >= budget or time.time() > deadline:
                break

            # Sort population by fitness
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Keep top half, generate new half
            half = pop_size // 2
            new_population = population[:half].copy()

            # Generate new individuals from top parents
            for _ in range(pop_size - half):
                if calls >= budget or time.time() > deadline:
                    break
                # Pick a random parent from top half
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

            # Combine and sort
            population = new_population
            pop_f = pop_f[:half] + new_f

            # Adapt sigma based on progress
            # If best improved recently, increase sigma slightly; else decrease
            if gen > 0 and gen % 5 == 0:
                # Check if we're making progress
                recent_best = min(pop_f)
                if recent_best < best_f * 0.99 or (best_f > 0 and recent_best < best_f - 0.1):
                    sigma *= 1.05  # slight increase
                else:
                    sigma *= 0.95  # slight decrease

            sigma = np.clip(sigma, 0.05, 5.0)

        # Phase 2: Rastrigin-aware refinement
        # Rastrigin has local minima at integer coordinates (in its native space)
        # After rotation/shift, we need to find the nearest "grid" points
        # Strategy: for the best point, try snapping each coordinate to nearby integers
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f

            # Try multiple refinement strategies
            # Strategy 1: Coordinate-wise refinement with decreasing step
            step = 0.3
            for _ in range(50):
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
            # For Rastrigin, minima are at integer coordinates in the native space
            # After rotation, these become a rotated grid
            # We can try points near the current best that are close to integers
            if calls < budget and time.time() < deadline:
                # Generate candidate points near best_x
                candidates = []
                # Round each coordinate to nearest integer
                for d in range(dim):
                    base = x_local.copy()
                    for offset in [-1, 0, 1]:
                        y = base.copy()
                        y[d] = round(y[d]) + offset
                        y = np.clip(y, -5, 5)
                        candidates.append(y)
                # Also try the pure rounded point
                candidates.append(np.clip(np.round(x_local), -5, 5))

                # Evaluate unique candidates
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
                x_local = best_x.copy() if best_f < f_local else x_local
                f_local = min(best_f, f_local)
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
