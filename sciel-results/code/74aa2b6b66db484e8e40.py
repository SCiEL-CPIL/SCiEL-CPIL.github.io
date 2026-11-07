"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: population exploration + (1+1)-ES exploitation + Rastrigin-aware refinement."""

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

        # Phase 1: Population-based exploration to find good basins
        # Use a small population of 8, evolve for 5 generations
        pop_size = 8
        sigma = 1.5

        # Initialize population: center + random points
        population = [np.zeros(dim)]
        for _ in range(pop_size - 1):
            population.append(rng.uniform(-4, 4, dim))

        # Evaluate initial population
        pop_f = []
        for x in population:
            if calls >= budget or time.time() > deadline:
                break
            pop_f.append(eval_func(x))

        # Evolve population for a few generations
        for gen in range(5):
            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Keep top half, generate new half
            half = pop_size // 2
            new_population = population[:half].copy()

            for _ in range(pop_size - half):
                if calls >= budget or time.time() > deadline:
                    break
                parent_idx = rng.integers(0, half)
                parent = population[parent_idx]
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                new_population.append(child)

            new_f = []
            for x in new_population[half:]:
                if calls >= budget or time.time() > deadline:
                    break
                new_f.append(eval_func(x))

            population = new_population
            pop_f = pop_f[:half] + new_f

            # Adapt sigma
            if gen > 0 and gen % 2 == 0:
                recent_best = min(pop_f)
                if recent_best < best_f * 0.99 or (best_f > 0 and recent_best < best_f - 0.1):
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            sigma = np.clip(sigma, 0.1, 4.0)

        # Phase 2: (1+1)-ES exploitation from the best point found
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_es = 1.0  # start with moderate step for (1+1)-ES

            # Success rate tracking
            window = 10
            recent = [False] * window
            idx = 0

            # Use ~150 calls for (1+1)-ES
            max_es_calls = min(150, budget - calls)
            for _ in range(max_es_calls):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1

                # Adapt sigma based on success rate
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma_es *= 1.15
                    elif sr < 0.2:
                        sigma_es *= 0.85
                    else:
                        sigma_es *= 1.0

                sigma_es = np.clip(sigma_es, 0.05, 3.0)

        # Phase 3: Rastrigin-aware refinement
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
