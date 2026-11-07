"""Adaptive population-based optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """
    Population-based optimizer with adaptive step size and diversity maintenance.
    Designed for the BBOB f20 (Schwefel) function on [-5, 5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Diverse initial population
        # Start with a set of diverse points covering the search space
        pop_size = 20
        population = []

        # Origin
        population.append(np.zeros(dim))

        # Vertices of [-4, 4]^5 (subset to keep budget reasonable)
        # Use a space-filling approach instead of all 32 vertices
        for _ in range(pop_size - 1):
            p = rng.uniform(-4.0, 4.0, dim)
            population.append(p)

        # Evaluate initial population
        for p in population:
            if time.time() > deadline or calls_made >= self.budget:
                break
            fx = safe_call(np.clip(p, -5.0, 5.0))

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Phase 2: Adaptive population-based search
        # Maintain a population and evolve it with adaptive step sizes
        # Each individual has its own step size that adapts based on success rate

        # Create individuals: (x, f, sigma)
        individuals = []
        for p in population:
            x_clipped = np.clip(p, -5.0, 5.0)
            # Re-evaluate to get consistent values (or use cached if available)
            # For simplicity, we'll re-evaluate in the loop
            individuals.append({'x': x_clipped, 'f': np.inf, 'sigma': 1.0, 'success': 0, 'attempts': 0})

        # Evaluate all individuals
        for ind in individuals:
            if time.time() > deadline or calls_made >= self.budget:
                break
            ind['f'] = safe_call(ind['x'])

        # Sort by fitness
        individuals.sort(key=lambda x: x['f'])

        # Main search loop
        stagnation_count = 0
        prev_best = best_f

        while calls_made < self.budget and time.time() < deadline:
            # Check for stagnation
            if best_f < prev_best - 1e-8:
                stagnation_count = 0
            else:
                stagnation_count += 1
            prev_best = best_f

            # If stagnating, increase exploration
            if stagnation_count > 50:
                # Inject diversity: replace worst half with random points
                half = len(individuals) // 2
                for i in range(half, len(individuals)):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    individuals[i]['x'] = rng.uniform(-5.0, 5.0, dim)
                    individuals[i]['f'] = safe_call(individuals[i]['x'])
                    individuals[i]['sigma'] = 2.0
                    individuals[i]['success'] = 0
                    individuals[i]['attempts'] = 0
                individuals.sort(key=lambda x: x['f'])
                stagnation_count = 0
                continue

            # For each individual, generate an offspring
            for i, ind in enumerate(individuals):
                if time.time() > deadline or calls_made >= self.budget:
                    break

                # Adaptive sigma: increase if failing, decrease if succeeding
                if ind['attempts'] > 5:
                    success_rate = ind['success'] / ind['attempts']
                    if success_rate < 0.3:
                        ind['sigma'] *= 1.5
                    elif success_rate > 0.7:
                        ind['sigma'] *= 0.7
                    ind['sigma'] = np.clip(ind['sigma'], 0.01, 5.0)

                # Generate offspring: mutation + occasional crossover with best
                if rng.random() < 0.3 and i > 0:
                    # Crossover with a random better individual
                    parent_idx = rng.integers(0, i)
                    parent = individuals[parent_idx]['x']
                    alpha = rng.uniform(0.2, 0.8)
                    child_x = alpha * ind['x'] + (1 - alpha) * parent
                    child_x += ind['sigma'] * rng.standard_normal(dim)
                else:
                    # Pure mutation
                    child_x = ind['x'] + ind['sigma'] * rng.standard_normal(dim)

                child_x = np.clip(child_x, -5.0, 5.0)
                child_f = safe_call(child_x)

                # Update individual stats
                ind['attempts'] += 1
                if child_f < ind['f']:
                    ind['success'] += 1

                # Replace individual if child is better
                if child_f < ind['f']:
                    ind['x'] = child_x
                    ind['f'] = child_f
                else:
                    # Keep the better of parent and child, but allow exploration
                    # with some probability, keep the child to maintain diversity
                    if rng.random() < 0.2:
                        ind['x'] = child_x
                        ind['f'] = child_f

            # Sort population
            individuals.sort(key=lambda x: x['f'])

            # Update best
            if individuals[0]['f'] < best_f:
                best_f = individuals[0]['f']
                best_x = individuals[0]['x'].copy()

        # Phase 3: Final local refinement around the best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            # Use a simple hill-climbing with decreasing step size
            sigma = 0.5
            local_best_x = best_x.copy()
            local_best_f = best_f

            while calls_made < self.budget and time.time() < deadline:
                x_pert = np.clip(local_best_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fx = safe_call(x_pert)
                if fx < local_best_f:
                    local_best_x = x_pert
                    local_best_f = fx
                    sigma *= 0.8
                    if sigma < 0.01:
                        break
                else:
                    sigma *= 1.3
                    if sigma > 2.0:
                        sigma = 2.0

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x

        return best_x, best_f
# EVOLVE-BLOCK-END
