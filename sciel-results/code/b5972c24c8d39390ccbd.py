"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential evolution with coordinate descent polish for attractive sector."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin within 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_counter = [0]

        def check_time():
            return time.time() < deadline

        def check_budget():
            return calls_counter[0] < budget

        def eval_point(x):
            """Evaluate a point, updating best if improved."""
            nonlocal best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_counter[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def differential_evolution(pop_size, max_generations=None):
            """Run differential evolution with population size pop_size."""
            nonlocal best_x, best_f
            if not check_time() or not check_budget():
                return

            # Initialize population
            population = np.zeros((pop_size, dim))
            values = np.zeros(pop_size)

            # Diverse initial population
            population[0] = np.zeros(dim)  # Center
            for i in range(1, pop_size):
                population[i] = rng.uniform(-4, 4, dim)
                population[i] = np.clip(population[i], -5, 5)

            for i in range(pop_size):
                if not check_time() or not check_budget():
                    return
                values[i] = eval_point(population[i])

            # DE parameters
            F = 0.8  # Mutation factor
            CR = 0.7  # Crossover rate

            if max_generations is None:
                max_generations = 200

            for gen in range(max_generations):
                if not check_time() or not check_budget():
                    break

                # Sort by fitness
                order = np.argsort(values)
                population = population[order]
                values = values[order]

                # Check convergence
                if values[0] > 0:
                    spread = values[-1] - values[0]
                else:
                    spread = values[-1] - values[0]
                if spread < 1e-10 * (1 + abs(values[0])):
                    break

                for i in range(pop_size):
                    if not check_time() or not check_budget():
                        return

                    # Select three distinct individuals different from i
                    candidates = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(candidates, size=3, replace=False)

                    # Mutation: v = x_a + F * (x_b - x_c)
                    v = population[a] + F * (population[b] - population[c])
                    v = np.clip(v, -5, 5)

                    # Binomial crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]

                    trial = np.clip(trial, -5, 5)
                    f_trial = eval_point(trial)

                    # Selection
                    if f_trial <= values[i]:
                        population[i] = trial
                        values[i] = f_trial

        def coordinate_descent_polish(x_start, max_iterations=50):
            """Coordinate descent with adaptive step sizes for final polish."""
            nonlocal best_x, best_f
            if not check_time() or not check_budget():
                return

            x = x_start.copy()
            f = eval_point(x)

            # Adaptive step sizes per dimension
            step_sizes = np.full(dim, 0.5)

            for _ in range(max_iterations):
                if not check_time() or not check_budget():
                    break

                improved = False
                for i in range(dim):
                    if not check_time() or not check_budget():
                        break

                    # Try both directions
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step_sizes[i]
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = eval_point(x_trial)

                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            step_sizes[i] *= 1.3  # Increase step size on success
                            break
                    else:
                        # No improvement in either direction, decrease step size
                        step_sizes[i] *= 0.7

                if not improved:
                    break

            # Final fine-tuning with smaller steps
            if check_time() and check_budget():
                for _ in range(20):
                    if not check_time() or not check_budget():
                        break
                    improved = False
                    for i in range(dim):
                        if not check_time() or not check_budget():
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * 0.05
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = eval_point(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                improved = True
                                break
                    if not improved:
                        break

        # Phase 1: Differential evolution for global exploration
        # Use ~60% of budget for DE
        de_budget = int(budget * 0.6)
        pop_size = 8  # Small population as per directive
        max_gen = max(1, (de_budget - pop_size) // pop_size)

        differential_evolution(pop_size, max_generations=max_gen)

        # Phase 2: Coordinate descent polish from best found
        if check_time() and check_budget() and best_x is not None:
            remaining = budget - calls_counter[0]
            if remaining >= 10:
                coordinate_descent_polish(best_x, max_iterations=min(50, remaining // 10))

        return best_x, best_f
# EVOLVE-BLOCK-END
