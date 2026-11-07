"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        self.calls = 0
        budget = self.budget
        dim = self.dim

        def wrapped_func(x):
            self.calls += 1
            if self.calls > budget:
                raise StopIteration()
            return float(func(x))

        # Differential Evolution Parameters
        pop_size = 10 + dim  # 15 for dim=5
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        f_pop = np.array([wrapped_func(x) for x in pop])

        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]

        # Main DE Loop
        while self.calls < budget:
            for i in range(pop_size):
                if self.calls >= budget:
                    break

                # Select 3 distinct random indices different from i
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)

                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover: trial vector
                trial = pop[i].copy()
                for j in range(dim):
                    if np.random.rand() < CR or j == np.random.randint(dim):
                        trial[j] = v[j]

                # Evaluate trial
                f_trial = wrapped_func(trial)

                # Selection
                if f_trial <= f_pop[i]:
                    pop[i] = trial
                    f_pop[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
