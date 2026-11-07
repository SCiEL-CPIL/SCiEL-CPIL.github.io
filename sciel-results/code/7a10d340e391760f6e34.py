"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with small population for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Population-based Differential Evolution
        # lambda = 5 (small population), CR = 0.5, F = 0.8
        lam = 5
        CR = 0.5
        F = 0.8

        # Initialize population: mix of center and random points
        pop = []
        pop_f = []

        # First point at origin
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        pop.append(x0)
        pop_f.append(f0)

        # Add random points
        for i in range(lam - 1):
            xi = rng.uniform(-4.0, 4.0, dim)
            fi = safe_call(xi)
            pop.append(xi)
            pop_f.append(fi)

        # Convert to arrays for easier manipulation
        pop = np.array(pop)  # shape (lam, dim)
        pop_f = np.array(pop_f)  # shape (lam,)

        # Main DE loop
        while call_count < budget and time.time() < deadline - 5.0:
            # Sort population by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            # Generate offspring for each individual
            offspring = np.zeros_like(pop)
            offspring_f = np.full(lam, np.inf)

            for i in range(lam):
                if call_count >= budget or time.time() > deadline - 5.0:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(lam))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover: binomial crossover with rate CR
                trial = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                trial = np.clip(trial, -5.0, 5.0)
                ft = safe_call(trial)
                offspring[i] = trial
                offspring_f[i] = ft

                # Selection: keep better of parent and offspring
                if ft <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = ft

            # Check for stagnation: if no improvement in last lam generations, restart
            # Simple stagnation check: if best_f hasn't improved significantly
            # For now, just continue the loop

            # Adaptive: if population has converged (small spread), add diversity
            spread = np.max(pop, axis=0) - np.min(pop, axis=0)
            if np.mean(spread) < 0.1 and call_count < budget - lam:
                # Inject a new random individual
                idx_worst = np.argmax(pop_f)
                new_x = rng.uniform(-5.0, 5.0, dim)
                new_f = safe_call(new_x)
                if new_f < pop_f[idx_worst]:
                    pop[idx_worst] = new_x
                    pop_f[idx_worst] = new_f

        return best_x, best_f
# EVOLVE-BLOCK-END
