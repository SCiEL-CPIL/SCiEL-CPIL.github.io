import numpy as np


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 20  # Population size
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover rate

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        NP = self.NP
        F = self.F
        CR = self.CR
        budget = self.budget

        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        fvals = np.array([func(x) for x in pop])
        budget -= NP

        best_idx = np.argmin(fvals)
        x_best = pop[best_idx].copy()
        f_best = fvals[best_idx]

        if budget <= 0:
            return x_best, f_best

        # Main loop
        while budget > 0:
            for i in range(NP):
                if budget <= 0:
                    break
                
                # Select three distinct random individuals r1, r2, r3 != i
                # We need 3 distinct indices from [0, NP) excluding i
                # Simple way: sample 3 distinct from range(NP)
                # But we must ensure they are not i.
                # Since NP=20, we can just sample and check.
                # Or use permutation.
                # For efficiency, let's just sample.
                
                # Get 3 distinct indices from 0..NP-1
                # If NP is small, this is tricky. NP=20 is fine.
                indices = list(range(NP))
                indices.remove(i)
                r1, r2, r3 = rng.choice(indices, size=3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                u = pop[i].copy()
                # Choose random position for crossover
                j_rand = rng.integers(0, dim)
                
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                budget -= 1
                
                # Selection
                if f_u < fvals[i]:
                    pop[i] = u
                    fvals[i] = f_u
                    if f_u < f_best:
                        f_best = f_u
                        x_best = u.copy()

        return x_best, f_best
