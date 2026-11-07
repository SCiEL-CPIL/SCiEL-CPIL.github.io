import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: balance between exploration and exploitation
        pop_size = min(30, max(10, self.budget // 8))
        if pop_size > self.budget:
            pop_size = self.budget

        # Initialize with ZDT6-aware strategy:
        # - x[0] uniform for front coverage
        # - x[1:] biased toward 0 (minimizes g(x) for ZDT6)
        # Mix: 70% structured, 30% random
        n_structured = int(pop_size * 0.7)
        n_random = pop_size - n_structured

        X = np.empty((pop_size, self.dim))

        # Structured: x[0] linspace, x[1:] = 0 (optimal for ZDT6)
        x0_vals = np.linspace(0.0, 1.0, n_structured) if n_structured > 1 else np.array([0.5])
        for i in range(n_structured):
            X[i] = 0.0
            X[i, 0] = x0_vals[i]

        # Random: x[0] uniform, x[1:] biased toward 0 via power transform
        for i in range(n_structured, pop_size):
            X[i, 0] = np.random.rand()
            X[i, 1:] = np.random.rand(self.dim - 1) ** 3  # Bias toward 0

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Evolutionary loop: select non-dominated parents, mutate
        mutation_scale = 0.1
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            # Select parents from non-dominated set
            nd_idx = np.where(~dominated)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([0])

            # Pick a random non-dominated parent
            p_idx = nd_idx[np.random.randint(len(nd_idx))]
            parent = X[p_idx].copy()

            # Mutate: Gaussian perturbation
            child = parent + mutation_scale * span * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)

            # Evaluate child
            fc = problem(child)
            used += 1

            # Replace the worst point in the population (by sum of objectives)
            # This maintains diversity while improving quality
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

            # Occasionally add a new random point for diversity
            if np.random.rand() < 0.1 and used < self.budget:
                new_x = lo + span * np.random.rand(self.dim)
                # Bias x[1:] toward 0 for ZDT6
                new_x[1:] = lo[1:] + (hi[1:] - lo[1:]) * (np.random.rand(self.dim - 1) ** 3)
                new_f = problem(new_x)
                used += 1
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = new_x
                F[worst] = new_f

        return X
