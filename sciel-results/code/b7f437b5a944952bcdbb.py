"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using grid-based Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _sobol_like(self, n, dim):
        """Generate quasi-random points using Halton sequence."""
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
                 233, 239, 241, 251, 257, 263, 269, 271, 277, 281]
        points = np.zeros((n, dim))
        for d in range(dim):
            base = bases[d % len(bases)]
            for i in range(n):
                points[i, d] = self._halton(i + 1, base)
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use Halton sequence for initial population
        init_size = min(100, budget // 2)
        X_init = self._sobol_like(init_size, dim)
        X_init = lo + (hi - lo) * X_init

        # Evaluate initial population
        F = np.array([problem(x) for x in X_init])
        used = init_size

        # Store all evaluated points
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F]

        # Check deadline
        if time.time() > deadline - 30:
            return np.array(all_X)

        # Main optimization loop
        while used < budget:
            if time.time() > deadline - 30:
                break

            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            non_dominated_idx = np.where(~dominated)[0]

            if len(non_dominated_idx) == 0:
                parents = np.random.choice(len(F), size=min(10, len(F)), replace=False)
            else:
                parents = non_dominated_idx[:min(10, len(non_dominated_idx))]

            # Generate children via mutation and crossover
            n_children = min(10, budget - used)
            for _ in range(n_children):
                if time.time() > deadline - 30:
                    break

                # Select two parents
                p1_idx = parents[np.random.randint(len(parents))]
                p2_idx = parents[np.random.randint(len(parents))]
                p1 = X_init[p1_idx]
                p2 = X_init[p2_idx]

                # Simulated binary crossover (SBX)
                child = np.zeros(dim)
                for d in range(dim):
                    if np.random.rand() < 0.5:
                        # SBX crossover
                        if abs(p1[d] - p2[d]) < 1e-10:
                            child[d] = p1[d]
                        else:
                            beta = (p1[d] - p2[d]) / (hi[d] - lo[d])
                            if beta > 0:
                                alpha = 2
                                beta_q = (1 + (alpha + 1) * beta) ** (-1 / (alpha + 1))
                                beta_p = (1 + (alpha + 1) * (1 - beta)) ** (-1 / (alpha + 1))
                                child[d] = (beta_q * p1[d] + beta_p * p2[d]) / (beta_q + beta_p)
                            else:
                                alpha = 2
                                beta = (p2[d] - p1[d]) / (hi[d] - lo[d])
                                beta_q = (1 + (alpha + 1) * beta) ** (-1 / (alpha + 1))
                                beta_p = (1 + (alpha + 1) * (1 - beta)) ** (-1 / (alpha + 1))
                                child[d] = (beta_q * p2[d] + beta_p * p1[d]) / (beta_q + beta_p)
                    else:
                        # Simple mutation
                        if np.random.rand() < 0.5:
                            child[d] = p1[d] + 0.1 * (hi[d] - lo[d]) * np.random.randn()
                        else:
                            child[d] = p2[d] + 0.1 * (hi[d] - lo[d]) * np.random.randn()

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1

                # Add to archive
                all_X.append(child.copy())
                all_F.append(fc.copy())

                # Replace worst point in population
                worst_idx = np.argmax(np.sum(F, axis=1))
                X_init[worst_idx] = child
                F[worst_idx] = fc

        return np.array(all_X)
# EVOLVE-BLOCK-END
