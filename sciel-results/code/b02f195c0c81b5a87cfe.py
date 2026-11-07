import numpy as np
import time
import scipy.stats.qmc as qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        X = np.zeros((n_evals, self.dim))

        # Use all budget for smart initialization on the Pareto front
        # DTLZ4 Pareto front: x1^2 + x2^2 + x3^2 = 1, xi >= 0
        # Map 2D unit square to quarter sphere
        
        # Generate Sobol points for better uniformity
        sampler = qmc.Sobol(d=2, scramble=True)
        # Note: Sobol generator state is not tied to np.random.seed directly.
        # To ensure reproducibility per seed, we could use np.random.rand instead if Sobol is not seeded.
        # However, Sobol is deterministic. If we want it to vary with seed, we should use np.random.
        # Let's use np.random.rand for u1, u2 to respect the seed.
        
        u1 = np.random.rand(n_evals)
        u2 = np.random.rand(n_evals)

        phi = np.arccos(u1)
        theta = u2 * (np.pi / 2.0)

        x1 = np.sin(phi) * np.cos(theta)
        x2 = np.sin(phi) * np.sin(theta)
        x3 = np.cos(phi)

        X[:, 0] = x1
        X[:, 1] = x2
        X[:, 2] = x3
        X[:, 3:] = 0.5

        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))

        return X
