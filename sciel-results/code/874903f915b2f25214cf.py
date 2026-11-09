"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """JADE-style adaptive DE with deadline guard for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Population size
        NP = 20
        # Initialize population
        X = rng.uniform(-5, 5, (NP, dim))
        F_vals = np.array([func(x) for x in X])
        best_idx = np.argmin(F_vals)
        best_x = X[best_idx].copy()
        best_f = F_vals[best_idx]

        # JADE parameters
        F1 = 0.5
        CR1 = 0.1
        F2 = 0.8
        CR2 = 0.9
        F = 0.5
        CR = 0.5

        # pbest: for each individual, the best individual it has ever seen
        pbest = np.arange(NP)
        pbest_f = F_vals.copy()

        calls_made = NP
        gen = 0
        max_gen = 200

        while calls_made < budget and gen < max_gen and time_ok():
            gen += 1
            # Sort by fitness
            order = np.argsort(F_vals)
            X = X[order]
            F_vals = F_vals[order]
            # Update pbest
            for i in range(NP):
                if F_vals[i] < pbest_f[pbest[i]]:
                    pbest[i] = i
                    pbest_f[i] = F_vals[i]

            # Generate trial vectors
            new_X = np.empty_like(X)
            new_F = np.empty(NP)

            for i in range(NP):
                if not time_ok():
                    break
                # JADE mutation
                r = rng.random()
                if r < 0.5:
                    # "best/1/bin"
                    F_i = rng.laplace(F1, 1.0)
                    F_i = np.clip(F_i, 0.0, 1.0)
                    CR_i = rng.laplace(CR1, 1.0)
                    CR_i = np.clip(CR_i, 0.0, 1.0)
                    j_rand = rng.integers(dim)
                    v = X[0] + F_i * (X[rng.integers(NP)] - X[rng.integers(NP)])
                    v = np.where(rng.random(dim) <= CR_i, v, X[i])
                    v[j_rand] = X[i][j_rand]
                else:
                    # "pbest/1/bin"
                    F_i = rng.laplace(F2, 1.0)
                    F_i = np.clip(F_i, 0.0, 1.0)
                    CR_i = rng.laplace(CR2, 1.0)
                    CR_i = np.clip(CR_i, 0.0, 1.0)
                    j_rand = rng.integers(dim)
                    pb = pbest[i]
                    v = X[pb] + F_i * (X[rng.integers(NP)] - X[rng.integers(NP)])
                    v = np.where(rng.random(dim) <= CR_i, v, X[i])
                    v[j_rand] = X[i][j_rand]

                v = np.clip(v, -5, 5)
                new_X[i] = v
                calls_made += 1
                new_F[i] = func(v)

                if new_F[i] < F_vals[i]:
                    F_vals[i] = new_F[i]
                    X[i] = v

            # Update global best
            idx = np.argmin(F_vals)
            if F_vals[idx] < best_f:
                best_f = F_vals[idx]
                best_x = X[idx].copy()

            # Update JADE parameters
            # F = F + (F1 - F) * (some adaptation)
            # Simple adaptation: use exponential moving average
            F = 0.9 * F + 0.1 * 0.5
            CR = 0.9 * CR + 0.1 * 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
