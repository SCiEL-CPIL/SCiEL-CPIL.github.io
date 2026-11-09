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
        pbest_idx = np.arange(NP)
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
            # Update pbest: track best individual per position
            for i in range(NP):
                if F_vals[i] < pbest_f[i]:
                    pbest_idx[i] = i
                    pbest_f[i] = F_vals[i]

            # Generate trial vectors
            for i in range(NP):
                if not time_ok() or calls_made >= budget:
                    break
                # JADE mutation
                r = rng.random()
                if r < 0.3:
                    # "current-to-pbest/1/bin" - good for exploitation
                    F_i = rng.laplace(0.5, 0.5)
                    F_i = np.clip(F_i, 0.0, 1.0)
                    CR_i = rng.laplace(0.5, 0.3)
                    CR_i = np.clip(CR_i, 0.0, 1.0)
                    j_rand = rng.integers(dim)
                    pb = pbest_idx[i]
                    v = X[i] + F_i * (X[pb] - X[rng.integers(NP)])
                    v = np.where(rng.random(dim) <= CR_i, v, X[i])
                    v[j_rand] = X[i][j_rand]
                elif r < 0.6:
                    # "best/1/bin" - good for exploitation
                    F_i = rng.laplace(F1, 1.0)
                    F_i = np.clip(F_i, 0.0, 1.0)
                    CR_i = rng.laplace(CR1, 1.0)
                    CR_i = np.clip(CR_i, 0.0, 1.0)
                    j_rand = rng.integers(dim)
                    v = X[0] + F_i * (X[rng.integers(NP)] - X[rng.integers(NP)])
                    v = np.where(rng.random(dim) <= CR_i, v, X[i])
                    v[j_rand] = X[i][j_rand]
                else:
                    # "current-to-rand/1/bin" - good for exploration
                    F_i = rng.laplace(0.5, 0.5)
                    F_i = np.clip(F_i, 0.0, 1.0)
                    CR_i = rng.laplace(0.5, 0.3)
                    CR_i = np.clip(CR_i, 0.0, 1.0)
                    j_rand = rng.integers(dim)
                    r1 = rng.integers(NP)
                    r2 = rng.integers(NP)
                    v = X[i] + F_i * (X[r1] - X[r2])
                    v = np.where(rng.random(dim) <= CR_i, v, X[i])
                    v[j_rand] = X[i][j_rand]

                v = np.clip(v, -5, 5)
                calls_made += 1
                new_f = func(v)

                if new_f < F_vals[i]:
                    F_vals[i] = new_f
                    X[i] = v

            # Update global best
            idx = np.argmin(F_vals)
            if F_vals[idx] < best_f:
                best_f = F_vals[idx]
                best_x = X[idx].copy()

            # Update JADE parameters with adaptation
            F = 0.9 * F + 0.1 * 0.5
            CR = 0.9 * CR + 0.1 * 0.5

        # Local refinement using remaining budget
        if calls_made < budget and time_ok():
            # Use Nelder-Mead style local search around best
            step = 0.1
            for _ in range(100):
                if not time_ok() or calls_made >= budget:
                    break
                for d in range(dim):
                    if calls_made >= budget or not time_ok():
                        break
                    for direction in [-1, 1]:
                        if calls_made >= budget or not time_ok():
                            break
                        trial = best_x.copy()
                        trial[d] += direction * step
                        trial = np.clip(trial, -5, 5)
                        calls_made += 1
                        trial_f = func(trial)
                        if trial_f < best_f:
                            best_f = trial_f
                            best_x = trial.copy()
                            step = max(step * 0.7, 0.01)
                        else:
                            step = max(step * 0.8, 0.01)

        return best_x, best_f
# EVOLVE-BLOCK-END
