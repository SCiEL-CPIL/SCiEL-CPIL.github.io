"""PSO + local polish optimizer for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Particle swarm optimization with a final local-polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin vs 330s wall-clock kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        LO, HI = -5.0, 5.0
        LB = np.full(dim, LO)
        UB = np.full(dim, HI)

        # ----- Population setup (PSO) -----
        n_particles = 16
        # broad initial spread to get early global coverage
        X = rng.uniform(LO, HI, size=(n_particles, dim))
        # velocities start small-ish; scaled by box size
        v_scale = 0.6 * (HI - LO)
        V = rng.uniform(-v_scale, v_scale, size=(n_particles, dim))

        # evaluate initial population
        F = np.empty(n_particles)
        calls = 0
        for i in range(n_particles):
            F[i] = func(X[i])
            calls += 1

        idx = int(np.argmin(F))
        gbest = X[idx].copy()
        gbest_f = F[idx]
        pbest = X.copy()
        pbest_f = F.copy()

        # PSO hyperparameters
        w = 0.9      # inertia
        c1 = 1.4     # cognitive
        c2 = 1.4     # social
        w_min = 0.4
        w_max = 0.9
        v_max = 1.2 * (HI - LO)  # clamp velocity magnitude per component

        # budget split: PSO phase then local polish
        n_pso = int(budget * 0.75)

        it = 0
        while calls < n_pso:
            if time.time() > deadline - 5.0:
                break
            it += 1
            # linearly decay inertia
            frac = min(1.0, it / max(1, n_pso))
            w = w_max - (w_max - w_min) * frac

            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))
            V = (w * V
                 + c1 * r1 * (pbest - X)
                 + c2 * r2 * (gbest - X))
            # clamp velocity
            V = np.clip(V, -v_max, v_max)

            X = X + V
            X = np.clip(X, LB, UB)

            for i in range(n_particles):
                if calls >= n_pso:
                    break
                fi = func(X[i])
                calls += 1
                if fi < pbest_f[i]:
                    pbest_f[i] = fi
                    pbest[i] = X[i].copy()
                    if fi < gbest_f:
                        gbest_f = fi
                        gbest = X[i].copy()

        # ----- Local polish phase (use remaining budget) -----
        # refine around gbest with a shrinking step
        x = gbest.copy()
        fx = gbest_f
        # estimate a local scale from recent progress
        sigma = 0.3 * (HI - LO)
        for _ in range(budget - calls):
            if time.time() > deadline - 2.0:
                break
            if calls >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), LB, UB)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.85
            # tighten around gbest as we converge
            if fx < gbest_f:
                gbest_f = fx
                gbest = x.copy()

        return gbest, gbest_f
# EVOLVE-BLOCK-END
