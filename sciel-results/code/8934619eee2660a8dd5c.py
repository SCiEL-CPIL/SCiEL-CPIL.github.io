"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + (1+1)-ES exploitation with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before the 330s kill

        # ---------- Phase 1: Latin Hypercube exploration ----------
        n_lhs = 60
        rng = np.random.default_rng(12345)
        perm = [rng.permutation(n_lhs) for _ in range(dim)]
        lxs = np.array([(perm[d] + rng.random()) / n_lhs for d in range(dim)]).T
        lxs = lxs * 10.0 - 5.0  # map to [-5,5]
        lxs = np.clip(lxs, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            fx = func(lxs[i])
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = lxs[i].copy()

        # ---------- Phase 2: (1+1)-ES with periodic restarts ----------
        sigma = 1.2
        sigma_min = 1e-4
        sigma_max = 5.0
        x = best_x.copy()
        fx = best_f

        global_best_x = best_x.copy()
        global_best_f = best_f

        no_improve = 0
        step = 0
        restart_interval = 100

        while calls < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            calls += 1
            step += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.25, sigma_max)
                no_improve = 0
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
            else:
                sigma = max(sigma * 0.85, sigma_min)
                no_improve += 1

            # Periodic restart: escape local minima
            if step % restart_interval == 0 or no_improve >= 40:
                if time.time() > deadline:
                    break
                if rng.random() < 0.7:
                    x = global_best_x + sigma * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                else:
                    x = rng.uniform(-5.0, 5.0, dim)
                fx = func(x)
                calls += 1
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
                sigma = 1.2
                no_improve = 0

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
