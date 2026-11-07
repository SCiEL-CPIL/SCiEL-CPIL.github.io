"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio-based continuous optimizer with time guard and multiple strategies."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        def timed_call(x):
            return func(np.asarray(x, dtype=float))

        best_x = None
        best_f = float('inf')
        calls = 0

        def update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        def time_left():
            return deadline - time.time()

        # --- Phase 0: cheap initial sampling for early progress ---
        # A few random points to get a reasonable starting value quickly.
        for _ in range(8):
            if calls >= budget or time_left() <= 5.0:
                break
            x = rng.uniform(-5, 5, dim)
            f = timed_call(x)
            calls += 1
            update(x, f)

        # --- Phase 1: (1+1)-ES with 1/5 rule on the current best ---
        def run_11es(x0, fx, n_steps):
            nonlocal calls
            x = np.array(x0, dtype=float)
            f = fx
            sigma = 1.0
            for _ in range(n_steps):
                if calls >= budget or time_left() <= 5.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = timed_call(y)
                calls += 1
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.2
                    update(x, f)
                else:
                    sigma *= 0.9
            return x, f

        # --- Phase 2: (1+lambda)-ES ---
        def run_1lambdees(x0, fx, n_steps, lam=4):
            nonlocal calls
            x = np.array(x0, dtype=float)
            f = fx
            sigma = 0.8
            for _ in range(n_steps):
                if calls >= budget or time_left() <= 5.0:
                    break
                # Generate lambda offspring in one batch
                pops = rng.standard_normal((lam, dim)) * sigma
                offspring = np.clip(x[None, :] + pops, -5, 5)
                # Evaluate
                for y in offspring:
                    if calls >= budget or time_left() <= 5.0:
                        break
                    fy = timed_call(y)
                    calls += 1
                    if fy < f:
                        x, f = y, fy
                        update(x, f)
                # Adapt sigma based on how many improved
                # simple: if best improved, increase, else decrease
                # We'll just use a fixed schedule-ish: small increase on improvement
                # Recompute: track if any improved this round
                # (approximate)
                # Use a simple rule: sigma *= 1.1 if improved else 0.9
                # We don't track per-round improvement easily; use a heuristic:
                # if best_f decreased since start of round -> increase
                # For simplicity, alternate gently
                sigma *= 0.95
            return x, f

        # --- Phase 3: CMA-ES-like (simplified) ---
        def run_cma(x0, fx, n_steps):
            nonlocal calls
            x = np.array(x0, dtype=float)
            f = fx
            sigma = 0.5
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = 1 / (dim + 3)
            c1 = 2 / ((dim + 1.3) ** 2 + 1)
            cmu = min(0.4 - c1, 2 / ((dim + 2) ** 2 - 1))
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            lam = 4
            for _ in range(n_steps):
                if calls >= budget or time_left() <= 5.0:
                    break
                # Sample lambda points
                B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                z = rng.standard_normal((lam, dim))
                offspring = x[None, :] + sigma * (z @ B.T)
                offspring = np.clip(offspring, -5, 5)
                # Evaluate
                fs = []
                for y in offspring:
                    if calls >= budget or time_left() <= 5.0:
                        break
                    fy = timed_call(y)
                    calls += 1
                    fs.append(fy)
                    if fy < f:
                        x, f = y, fy
                        update(x, f)
                if len(fs) < lam:
                    break
                fs = np.array(fs)
                order = np.argsort(fs)
                # Update mean
                mu_eff = lam // 2
                weights = np.zeros(lam)
                w = np.log(mu_eff + 0.5) - np.log(1, )  # placeholder
                # Simpler: use top half weights
                for i in range(mu_eff):
                    weights[order[i]] = 1.0 / mu_eff
                mu = np.sum(weights)
                new_x = np.sum(offspring * (weights / mu)[:, None], axis=0)
                # Update C
                y_vec = (new_x - x) / sigma
                pc = (1 - cs) * pc + cs * y_vec
                hs = np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** 2) / chi
                c_sigma = c1 + 2 * (mu_eff / dim - 1) * c1
                C = (1 - c_sigma * hs ** 2 / dim - c1 + cmu) * C \
                    + c1 * np.outer(pc, pc) \
                    + cmu * np.sum(weights[order[:mu_eff]] *
                                   np.outer(y_vec, y_vec) * 0, axis=0)  # simplified
                # Simplified: skip full rank-mu update to keep speed
                C = (1 - c1) * C + c1 * np.outer(pc, pc)
                # Regularize
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)
                x = new_x
                # Adapt sigma
                sigma *= np.exp(cs / (dim + 3) * (hs / chi - 1))
            return x, f

        # --- Allocate budget across phases with restarts ---
        # Use the best found so far as starting point.
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float('inf')

        # Phase 1: (1+1)-ES
        x1, f1 = run_11es(best_x, best_f, min(150, budget - calls))
        update(x1, f1)

        # Phase 2: (1+lambda)-ES from current best
        if time_left() > 10.0 and calls < budget:
            x2, f2 = run_1lambdees(best_x, best_f, min(100, budget - calls))
            update(x2, f2)

        # Phase 3: CMA-ES-like from current best
        if time_left() > 10.0 and calls < budget:
            x3, f3 = run_cma(best_x, best_f, min(100, budget - calls))
            update(x3, f3)

        # Final polish: (1+1)-ES with small sigma
        if time_left() > 5.0 and calls < budget:
            x4, f4 = run_11es(best_x, best_f, min(50, budget - calls))
            update(x4, f4)

        return best_x, best_f
# EVOLVE-BLOCK-END
