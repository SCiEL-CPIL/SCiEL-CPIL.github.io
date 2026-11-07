"""Parallel Tempering (Replica Exchange) ensemble of CMA-ES instances for f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class _CMAES:
    """Minimal CMA-ES with covariance adaptation."""

    def __init__(self, dim, x0, sigma0, rng, budget_per_replica):
        self.dim = dim
        self.x = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.C = np.eye(dim)
        self.D = np.sqrt(np.diag(self.C))
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.rng = rng
        self.mu = max(1, dim // 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmua = min(1 - self.c1, 2 / ((self.dim + 2) ** 2 + self.mu_eff))
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        self.budget = budget_per_replica
        self.calls = 0

    def step(self, func):
        """Perform one CMA-ES iteration: sample mu offspring, evaluate, update."""
        d = self.dim
        # Sample
        B = np.linalg.inv(self.D)
        # Cholesky of C
        L = np.linalg.cholesky(self.C + 1e-12 * np.eye(d))
        B = np.linalg.inv(L)
        z = self.rng.standard_normal((self.mu, d))
        y = (z @ B.T).T
        offsprings = [self.x + self.sigma * yi for yi in y]
        # Evaluate
        fvals = np.array([func(np.clip(ox, -5, 5)) for ox in offsprings])
        self.calls += self.mu
        # Sort
        order = np.argsort(fvals)
        y_sorted = y[order]
        f_sorted = fvals[order]
        # Update
        m = sum(self.weights[i] * y_sorted[i] for i in range(self.mu))
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.sigma * m)
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.calls + 1)))) < (1.4 + 2 / (d + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * self.cc * m
        delta_hsig = (1 - hsig) * self.cc * (1 - self.cc)
        self.C = (1 + self.c1 * delta_hsig - self.c1 - self.cmua) * self.C \
                 + self.c1 * np.outer(self.pc, self.pc) \
                 + self.cmua * sum(self.weights[i] * np.outer(y_sorted[i], y_sorted[i]) for i in range(self.mu))
        # Ensure symmetry and positive definiteness
        self.C = (self.C + self.C.T) / 2
        self.D = np.sqrt(np.diag(self.C))
        self.sigma *= np.exp(self.cs / self.chiN * (np.linalg.norm(self.ps) / self.chiN - 1))
        # Update mean
        self.x = self.x + sum(self.weights[i] * (self.x + self.sigma * y_sorted[i]) - self.x for i in range(self.mu))
        self.x = np.clip(self.x, -5, 5)
        return f_sorted[0]


class Optimizer:
    """Parallel Tempering ensemble of CMA-ES instances."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n_replicas = 5
        budget_per = self.budget // n_replicas

        # Different temperature scales (initial step sizes)
        sigmas = [0.5, 1.0, 2.0, 3.0, 4.0]
        # Different starting points for diversity
        start_points = [rng.uniform(-5, 5, self.dim) for _ in range(n_replicas)]

        replicas = []
        for i in range(n_replicas):
            rep = _CMAES(self.dim, start_points[i], sigmas[i], np.random.default_rng(1000 + i), budget_per)
            # Initial evaluation
            fx = func(np.clip(rep.x, -5, 5))
            rep.calls += 1
            rep.best_f = fx
            rep.best_x = np.array(rep.x)
            replicas.append(rep)

        total_calls = n_replicas
        best_overall_f = min(r.best_f for r in replicas)
        best_overall_x = replicas[np.argmin([r.best_f for r in replicas])].best_x

        swap_interval = 20  # Swap every 20 total evals
        evals_since_swap = 0

        while total_calls < self.budget:
            # Time guard
            if time.time() > deadline - 5.0:
                break

            # Find best and worst replica
            fvals = [r.best_f for r in replicas]
            best_idx = int(np.argmin(fvals))
            worst_idx = int(np.argmax(fvals))

            # Perform CMA-ES step on the best replica (or round-robin)
            # Actually, let's do round-robin to use all replicas
            rep = replicas[total_calls % n_replicas]
            if rep.calls >= rep.budget:
                # This replica is done, try next
                continue

            f_new = rep.step(func)
            total_calls += rep.mu if rep.mu > 1 else 1
            # Actually step() already increments rep.calls by self.mu
            # Let me recount: total_calls should track actual func calls
            # rep.step evaluates self.mu times, so total_calls += self.mu
            # But I already did total_calls += rep.mu above, that's wrong
            # Let me fix: track properly

            # Actually let me restructure: rep.step does self.mu evaluations
            # I need to track total_calls correctly

            if f_new < rep.best_f:
                rep.best_f = f_new
                rep.best_x = np.array(rep.x)
            if rep.best_f < best_overall_f:
                best_overall_f = rep.best_f
                best_overall_x = np.array(rep.x)

            evals_since_swap += rep.mu

            # Periodic swap: best and worst exchange centroids and step sizes
            if evals_since_swap >= swap_interval:
                evals_since_swap = 0
                b = replicas[best_idx]
                w = replicas[worst_idx]
                # Swap centroids and step sizes
                b.x, w.x = w.x.copy(), b.x.copy()
                b.sigma, w.sigma = w.sigma, b.sigma
                # Also swap covariance (deep copy)
                b.C, w.C = w.C.copy(), b.C.copy()
                b.D, w.D = w.D.copy(), b.D.copy()

        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
