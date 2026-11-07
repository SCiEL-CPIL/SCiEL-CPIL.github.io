"""Multi-start CMA-ES with adaptive restarts and early exploration for f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class _CMAES:
    """Minimal CMA-ES with covariance adaptation."""

    def __init__(self, dim, x0, sigma0, rng):
        self.dim = dim
        self.x = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.C = np.eye(dim)
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
        self.calls = 0
        self.best_f = np.inf
        self.best_x = self.x.copy()

    def step(self, func):
        """Perform one CMA-ES iteration: sample mu offspring, evaluate, update."""
        d = self.dim
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
        # Update best
        if f_sorted[0] < self.best_f:
            self.best_f = f_sorted[0]
            self.best_x = np.clip(offsprings[order[0]], -5, 5)
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
        self.sigma *= np.exp(self.cs / self.chiN * (np.linalg.norm(self.ps) / self.chiN - 1))
        # Update mean
        self.x = self.x + sum(self.weights[i] * (self.x + self.sigma * y_sorted[i]) - self.x for i in range(self.mu))
        self.x = np.clip(self.x, -5, 5)
        return f_sorted[0]


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        total_calls = 0
        best_f = np.inf
        best_x = np.zeros(self.dim)

        # Phase 1: Random exploration to find good starting points
        n_explore = 30
        explore_points = []
        for i in range(n_explore):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            total_calls += 1
            explore_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                break

        # Sort by fitness
        explore_points.sort(key=lambda t: t[0])

        # Phase 2: Multi-start CMA-ES with adaptive restarts
        # Use top 5 starting points
        n_starts = min(5, n_explore)
        stagnation_window = 40  # No improvement for this many evals triggers restart

        for start_idx in range(n_starts):
            if total_calls >= self.budget or time.time() > deadline:
                break

            # Allocate remaining budget to this start
            remaining = self.budget - total_calls
            alloc = max(20, remaining // (n_starts - start_idx))

            # Start from the best point found so far (or the explore point)
            if start_idx == 0:
                x0 = best_x.copy()
            else:
                x0 = explore_points[start_idx][1].copy()

            # Adaptive sigma based on problem scale
            sigma0 = 1.0

            cma = _CMAES(self.dim, x0, sigma0, np.random.default_rng(1000 + start_idx))
            # Initial evaluation
            fx = func(np.clip(cma.x, -5, 5))
            total_calls += 1
            cma.calls = 1
            cma.best_f = fx
            cma.best_x = np.clip(cma.x, -5, 5)
            if fx < best_f:
                best_f = fx
                best_x = cma.best_x.copy()

            best_f_this_run = fx
            calls_since_improvement = 0

            # Run CMA-ES until budget exhausted or stagnation detected
            while total_calls < self.budget and cma.calls < alloc:
                if time.time() > deadline:
                    break

                f_new = cma.step(func)
                total_calls += cma.mu

                if f_new < best_f_this_run:
                    best_f_this_run = f_new
                    calls_since_improvement = 0
                    if f_new < best_f:
                        best_f = f_new
                        best_x = cma.best_x.copy()
                else:
                    calls_since_improvement += cma.mu

                # Stagnation check: restart if no improvement for stagnation_window evals
                if calls_since_improvement >= stagnation_window:
                    break

            # Update start point for next restart to current best
            if start_idx < n_starts - 1:
                explore_points[start_idx + 1] = (best_f, best_x.copy())

        return best_x, best_f
# EVOLVE-BLOCK-END
