"""Differential Evolution with adaptive parameters and local CMA-ES polish for f17."""
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

    def step(self, func, max_calls):
        """Perform one CMA-ES iteration: sample mu offspring, evaluate, update."""
        d = self.dim
        if self.calls >= max_calls:
            return self.best_f
        # Cholesky of C
        try:
            L = np.linalg.cholesky(self.C + 1e-12 * np.eye(d))
            B = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            self.C = np.eye(d)
            L = np.eye(d)
            B = np.eye(d)
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
        self.sigma = np.clip(self.sigma, 1e-10, 100.0)
        # Update mean
        self.x = self.x + sum(self.weights[i] * (self.x + self.sigma * y_sorted[i]) - self.x for i in range(self.mu))
        self.x = np.clip(self.x, -5, 5)
        return f_sorted[0]


class Optimizer:
    """Differential Evolution with adaptive parameters and local CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(np.clip(x, -5, 5)))
            if f < best_f:
                best_f = f
                best_x = np.clip(x, -5, 5).copy()
            return f

        rng = np.random.default_rng(42)

        # Differential Evolution Parameters
        NP = 12
        F = 0.8
        CR = 0.8
        F_min, F_max = 0.3, 1.2
        CR_min, CR_max = 0.5, 0.95

        # 1. Initialization
        population = np.zeros((NP, dim))
        fitness = np.zeros(NP)
        for i in range(NP):
            x = rng.uniform(-4.0, 4.0, size=dim)
            population[i] = x
            fitness[i] = eval_func(x)

        idx_best = np.argmin(fitness)
        best_f = fitness[idx_best]
        best_x = population[idx_best].copy()

        # 2. Main DE Loop
        no_improve = 0
        while calls < budget - 60 and time_ok():
            for i in range(NP):
                if calls >= budget - 60 or not time_ok():
                    break

                # Select 3 distinct random indices from [0, NP) excluding i
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)

                # Adaptive F and CR
                F_adapt = F_min + (F_max - F_min) * (1 - no_improve / 50.0)
                CR_adapt = CR_min + (CR_max - CR_min) * (1 - no_improve / 50.0)
                F_adapt = np.clip(F_adapt, F_min, F_max)
                CR_adapt = np.clip(CR_adapt, CR_min, CR_max)

                # Mutation: DE/rand/1
                v = population[r1] + F_adapt * (population[r2] - population[r3])
                v = np.clip(v, -5.0, 5.0)

                # Crossover: Binomial
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR_adapt or j == j_rand:
                        trial[j] = v[j]

                # Evaluate and Select
                f_trial = eval_func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    no_improve = 0
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    no_improve += 1

        # 3. Local CMA-ES Polish
        if calls < budget and time_ok():
            remaining = budget - calls
            if remaining > 30:
                cma_budget = min(remaining, 150)
                cma = _CMAES(dim, best_x, 0.5, np.random.default_rng(12345))
                # Initial evaluation for CMA-ES
                fx = eval_func(cma.x)
                cma.calls = 1
                cma.best_f = fx
                cma.best_x = np.clip(cma.x, -5, 5)
                if fx < best_f:
                    best_f = fx
                    best_x = cma.best_x.copy()

                while calls < budget and cma.calls < cma_budget and time_ok():
                    f_new = cma.step(func, cma_budget)
                    if f_new < best_f:
                        best_f = f_new
                        best_x = cma.best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
