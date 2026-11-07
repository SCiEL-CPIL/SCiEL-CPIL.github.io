"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES + LHS global search + L-BFGS-B local exploitation for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        global_best = None
        global_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Latin Hypercube Sampling to find a good starting basin
        n_lhs = 100
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            edges = np.linspace(-4.0, 4.0, n_lhs + 1)
            for i in range(n_lhs):
                lhs_points[i, d] = rng.uniform(edges[i], edges[i+1])
        perm = rng.permutation(n_lhs)
        lhs_points = lhs_points[perm]

        for i in range(n_lhs):
            if calls_made >= budget - 200 or time.time() > deadline - 60:
                break
            evaluate(lhs_points[i])

        if global_best is None:
            return None, np.inf

        # Phase 2: CMA-ES for global optimization from the best LHS point
        # CMA-ES is well-suited for multimodal functions like Lunacek
        if calls_made < budget - 100 and time.time() < deadline - 40:
            x0 = global_best.copy()
            remaining = budget - calls_made

            # CMA-ES with adaptive step size
            # Start with a moderate sigma to explore the neighborhood
            sigma0 = 1.0

            class CMAES:
                def __init__(self, x0, sigma, dim, budget, evaluate_func, bounds):
                    self.x = x0.copy()
                    self.sigma = sigma
                    self.dim = dim
                    self.budget = budget
                    self.evaluate = evaluate_func
                    self.bounds = bounds
                    self.calls = 0
                    self.pop_size = max(4, int(4 + 3 * np.log(dim)))

                    # CMA-ES parameters
                    self.C = np.eye(dim)
                    self.p_sigma = np.zeros(dim)
                    self.p_c = np.zeros(dim)
                    self.c_sigma = (self.pop_size + 2) / ((dim + 2) ** 2 + self.pop_size)
                    self.d_sigma = 1 + 2 * max(0, np.sqrt((self.pop_size - 1) / (dim + 1)) - 1) + self.c_sigma
                    self.c_c = (4 + self.pop_size / dim) / (dim + 4 + 2 * self.pop_size / dim)
                    self.c_1 = 2 / ((dim + 1.3) ** 2 + self.pop_size)
                    self.c_mu = min(1 - self.c_1, 2 * (self.pop_size / 2 - 1 + 1 / self.pop_size) / ((dim + 2) ** 2 + self.pop_size))
                    self.mu = self.pop_size // 2
                    self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                    self.weights /= self.weights.sum()
                    self.mu_eff = 1 / (self.weights ** 2).sum()

                def sample(self):
                    z = np.random.randn(self.dim)
                    # Transform using C^0.5
                    # Use eigen decomposition for C^0.5
                    eigvals, eigvecs = np.linalg.eigh(self.C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    x_sample = self.x + self.sigma * (C_sqrt @ z)
                    # Clip to bounds
                    x_sample = np.clip(x_sample, self.bounds[0], self.bounds[1])
                    return x_sample

                def step(self):
                    if self.calls >= self.budget:
                        return False

                    # Generate population
                    population = []
                    for _ in range(self.pop_size):
                        x_sample = self.sample()
                        fx = self.evaluate(x_sample)
                        self.calls += 1
                        population.append((fx, x_sample))
                        if time.time() > deadline - 30:
                            return False

                    # Sort by fitness
                    population.sort(key=lambda x: x[0])

                    # Update mean
                    w = self.weights
                    new_mean = sum(w[i] * population[i][1] for i in range(self.mu))

                    # Update covariance matrix
                    delta_mean = new_mean - self.x
                    self.x = new_mean

                    # Update step size
                    c_sigma = self.c_sigma
                    d_sigma = self.d_sigma
                    p_sigma = self.p_sigma + c_sigma * (self.sigma / self.sigma) * (delta_mean / self.sigma)
                    # Actually, p_sigma update: p_sigma = (1-c_sigma)*p_sigma + c_sigma * sqrt(mu_eff) * delta_mean / sigma
                    p_sigma = (1 - c_sigma) * self.p_sigma + c_sigma * np.sqrt(self.mu_eff) * delta_mean / self.sigma
                    h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (self.calls + 1)) < \
                                (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.mu_eff - 1 + 1) / (self.dim + 1)) else 0
                    self.sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(self.mu_eff) - 1))
                    self.p_sigma = p_sigma

                    # Update evolution paths
                    c_c = self.c_c
                    p_c = (1 - c_c) * self.p_c + h_sigma * c_c * delta_mean / self.sigma
                    self.p_c = p_c

                    # Update covariance matrix
                    c_1 = self.c_1
                    c_mu = self.c_mu
                    rank1 = np.outer(p_c, p_c)
                    rankmu = sum(w[i] * np.outer(population[i][1] - self.x, population[i][1] - self.x) for i in range(self.mu))
                    self.C = (1 + c_1 + c_mu - self.c_1) * self.C + c_1 * (rank1 - self.C) + c_mu * rankmu

                    # Ensure C is positive semi-definite
                    eigvals, eigvecs = np.linalg.eigh(self.C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                    return True

            try:
                cmaes = CMAES(x0, sigma0, dim, remaining - 20, evaluate, (-5.0, 5.0))
                while cmaes.step() and calls_made < budget - 50 and time.time() < deadline - 30:
                    pass
            except Exception:
                pass

        # Phase 3: L-BFGS-B polish from the best point
        if global_best is not None and calls_made < budget - 20 and time.time() < deadline - 20:
            x0 = global_best.copy()
            remaining = budget - calls_made

            maxfun = min(remaining - 10, 200)
            try:
                result = minimize(
                    evaluate,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxfun': maxfun, 'maxiter': 100}
                )
            except Exception:
                pass

        # Phase 4: If budget remains, do another round of exploration
        if calls_made < budget - 30 and time.time() < deadline - 15:
            rng2 = np.random.default_rng(54321)
            n_lhs2 = min(50, budget - calls_made - 20)
            if n_lhs2 > 10:
                lhs_points2 = np.zeros((n_lhs2, dim))
                for d in range(dim):
                    edges = np.linspace(-4.0, 4.0, n_lhs2 + 1)
                    for i in range(n_lhs2):
                        lhs_points2[i, d] = rng2.uniform(edges[i], edges[i+1])
                perm2 = rng2.permutation(n_lhs2)
                lhs_points2 = lhs_points2[perm2]

                for i in range(n_lhs2):
                    if calls_made >= budget - 10 or time.time() > deadline - 10:
                        break
                    evaluate(lhs_points2[i])

                # L-BFGS-B polish from new best if improved
                if global_best is not None and calls_made < budget - 10 and time.time() < deadline - 10:
                    x0 = global_best.copy()
                    remaining = budget - calls_made
                    maxfun = min(remaining - 5, 100)
                    try:
                        result = minimize(
                            evaluate,
                            x0,
                            method='L-BFGS-B',
                            bounds=[(-5.0, 5.0)] * dim,
                            options={'maxfun': maxfun, 'maxiter': 50}
                        )
                    except Exception:
                        pass

        return global_best, global_fx
# EVOLVE-BLOCK-END
