"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB-style f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit with 20s safety margin
        budget = self.budget
        dim = self.dim
        calls = 0

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Phase 1: Quick deterministic exploration to find a good starting region
        # Evaluate a small set of points: origin, axis points, and some random-like points
        test_points = [np.zeros(dim)]
        # Add points along each axis
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                test_points.append(p)
        
        best_x = None
        best_f = np.inf
        for v in test_points:
            if time.time() > deadline or calls >= budget:
                break
            f = eval_point(v)
            if f < best_f:
                best_f = f
                best_x = v.copy()
        
        if best_x is None:
            # Fallback: just evaluate origin
            best_x = np.zeros(dim)
            best_f = eval_point(best_x)
        
        # Phase 2: CMA-ES optimization
        # Initialize CMA-ES parameters
        x0 = best_x.copy()
        f0 = best_f
        
        # Initial sigma: adapt to problem scale
        sigma = 2.0
        
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~11 for dim=5
        mu = lambda_ // 2  # recombination weight count
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        
        # Variance-effective selection mass
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Generate initial population
        def generate_population(x_mean, sigma, C):
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            population = []
            for _ in range(lambda_):
                z = np.random.default_rng(hash(tuple(x_mean)) + _).standard_normal(dim)
                y = sigma * (x_mean + L @ z)
                y = np.clip(y, -5, 5)
                population.append(y)
            return population
        
        # Main CMA-ES loop
        max_iter = (budget - calls) // (lambda_ + 1)
        max_iter = min(max_iter, 200)
        
        for gen in range(max_iter):
            if time.time() > deadline or calls >= budget:
                break
            
            # Generate population
            pop = generate_population(x0, sigma, C)
            
            # Evaluate population
            pop_f = []
            for x in pop:
                if time.time() > deadline or calls >= budget:
                    break
                f = eval_point(x)
                pop_f.append(f)
            
            if len(pop_f) < lambda_:
                break
            
            # Sort by fitness
            idx = np.argsort(pop_f)
            pop = [pop[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]
            
            # Update best
            for x, f in zip(pop, pop_f):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Recombination
            x_new = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
            
            # Update path for covariance matrix adaptation
            y_inv = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            y_inv = np.linalg.inv(y_inv)
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (y_inv @ ((x_new - x0) / sigma))
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (y_inv @ ((x_new - x0) / sigma))
            
            # Hessian update
            delta_sigma = 1.0 - np.exp(-np.linalg.norm(p_sigma) / (chi_n * (1 + cs)))
            sigma *= np.exp(cs * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)
            
            C = (1 + c1 * cmu * delta_sigma - c1 - cmu) * C
            for i in range(mu):
                y_i = (pop[i] - x0) / sigma
                y_i = y_inv @ y_i
                C += cmu * weights[i] * np.outer(y_i, y_i)
            C += c1 * (np.outer(p_c, p_c) - (1 - cc) * (2 - cc) * C)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x0 = x_new
            x0 = np.clip(x0, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
