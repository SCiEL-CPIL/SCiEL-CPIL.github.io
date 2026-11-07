"""Optimizer for BBOB f6 (Attractive sector) using CMA-ES with global exploration."""
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for f6 with global-to-local strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_x = x_clipped.copy()
                best_f = f
            return f

        # Phase 1: Global exploration with 50 uniform random evaluations
        # This helps identify a good basin early for AOCC
        rng = np.random.default_rng(42)
        n_global = min(50, budget)
        
        # Include origin and some structured points
        candidates = [np.zeros(dim)]
        for _ in range(n_global - 1):
            candidates.append(rng.uniform(-4.0, 4.0, dim))
        
        # Evaluate all candidates
        eval_results = []
        for x in candidates:
            if calls >= budget or time.time() > deadline:
                break
            f = eval_func(x)
            eval_results.append((f, x))
        
        if not eval_results:
            # Fallback
            x0 = np.zeros(dim)
            eval_func(x0)
            return best_x, best_f

        # Sort by fitness
        eval_results.sort(key=lambda x: x[0])
        x0 = eval_results[0][1]
        
        # Phase 2: CMA-ES local optimization from the best point
        remaining_budget = budget - calls
        
        if remaining_budget < 20:
            return best_x, best_f

        # CMA-ES implementation
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        pop_size = min(pop_size, 12)
        
        # Initial sigma based on box size
        sigma = 1.5
        
        # CMA-ES parameters
        cc = (4 + dim / dim) / (dim + 4.1)
        cs = (dim + 1.7) / (dim + 9.2)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - cc)
        cmu = (2 / ((dim + 2) ** 2 + mu_eff)) * (min(1 - c1, 2 / ((dim + 1) ** 2 + mu_eff)))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initialize CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute Cholesky factor
        def chol(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                return eigvecs @ np.diag(np.sqrt(eigvals))
        
        B = chol(C)
        D = np.sqrt(np.diag(B.T @ B))
        
        # CMA-ES loop
        iter_count = 0
        max_iter = remaining_budget // pop_size + 10
        
        for iteration in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break
            
            # Generate population
            population = []
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (B @ z)
                f = eval_func(x)
                population.append((f, x))
            
            if len(population) < pop_size:
                break
            
            # Sort by fitness
            population.sort(key=lambda x: x[0])
            
            # Update best
            if population[0][0] < best_f:
                best_x = population[0][1].copy()
                best_f = population[0][0]
            
            # Check convergence
            if sigma < 1e-12:
                break
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i][1]
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / D @ (B.T @ (new_mean - mean))
            if np.linalg.norm(ps) / chi_n < (1.4 + cc) * np.sqrt(1 + (1 - cs) ** iter_count):
                pc = (1 - cc) * pc + cc * (new_mean - mean) / sigma
            
            # Update covariance matrix
            delta = 1 if np.linalg.norm(ps) / chi_n <= (1.4 + cc) * np.sqrt(1 + (1 - cs) ** iteration) else 0
            C = (1 + c1 * delta - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                diff = (population[i][1] - mean) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 10.0)  # Cap sigma
            sigma = max(sigma, 1e-12)
            
            mean = new_mean
            
            # Update Cholesky factor
            B = chol(C)
            D = np.sqrt(np.diag(B.T @ B))
            
            # Check for stagnation
            if iteration > 20 and np.all(np.abs(np.diag(C)) < 1e-8):
                break
        
        return best_x, best_f
