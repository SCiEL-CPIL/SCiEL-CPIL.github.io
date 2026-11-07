"""Optimized black-box continuous optimizer for BBOB-style functions."""
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is robust to rotations and multimodality, suitable for BBOB functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Initial Sampling
        n_init = max(10, self.budget // 10)
        init_points = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            init_points.append((fx, x))
            
        if best_x is None:
            x0 = np.zeros(dim)
            best_f = evaluate(x0)
            best_x = x0
            init_points = [(best_f, x0)]

        init_points.sort(key=lambda t: t[0])
        x_mean = init_points[0][1].copy()
        
        # Estimate initial sigma
        if len(init_points) > 5:
            good_xs = np.array([p[1] for p in init_points[:len(init_points)//2]])
            sigma = np.std(good_xs, axis=0).mean()
            sigma = np.clip(sigma, 0.5, 2.0)
        else:
            sigma = 1.0

        # Phase 2: CMA-ES
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cm = 1.0 / (dim**1.8 + mueff)
        c1 = (2 / ((dim + 1.3)**2 + mueff))
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(20*dim**2))
        
        gen = 0
        remaining = self.budget - calls_made
        
        while calls_made < self.budget and time.time() < deadline:
            if remaining < lam:
                break
                
            # Sample
            population = []
            for _ in range(lam):
                z = rng.multivariate_normal(np.zeros(dim), C)
                x_new = x_mean + sigma * z
                population.append(x_new)
            
            # Evaluate
            f_values = np.array([evaluate(x) for x in population])
            
            # Sort
            idx = np.argsort(f_values)
            population = np.array(population)[idx]
            f_values = f_values[idx]
            
            # Update Mean
            x_new_mean = np.sum(weights[:, None] * population[:mu], axis=0)
            
            # Update Step Size
            # Calculate hsig
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (x_new_mean - x_mean)
            
            # Note: Standard CMA-ES uses C^-1/2 transformation in ps update.
            # For dim=5, we compute it.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (x_new_mean - x_mean))
            
            norm_ps = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1-cs)**(2*(gen+1)/cs))
            if denom < 1e-10:
                denom = 1e-10
            hsig = 1 if norm_ps / denom < (1.4 + 2.0 / (dim + 1)) * chi_n else 0
            
            # Update pc
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_new_mean - x_mean) / sigma
            
            # Update C
            y = np.array([(population[i] - x_mean) / sigma for i in range(mu)])
            
            # Rank-1 update
            C = (1 + cmu * (hsig - 1) - c1 - cmu) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            for i in range(mu):
                C += cmu * weights[i] * np.outer(y[i], y[i])
                
            # Symmetrize
            C = (C + C.T) / 2.0
            C += 1e-10 * np.eye(dim)
            
            # Update Mean and Sigma
            x_mean = x_new_mean
            sigma *= np.exp(cs / ds * (norm_ps / chi_n - 1))
            
            gen += 1
            remaining -= lam
            
            # Convergence check
            if np.std(f_values[:mu]) < 1e-10 * (1 + abs(best_f)):
                break

        return best_x, best_f
