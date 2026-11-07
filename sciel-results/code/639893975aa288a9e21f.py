"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget

        # Global best tracking
        best_x = None
        best_f = np.inf

        # We'll use a population-based approach (CMA-ES-like) with restarts
        # Strategy: Use a small population CMA-ES with adaptive step size
        # and restart when progress stalls.

        # Initial population size
        pop_size = 10
        # Each "generation" uses pop_size evaluations
        # Total generations = budget / pop_size

        # We'll do multiple restarts if needed
        # For simplicity, let's do a single run with good parameters first

        # CMA-ES parameters
        lambda_ = pop_size  # population size
        # Step size
        sigma = 3.0  # initial step size (box is [-5,5]^5)

        # Initial center
        c = np.zeros(dim)

        # Covariance matrix (initially identity)
        C = np.eye(dim)
        # Eigen decomposition for sampling
        # We'll use Cholesky or just sample and transform

        # Learning rates
        cc = (4 + dim / (dim + 4)) / ((dim + 2) ** 2 + 4)
        cmu = min(0.75, 2 * (lambda_ - 2 + 1. / (dim + 2)) / ((dim + 2) ** 2 + lambda_ + 2))
        # Actually for small lambda, use simpler adaptation

        # Let's use a simpler approach: (mu, lambda)-CMA with rank-1 and rank-mu updates

        mu = lambda_ // 2  # number of best solutions used for update
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Success rate tracking for sigma adaptation
        # We'll use a simple approach: if we make progress, increase sigma slightly

        remaining = total_budget
        x_best = None
        f_best = np.inf

        # First evaluation
        x0 = np.random.uniform(-5, 5, dim)
        f0 = func(x0)
        remaining -= 1
        x_best = x0.copy()
        f_best = f0
        best_x = x0.copy()
        best_f = f0

        # If we have enough budget, run CMA-ES
        if remaining < lambda_:
            return x_best, f_best

        # CMA-ES loop
        while remaining >= lambda_:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Sample from N(c, sigma^2 * C)
            # Use Cholesky decomposition of C
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            
            # Sample
            z = np.random.randn(lambda_, dim)
            # Transform: c + sigma * z @ L.T
            pops = c[None, :] + sigma * (z @ L.T)
            # Clip to box
            pops = np.clip(pops, -5, 5)

            # Evaluate
            fs = np.empty(lambda_)
            for i in range(lambda_):
                fs[i] = func(pops[i])
                remaining -= 1
                if fs[i] < f_best:
                    f_best = fs[i]
                    x_best = pops[i].copy()
                    best_x = x_best.copy()
                    best_f = f_best

            if remaining < 1:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            fs_sorted = fs[idx]
            pops_sorted = pops[idx]

            # Update mean
            c_new = np.sum(w[:, None] * pops_sorted[:mu], axis=0)
            
            # Update evolution path ps
            inv_sqrt_C = np.linalg.inv(L)  # L is Cholesky, so L @ L.T = C
            # Actually, we need C^{-1/2}
            # For simplicity, let's use a simplified update
            
            # Simplified CMA-ES updates
            # Update ps
            ps = (1 - cc) * ps + np.sqrt(cc * (2 - cc) * mueff) * inv_sqrt_C.T @ (c_new - c) / sigma
            
            # Check for divergence
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cc) ** (2 * remaining)) > 1.4 * (1 + cmu) * np.sqrt(dim):
                # Reset ps
                ps = np.zeros(dim)
            
            # Update pc
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cc) ** 2) < (1.4 + cmu) * np.sqrt(dim):
                pc = (1 - cc) * pc + cc * (c_new - c) / sigma
            
            # Update covariance
            delta_pc = (1 + cmu) * cc * pc * pc.T - cc * C
            v = np.zeros((dim, mu))
            for i in range(mu):
                v[:, i] = (pops_sorted[i] - c) / sigma
            C = (1 + 2 * cmu) * C + cmu * (v @ (w[:, None] * v).T) + (1 - cmu - 2 * cmu) * delta_pc
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(dim)
            
            # Update sigma
            # Use target success rate of ~1/5
            # Simple approach: adjust sigma based on progress
            # If we made progress, keep sigma; otherwise reduce
            progress = fs_sorted[0] < fs_sorted[-1] * 0.999 or (fs_sorted[-1] - fs_sorted[0]) < 1e-10
            # Actually, let's use a simpler adaptation
            # Count successes in last lambda
            # For now, just keep sigma constant or use a simple rule
            # Let's reduce sigma slightly each generation for convergence
            sigma *= 0.95
            if sigma < 0.01:
                sigma = 0.01

            c = c_new

            # Check for stagnation: if no improvement in best for many generations
            # We'll track this with a counter
            # For simplicity, let's just continue

        # If we have remaining budget, do local refinement around best
        if remaining > 0 and time.time() < deadline:
            # Local search around best_x
            local_sigma = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + local_sigma * np.random.randn(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    best_x = x.copy()
                    best_f = fx
                    local_sigma *= 1.1
                else:
                    local_sigma *= 0.9
                    if local_sigma < 0.001:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
