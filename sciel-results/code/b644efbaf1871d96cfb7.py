"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts for rotated Weierstrass landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0

        def cma_es_run(center, sigma_init, max_calls):
            nonlocal best_x, best_fx, calls
            
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2  # number of parents
            # Weights (log-sum)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w**2)
            
            # CMA parameters
            c_sigma = (mueff + 2) / ((n + 8) * (mueff / 2 + 5))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = (2 - mueff / (n + 1) + 2 / (n + 1)) / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1. / mueff) / ((n + 2)**2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
            
            # State
            x_mean = center.copy()
            sigma = sigma_init
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            no_improve = 0
            local_best = np.inf
            
            while calls < budget and (budget - calls) >= lam and time.time() < deadline:
                # Sample population
                samples = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (B @ (D * z))
                    samples.append(y)
                
                # Evaluate
                vals = []
                for y in samples:
                    if calls >= budget or time.time() > deadline:
                        break
                    fy = func(np.clip(y, -5, 5))
                    calls += 1
                    vals.append(fy)
                    if fy < best_fx:
                        best_fx = fy
                        best_x = np.clip(y, -5, 5).copy()
                    if fy < local_best:
                        local_best = fy
                        no_improve = 0
                    else:
                        no_improve += 1
                
                if len(vals) < lam:
                    break
                
                # Sort
                idx = np.argsort(vals)
                y_sorted = [samples[i] for i in idx]
                f_sorted = [vals[i] for i in idx]
                
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = sum(w[i] * y_sorted[i] for i in range(mu))
                
                # Check for convergence
                if np.linalg.norm(x_mean - x_mean_old) < 1e-12 * max(1.0, sigma):
                    break
                
                # Update step size
                y_mean = (x_mean - x_mean_old) / (sigma * np.linalg.norm(B.T @ (D * (B.T @ (x_mean - x_mean_old) / sigma)))) if np.linalg.norm(x_mean - x_mean_old) > 1e-15 else np.zeros(n)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (y_mean / chi_n)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Clamp sigma
                sigma = max(1e-10, min(sigma, 10.0))
                
                # Update covariance
                y_w = [y_sorted[i] - x_mean_old for i in range(mu)]
                delta_h = (1 - c_c) * np.linalg.norm(p_c) / chi_n
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_mean_old) / (sigma * chi_n)
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C
                for i in range(mu):
                    C += c_mu * w[i] * np.outer(y_w[i] / sigma, y_w[i] / sigma)
                
                # Eigen decomposition (recompute periodically for numerical stability)
                if calls % (lam * 5) == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                    C = B @ (D**2) @ B.T
                    # Symmetrize
                    C = (C + C.T) / 2
                
                # Stop if no improvement for too long
                if no_improve > 3 * lam:
                    break
                
                # Stop if sigma too small
                if sigma < 1e-8:
                    break

            return best_x, best_fx

        # Phase 1: Multiple CMA-ES restarts with increasing focus
        # First restart: broad exploration
        if calls < budget:
            center = rng.uniform(-3, 3, dim)
            cma_es_run(center, 2.0, budget)

        # Second restart: medium focus around best
        if best_x is not None and calls < budget:
            center = best_x + rng.normal(0, 0.5, dim)
            center = np.clip(center, -5, 5)
            cma_es_run(center, 1.0, budget)

        # Third restart: tight focus
        if best_x is not None and calls < budget:
            center = best_x + rng.normal(0, 0.2, dim)
            center = np.clip(center, -5, 5)
            cma_es_run(center, 0.5, budget)

        # Final phase: local fine-tuning with small (1+1)-ES
        if best_x is not None and calls < budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.05
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
