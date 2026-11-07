"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        # CMA-ES hyperparameters
        n = self.dim
        # lambda population size
        lam = 4 + int(3 * np.log(n))
        # mu number of parents
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        # step size adaptation
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        # covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - c_c + mueff / (n + 1.8)) / ((n + 1.3 + c_c) * (n + 0.5 + mueff / (n + 1.8)))
        c_mu = (2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (n + 0.5 + mueff / (n + 1.8))) - c_1) / (1 + c_c)
        
        # initial state
        x = np.random.default_rng(0).uniform(-5, 5, n)
        sigma = 2.0
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # chi_n for normalization
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # initial population
        rng = np.random.default_rng(1)
        pop = [x + sigma * (B @ (D * rng.standard_normal(n))) for _ in range(lam)]
        pop = [np.clip(p, -5, 5) for p in pop]
        vals = [func(p) for p in pop]
        
        best_x = pop[np.argmin(vals)]
        best_f = min(vals)
        
        # sort population
        idx = np.argsort(vals)
        pop = [pop[i] for i in idx]
        vals = [vals[i] for i in idx]
        
        # main loop
        calls_made = lam
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * (pop[i] - x)
            x_new = x + y_w
            
            # update step size
            y_sigma = (B @ (D * p_sigma)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * chi_n) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # update covariance
            y_c = (B @ (D * p_c)) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * y_c
            delta_heav = (1 - c_c) * (1 - c_c**2) * np.linalg.norm(p_c)**2 / n
            C = (1 + c_1 + c_mu - c_c * c_c) * C
            C -= c_1 * (np.outer(p_c, p_c) - C)
            for i in range(mu):
                z = (pop[i] - x) / sigma
                z = np.linalg.solve(B, z / D)
                C += c_mu * w[i] * np.outer(z, z)
            
            # re-orthogonalize B and D
            # Use SVD for stability
            U, S, Vt = np.linalg.svd(C)
            B = U @ Vt
            D = np.sqrt(S)
            
            x = x_new
            
            # generate new population
            new_pop = [x + sigma * (B @ (D * rng.standard_normal(n))) for _ in range(lam)]
            new_pop = [np.clip(p, -5, 5) for p in new_pop]
            new_vals = [func(p) for p in new_pop]
            calls_made += lam
            
            # update best
            min_val = min(new_vals)
            if min_val < best_f:
                best_f = min_val
                best_x = new_pop[np.argmin(new_vals)]
            
            # sort new population
            idx = np.argsort(new_vals)
            pop = [new_pop[i] for i in idx]
            vals = [new_vals[i] for i in idx]
            
            # early stopping if sigma is too small or no improvement for a long time
            if sigma < 1e-12:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
