"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
CMA-ES adapts the search distribution to the local curvature, which is
crucial for f14 where different dimensions have different powers.
It is robust to initialization and efficient for small dimensions.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f14 with hidden affine transform."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(d))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff)))
        
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # State
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        best_x = x.copy()
        best_f = float(func(x))
        calls = 1
        
        while calls < budget and time.time() < deadline:
            # Generate population
            pop = []
            for _ in range(lambda_):
                z = np.random.randn(d)
                try:
                    L = np.linalg.cholesky(C)
                    xi = x + sigma * (L @ z)
                except:
                    xi = x + sigma * z
                xi = np.clip(xi, -5.0, 5.0)
                fi = float(func(xi))
                calls += 1
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                pop.append((xi, fi))
            
            if calls >= budget:
                break

            # Sort
            pop.sort(key=lambda t: t[1])
            
            # Selection
            parents = np.array([pop[i][0] for i in range(mu)])
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * parents, axis=0)
            
            # Update ps
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except:
                C_inv_sqrt = np.eye(d)
            
            step_size = np.sqrt(mu_eff * c_sigma * (2 - c_sigma) / c_sigma)
            ps = (1 - c_sigma) * ps + step_size * (C_inv_sqrt @ (x_new - x))
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (x_new - x) / sigma
            
            # Update C
            rank_1 = pc * pc.T
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(parents - x, parents - x), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            C = (C + C.T) / 2
            
            # Ensure PD
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x = x_new
            
        return best_x, float(best_f)
