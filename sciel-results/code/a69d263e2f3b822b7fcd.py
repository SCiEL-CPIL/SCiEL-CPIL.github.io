"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        lam = max(lam, 10)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        max_generations = budget // lam
        
        for gen in range(max_generations):
            if calls >= budget:
                break
            
            population = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = B @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                fx = func(x_new)
                calls += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()
                
                population.append((fx, x_new, y))
                
                if calls >= budget:
                    break
            
            if len(population) < lam:
                break
                
            population.sort(key=lambda x: x[0])
            parents = population[:mu]
            
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * parents[i][1]
            
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (B @ (x_new_mean - x_mean) / sigma)
            
            c_sigma_hat = np.sqrt(dim) * (1 + 2 * max(0, (cc - 1) / (dim + 2)))
            sigma *= np.exp(cs * (np.linalg.norm(pc) / c_sigma_hat - 1))
            
            if sigma > 10: sigma = 10
            if sigma < 1e-8: sigma = 1e-8
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (x_new_mean - x_mean) / sigma)
            
            v2 = np.zeros((dim, dim))
            sum_wi_yi_norm_sq = 0
            for i in range(mu):
                y_i = B @ (parents[i][1] - x_mean) / sigma
                v2 += weights[i] * np.outer(y_i, y_i)
                sum_wi_yi_norm_sq += weights[i] * np.dot(y_i, y_i)
            
            C = (1 + c1 * (np.dot(pc, pc) / dim - 1) + cmu * (sum_wi_yi_norm_sq / dim - 1)) * C + c1 * np.outer(pc, pc) + cmu * v2
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            
            B = eigvecs
            D = np.sqrt(eigvals)
            x_mean = x_new_mean
            
        return best_x, best_f
# EVOLVE-BLOCK-END
