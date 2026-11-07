"""Rewritten optimizer for BBOB f23 (Katsuura) with CMA-ES and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Katsuura function optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES for a given budget, return best x, fx."""
        d = self.dim
        N = 4 + int(3 * np.log(d))  # population size
        N = max(2, min(N, budget // 2))
        
        # CMA-ES parameters
        cc = (4 + d / (d + 4)) / ((d + 1.7) + (4 + d / (d + 4)))
        cs = (d + 1.9) / (d + 6.5)
        cmu = min(1 - cc, 2 / ((d + 1.7) ** 2 + d))
        
        # Initialize
        x = x0.copy()
        fx = func(x)
        sigma = sigma0
        C = np.eye(d)
        
        # Compute matrix square roots
        eigvals, eigvecs = np.linalg.eigh(C)
        C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
        
        # Cumulation paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        calls = 1
        best_x = x.copy()
        best_fx = fx
        chi_N = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Weights
            weights = np.ones(N) / (N / 2)
            mueff = int(np.sum(weights[:max(1, N // 2)]) ** 2 / np.sum(weights ** 2))
            mueff = max(1, min(mueff, N // 2))
            
            # Sample N points
            Y = rng.standard_normal((N, d))
            X = x + sigma * (Y @ C_sqrt)
            
            # Evaluate
            F = np.array([func(np.clip(xi, -5, 5)) for xi in X])
            calls += N
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = np.clip(X_sorted[0], -5, 5)
            
            # Update mean
            w_sum = np.sum(weights[:mueff])
            mu = X_sorted[:mueff].T @ (weights[:mueff] / w_sum)
            x_new = x + (mu - x)
            
            # Update sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (mu - x) / sigma
            if np.linalg.norm(p_sigma) / chi_N < 1.4 + 2 / (d + 1):
                sigma *= np.exp(cs / (5 * N) * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update covariance
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * (mu - x) / sigma
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * max(1, calls // N))) < (1.4 + 2 / (d + 1)) * chi_N
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            
            C_new = (1 + cmu * delta_hsig - cs * (2 - cs)) * C
            for i in range(mueff):
                yi = (X_sorted[i] - x) / sigma
                C_new += cmu * weights[i] / w_sum * np.outer(yi, yi)
            
            C_new = (C_new + C_new.T) / 2
            
            # Ensure positive definiteness
            try:
                eigvals_new, eigvecs_new = np.linalg.eigh(C_new)
                eigvals_new = np.maximum(eigvals_new, 1e-10)
                C = (eigvecs_new * eigvals_new) @ eigvecs_new.T
                C_inv_sqrt = (eigvecs_new / np.sqrt(eigvals_new)) @ eigvecs_new.T
                C_sqrt = (eigvecs_new * np.sqrt(eigvals_new)) @ eigvecs_new.T
            except np.linalg.LinAlgError:
                C = np.eye(d)
                C_inv_sqrt = np.eye(d)
                C_sqrt = np.eye(d)
            
            x = np.clip(x_new, -5, 5)
        
        return best_x, best_fx

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # safety margin
        
        rng = np.random.default_rng(12345)
        
        # Phase 1: Initial sampling
        n_init = min(40, B // 10)
        X_init = rng.uniform(-5, 5, (n_init, d))
        F_init = np.array([func(xi) for xi in X_init])
        calls = n_init
        
        best_idx = np.argmin(F_init)
        x0 = X_init[best_idx].copy()
        fx0 = F_init[best_idx]
        best_x = x0.copy()
        best_fx = fx0
        
        # Phase 2: CMA-ES from best point
        remaining = B - calls
        if remaining > 20:
            x_best, fx_best = self._cma_es(func, x0, 1.0, remaining, rng, deadline)
            if fx_best < best_fx:
                best_x = x_best
                best_fx = fx_best
        
        # Phase 3: One restart if time allows
        if time.time() < deadline - 15 and calls < B - 20:
            x_alt = rng.uniform(-5, 5, d)
            fx_alt = func(x_alt)
            calls += 1
            if fx_alt < best_fx:
                best_x = x_alt
                best_fx = fx_alt
                remaining2 = B - calls
                if remaining2 > 20:
                    x_best2, fx_best2 = self._cma_es(func, x_alt, 0.5, remaining2, rng, deadline)
                    if fx_best2 < best_fx:
                        best_x = x_best2
                        best_fx = fx_best2
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
