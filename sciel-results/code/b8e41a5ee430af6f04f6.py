"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts and time guard for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial sigma0, using up to budget calls."""
        n = self.dim
        x = x0.copy()
        sigma = sigma0
        fx = func(x)
        calls = 1
        
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1.7) + c_sigma) / ((n + 2) * (2 + mueff / (n + 1.7) - c_sigma))
        c_mu = (2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (2 - c_1) + 2 * (mueff - 2 + 1 / mueff) - c_sigma))
        c_mu = min(1 - c_1, c_mu)
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_x = x.copy()
        best_f = fx
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offsets = []
            for i in range(lam):
                z = rng.standard_normal(n)
                offsets.append(z)
            
            # Evaluate population
            pop = []
            for z in offsets:
                x_trial = np.clip(x + sigma * (B @ (D * z)), -5, 5)
                f_trial = func(x_trial)
                calls += 1
                pop.append((x_trial, f_trial))
            
            if calls >= budget:
                break
            
            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            
            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()
            
            # Update mean
            x_old = x.copy()
            x = sum(weights[i] * pop[i][0] for i in range(mu))
            x = np.clip(x, -5, 5)
            
            # Update step size (CSA)
            y = (x - x_old) / sigma if sigma > 1e-12 else np.zeros(n)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D * y))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e6)
            
            # Update covariance matrix
            yc = (x - x_old) / np.sqrt(sigma ** 2 + 1e-30)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * yc
            delta_heav = (1 + (1 - c_c) * np.sqrt((mueff - 1) / (n + 1))) * (np.linalg.norm(pc) ** 2 / n - 1)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                yi = (pop[i][0] - x_old) / np.sqrt(sigma ** 2 + 1e-30)
                rank_mu += weights[i] * np.outer(yi, yi)
            
            C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definite and update eigen decomposition
            C = (C + C.T) / 2
            try:
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-12))
                B = B2
                D = D2
            except np.linalg.LinAlgError:
                break
            
            # Check for stagnation
            if np.linalg.norm(x - x_old) < 1e-12 * max(1, np.linalg.norm(x)) and sigma < 1e-8:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Time guard: leave 5 seconds margin before 330s limit
        start_time = time.time()
        deadline = start_time + 325
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Strategy: Multiple restarts with different initializations
        # First, try a few random starts to find a good basin
        # Then run CMA-ES from the best found point
        
        # Phase 1: Random sampling to find promising regions
        n_random = min(50, budget // 5)
        candidates = []
        for i in range(n_random):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_made += 1
            candidates.append((x, fx))
        
        if not candidates:
            # Fallback: single point
            x = np.zeros(n)
            fx = func(x)
            return x, fx
        
        candidates.sort(key=lambda t: t[1])
        
        # Phase 2: Run CMA-ES from the best candidate
        x0 = candidates[0][0].copy()
        sigma0 = 1.0
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x0, candidates[0][1]
        
        # Run CMA-ES with most of the remaining budget
        cma_budget = remaining_budget
        x_best, f_best, cma_calls = self._cma_es(func, x0, sigma0, cma_budget, rng, deadline)
        calls_made += cma_calls
        
        if f_best < best_f:
            best_f = f_best
            best_x = x_best.copy()
        
        # Phase 3: If we have budget left, try another restart from a different region
        if calls_made < budget and time.time() < deadline:
            # Pick the second best candidate if it's sufficiently different
            if len(candidates) > 1:
                x0_2 = candidates[1][0].copy()
                # Check if it's in a different region
                if np.linalg.norm(x0_2 - x_best) > 1.0:
                    remaining = budget - calls_made
                    if remaining > 20:
                        x_best2, f_best2, cma_calls2 = self._cma_es(func, x0_2, 0.8, remaining, rng, deadline)
                        calls_made += cma_calls2
                        if f_best2 < best_f:
                            best_f = f_best2
                            best_x = x_best2.copy()
        
        if best_x is None:
            best_x = x0
            best_f = candidates[0][1]
        
        return best_x, best_f
# EVOLVE-BLOCK-END
