"""Derivative-free optimizer: CMA-ES with adaptive restart and multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with adaptive restart for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        def counted_func(x):
            self._call_count += 1
            if self._call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            return float(func(x))

        best_x = np.zeros(self.dim)
        best_f = np.inf

        # Phase 1: Multi-start with CMA-ES
        rng = np.random.default_rng(123)
        
        # Generate diverse starting points
        n_starts = 5
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Random uniform points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        # Evaluate all starts
        start_results = []
        for s in starts:
            if self._call_count >= self.budget:
                break
            if time.time() > deadline - 20:
                break
            fs = counted_func(s)
            start_results.append((fs, s.copy()))
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        if not start_results:
            x0 = np.zeros(self.dim)
            f0 = counted_func(x0)
            return x0, f0

        start_results.sort(key=lambda x: x[0])

        # Phase 2: CMA-ES refinement from best starts
        # Allocate remaining budget among top starts
        remaining = self.budget - self._call_count
        if remaining > 0:
            # Use CMA-ES from the best 2 starts
            n_cma = min(2, len(start_results))
            budget_per_cma = max(50, remaining // n_cma)
            
            for i in range(n_cma):
                if self._call_count >= self.budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                fs, xs = start_results[i]
                
                # Run CMA-ES
                x_opt, f_opt = self._cma_es(
                    counted_func, xs, budget_per_cma,
                    deadline=deadline,
                    initial_sigma=1.0
                )
                
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        # Phase 3: Final local refinement around best_x if budget remains
        remaining = self.budget - self._call_count
        if remaining > 20 and best_x is not None:
            if time.time() < deadline - 5:
                x_opt, f_opt = self._cma_es(
                    counted_func, best_x, remaining,
                    deadline=deadline,
                    initial_sigma=0.1
                )
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        return best_x, best_f

    def _cma_es(self, func, x0, max_calls, deadline, initial_sigma=1.0):
        """CMA-ES implementation with adaptive restart."""
        n = self.dim
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        lambda_ = min(lambda_, max_calls - 1)
        if lambda_ < 2:
            lambda_ = 2
        
        # Strategy parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial sigma
        sigma = initial_sigma
        
        # Mean and covariance matrix
        m = x0.copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt_inv = np.eye(n)
        
        # Update lambda and mu
        def update_chol(C, C_inv, C_sqrt_inv):
            """Update Cholesky decomposition."""
            L = np.linalg.cholesky(C)
            C_inv = np.linalg.inv(L @ L.T)
            C_sqrt_inv = np.linalg.inv(L)
            return C_inv, C_sqrt_inv

        C_inv, C_sqrt_inv = update_chol(C, C_inv, C_sqrt_inv)
        
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Best found
        best_x = x0.copy()
        best_f = np.inf
        
        calls_used = 0
        
        # Initial population
        def generate_population():
            """Generate new population."""
            pops = []
            for i in range(lambda_):
                z = np.random.standard_normal(n)
                x = m + sigma * (C_sqrt_inv @ z)
                x = np.clip(x, -5, 5)
                pops.append(x)
            return pops

        # Main loop
        while calls_used < max_calls:
            if time.time() > deadline - 5:
                break
            
            # Generate and evaluate population
            pops = generate_population()
            fvals = []
            for x in pops:
                if calls_used >= max_calls or time.time() > deadline - 5:
                    break
                fx = func(x)
                fvals.append(fx)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if len(fvals) < lambda_:
                break
            
            # Sort by function value
            indices = np.argsort(fvals)
            pops = [pops[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pops[0].copy()
            
            # Update mean
            m_old = m.copy()
            m = np.sum([weights[i] * pops[i] for i in range(mu)], axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt_inv @ ((m - m_old) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((calls_used + 1) / (cs * n)))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (m - m_old) / sigma
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + cmu * np.sum([
                weights[i] * np.outer(pops[i] - m_old, pops[i] - m_old)
                for i in range(mu)
            ], axis=0) + (c1 + c1 * delta_hsig - cmu) * np.outer(pc, pc)
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                C_inv, C_sqrt_inv = update_chol(C, C_inv, C_sqrt_inv)
            except np.linalg.LinAlgError:
                # Regularize if not positive definite
                C = C + 1e-6 * np.eye(n)
                try:
                    C_inv, C_sqrt_inv = update_chol(C, C_inv, C_sqrt_inv)
                except np.linalg.LinAlgError:
                    break
            
            # Update sigma
            sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / np.sqrt((n + 1) / (n - 1)) - 1))
            
            # Adaptive restart: if sigma is too small or no progress
            if sigma < 1e-8:
                break
            
            # Check convergence
            if np.std(fvals[:mu]) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
