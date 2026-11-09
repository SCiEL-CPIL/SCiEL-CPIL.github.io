import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.timeout = 290.0  # Safety margin from 300s limit

    def _evaluate(self, x, func):
        if time.time() - self.start_time > self.timeout:
            raise TimeoutError("Time limit exceeded")
        
        if self.calls >= self.budget:
            raise ValueError("Budget exceeded")
            
        self.calls += 1
        x = np.clip(x, -5.0, 5.0)
        f = func(x)
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            
        return f

    def _cma_es(self, x0, sigma0, func, max_evals=100):
        """
        CMA-ES implementation with budget tracking.
        """
        n = self.dim
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Initialize
        mean = x0.copy().astype(float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Initial population
        pop_size = 2 * mu
        best_local_f = np.inf
        best_local_x = mean.copy()
        
        evals_used = 0
        max_iters = max(1, max_evals // pop_size)
        
        for it in range(max_iters):
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                break
            
            # Sample population
            B = np.linalg.cholesky(C)
            z = rng.standard_normal((pop_size, n))
            pop = np.array([mean + sigma * (z[i] @ B.T) for i in range(pop_size)])
            pop = np.clip(pop, -5.0, 5.0)
            
            # Evaluate
            fvals = np.zeros(pop_size)
            for i in range(pop_size):
                try:
                    fvals[i] = self._evaluate(pop[i], func)
                    evals_used += 1
                except (TimeoutError, ValueError):
                    return mean, sigma
            
            # Sort
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_local_f:
                best_local_f = fvals[0]
                best_local_x = pop[0].copy()
            
            # Update mean
            mean_new = mean + sigma * (w[:mu] @ pop[:mu, :].T)
            
            # Update covariance matrix
            y = (mean_new - mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) / chi_n) < (1.4 + 2 / (n + 1))
            pc = (1 - c_c) * pc + hsig * c_c * y
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + \
                c_1 * (np.outer(pc, pc) + delta_hsig) + \
                c_mu * (w[:mu] * pop[:mu, :]).T @ (w[:mu] * pop[:mu, :])
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            mean = mean_new
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except:
                C = C + 1e-8 * np.eye(n)
            
            # Convergence check
            if sigma < 1e-10:
                break
            if np.max(fvals[:mu]) - np.min(fvals[:mu]) < 1e-10:
                break
        
        return mean, sigma

    def _nelder_mead(self, x0, func, max_iters=50):
        """
        Nelder-Mead for local refinement.
        """
        n = self.dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] = 0.1
        
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            try:
                fvals[i] = self._evaluate(simplex[i], func)
            except (TimeoutError, ValueError):
                return

        for _ in range(max_iters):
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                return

            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            x_ref = centroid + (centroid - worst)
            try:
                f_ref = self._evaluate(x_ref, func)
            except (TimeoutError, ValueError):
                return

            if f_ref < fvals[0]:
                x_exp = centroid + 2 * (centroid - worst)
                try:
                    f_exp = self._evaluate(x_exp, func)
                except (TimeoutError, ValueError):
                    return
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                x_con = centroid + 0.5 * (worst - centroid)
                try:
                    f_con = self._evaluate(x_con, func)
                except (TimeoutError, ValueError):
                    return
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        try:
                            fvals[i] = self._evaluate(simplex[i], func)
                        except (TimeoutError, ValueError):
                            return
                    break

            if np.std(fvals) < 1e-8:
                break

    def __call__(self, func):
        n = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Global exploration with CMA-ES restarts
        n_restarts = 10
        budget_per_restart = max(20, self.budget // (n_restarts + 2))
        
        for restart in range(n_restarts):
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                break
            
            # Initialize around current best or random
            if restart == 0:
                x0 = np.zeros(n)
                sigma0 = 2.0
            else:
                # Mix of random and perturbed best
                if restart % 2 == 0:
                    x0 = self.best_x + rng.standard_normal(n) * 0.5
                else:
                    x0 = rng.uniform(-4, 4, n)
                x0 = np.clip(x0, -5.0, 5.0)
                sigma0 = max(0.1, 1.5 - 0.1 * restart)
            
            # Run CMA-ES
            self._cma_es(x0, sigma0, func, max_evals=budget_per_restart)
        
        # Phase 2: Local refinement with Nelder-Mead
        if self.calls < self.budget:
            remaining = self.budget - self.calls
            if remaining > 30:
                self._nelder_mead(self.best_x, func, max_iters=min(50, remaining // 10))
        
        return self.best_x, self.best_f
