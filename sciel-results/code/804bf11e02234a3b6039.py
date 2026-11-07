"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB-style Weierstrass optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, max_iter, deadline, rng, global_best_x, global_best_f):
        dim = self.dim
        n = dim
        m = max(15, int(4 + 3 * np.log(n)))  # Population size
        
        # CMA-ES parameters
        cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
        cmu = min(0.5, (2 + 2. / (n + 1.)) / ((n + 2.) + 2. + 2. / (n + 1.)))
        cs = (n + 1.5 + cmu) / (n + 9.)
        ds = 1. + cs + 2. * max(0., (cc - 1.) / (n + 1.)) - cmu
        chi_n = np.sqrt(n) * (1. - 1. / (4. * n) + 1. / (21. * n**2))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        hsig = (1. + 2. / (n + 1.)) * chi_n

        f_x = func(x)
        if f_x < global_best_f[0]:
            global_best_f[0] = f_x
            global_best_x[0] = x.copy()

        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate samples
            B = np.linalg.cholesky(C)
            y_samples = rng.standard_normal((m, n))
            x_samples = x + sigma * (y_samples @ B.T)
            
            # Clip to bounds
            x_samples = np.clip(x_samples, -5, 5)
            
            # Evaluate
            f_samples = np.array([func(xi) for xi in x_samples])
            
            # Update global best
            min_idx = np.argmin(f_samples)
            if f_samples[min_idx] < global_best_f[0]:
                global_best_f[0] = f_samples[min_idx]
                global_best_x[0] = x_samples[min_idx].copy()

            # Sort
            order = np.argsort(f_samples)
            y_sorted = y_samples[order]
            x_sorted = x_samples[order]
            f_sorted = f_samples[order]

            # Weights
            w = np.log(m / 2. + 0.5) - np.log(np.arange(1, m + 1))
            w = w / w.sum()
            mu_eff = 1. / np.sum(w[:min(2*m, m)]**2)
            
            # Mean update
            x_new = x + sigma * np.sum(w[:, None] * y_sorted, axis=0)
            
            # Step size control
            ps = (1. - cs) * ps + np.sqrt(cs * (2. - cs) * mu_eff) * (B @ (w @ y_sorted))
            hsig = (np.linalg.norm(ps) / np.sqrt(1. - (1. - cs)**((it + 1) * cs)) < (1.4 + 2. / (n + 1.)) * chi_n)
            
            # Covariance update
            pc = (1. - cc) * pc + cc * hsig * (x_new - x) / sigma
            
            C = (1. + ds + 2. * cmu * hsig * (1 - hsig) - cc) * C
            C += cmu * np.sum(w[:, None, None] * np.outer(y_sorted, y_sorted), axis=0)
            
            # Ensure positive definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C = np.eye(n)
                eigvals, eigvecs = np.linalg.eigh(C)
            
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1.))
            sigma = np.clip(sigma, 1e-12, 100.0)
            
            x = x_new
            x = np.clip(x, -5, 5)
            
            # Check convergence
            if np.linalg.norm(x - x_new) < 1e-12 * (1 + np.linalg.norm(x)):
                break

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        rng = np.random.default_rng(42)
        
        global_best_x = [np.array([0.0] * self.dim)]
        global_best_f = [np.inf]
        
        # Initial random points to find a good starting region
        n_initial = min(10, self.budget // 10)
        for i in range(n_initial):
            x_init = rng.uniform(-5, 5, self.dim)
            f_init = func(x_init)
            if f_init < global_best_f[0]:
                global_best_f[0] = f_init
                global_best_x[0] = x_init.copy()
        
        # Multi-start CMA-ES
        n_restarts = 3
        budget_per_restart = (self.budget - n_initial) // n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Start from best found so far or a new random point
            if r == 0:
                x0 = global_best_x[0].copy()
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4, 4, self.dim)
                f_x0 = func(x0)
                if f_x0 < global_best_f[0]:
                    global_best_f[0] = f_x0
                    global_best_x[0] = x0.copy()
                sigma0 = 1.0
            
            self._cma_es(func, x0, sigma0, budget_per_restart, deadline, rng, 
                         global_best_x, global_best_f)
        
        return global_best_x[0], global_best_f[0]
# EVOLVE-BLOCK-END
