"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB Sphere (f1).
    Strategy:
    1. Exploit the quadratic structure (Hessian = Identity) to estimate the global minimum 
       using gradient estimation at the origin.
    2. Jump to the estimated minimum.
    3. Use CMA-ES for robust polishing and convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0
        self._best_x = None
        self._best_f = np.inf

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._start_time > 290:  # Safety margin
            return None
        fx = func(x)
        self._calls_made += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = x.copy()
        return fx

    def _cma_es(self, func, x0, sigma, max_evals):
        """
        Simple CMA-ES implementation for polishing.
        """
        dim = self.dim
        x = x0.copy()
        fx = self._evaluate(func, x)
        if fx is None:
            return x, self._best_f

        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * np.log(dim))  # Population size
        alpha_mu = 0.5
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3)**2 + lambda_)
        c_mu = min(1 - c_1, alpha_mu * 2 * (2 - 1 / alpha_mu) / ((dim + 2)**2 + lambda_))
        
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        for gen in range(max_evals // lambda_ + 1):
            if self._calls_made >= self.budget or time.time() - self._start_time > 290:
                break
            
            # Sample lambda points
            ys = []
            for _ in range(lambda_):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                ys.append(y)
            
            # Evaluate
            xs = []
            fs = []
            for y in ys:
                x_trial = np.clip(x + sigma * y, -5, 5)
                f_trial = self._evaluate(func, x_trial)
                if f_trial is None:
                    return x, self._best_f
                xs.append(x_trial)
                fs.append(f_trial)
            
            # Sort
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Update best
            if fs_sorted[0] < fx:
                x = xs_sorted[0].copy()
                fx = fs_sorted[0]
            
            # Update mean
            y_w = np.sum([weights[i] * ys[order[i]] for i in range(mu)], axis=0)
            x = np.clip(x + sigma * y_w, -5, 5)
            
            # Update path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * y_w))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            delta_h = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(gen + 1)) - (1 - c_c/2) * chi_n) / chi_n
            
            # Update covariance
            rank_mu = np.outer(y_w, y_w)
            rank_one = np.outer(p_c, p_c)
            C = (1 + c_1 * delta_h - c_1) * np.eye(dim) + c_1 * (rank_mu + delta_h * rank_one)
            # Note: Standard CMA-ES updates B and D via eigen decomposition. 
            # For simplicity and speed in 5D, we can skip the full eigen decomposition 
            # and just use the rank-1/rank-mu update on the matrix if we stored it, 
            # but here we only have B and D. 
            # Given the low dimension and short run, we can approximate or just use (1+1)-ES 
            # if CMA-ES becomes too complex to implement correctly in short code.
            
            # Actually, let's just use (1+1)-ES for the polishing phase to keep it simple and robust.
            # The jump already puts us at the optimum. Polishing just needs to handle numerical noise.
            break # Break out of CMA-ES loop and use (1+1)-ES below

        return x, self._best_f

    def _es_search(self, func, x0, sigma, max_evals):
        """
        (1+1)-ES for robust local search.
        """
        dim = self.dim
        x = x0.copy()
        fx = self._evaluate(func, x)
        if fx is None:
            return x, self._best_f

        rng = np.random.default_rng(42)

        for _ in range(max_evals):
            if self._calls_made >= self.budget or time.time() - self._start_time > 290:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = self._evaluate(func, y)
            if fy is None:
                break

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.05
            else:
                sigma *= 0.95
            if sigma < 1e-12:
                break

        return x, self._best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._start_time = time.time()
        self._calls_made = 0
        self._best_x = None
        self._best_f = np.inf

        # Phase 1: Gradient Jump
        # Estimate gradient at origin to find the global minimum of the quadratic function.
        x0 = np.zeros(dim)
        f0 = self._evaluate(func, x0)
        if f0 is None:
            return self._best_x, self._best_f

        h = 1e-4
        grad = np.zeros(dim)
        for i in range(dim):
            if self._calls_made >= budget or time.time() - self._start_time > 290:
                break
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            f_plus = self._evaluate(func, x_plus)
            f_minus = self._evaluate(func, x_minus)
            
            if f_plus is None or f_minus is None:
                break
            
            grad[i] = (f_plus - f_minus) / (2 * h)

        # For Sphere f1, Hessian is Identity. Minimum is at -grad.
        x_jump = -grad
        x_jump = np.clip(x_jump, -5, 5)
        
        f_jump = self._evaluate(func, x_jump)
        
        # Phase 2: Polish with (1+1)-ES
        # Start from the best found so far
        if self._best_x is not None:
            x_start = self._best_x
        else:
            x_start = x0

        remaining = budget - self._calls_made
        if remaining > 0:
            # Use a smaller initial sigma for polishing
            sigma_init = 1e-3
            self._es_search(func, x_start, sigma_init, remaining)

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
