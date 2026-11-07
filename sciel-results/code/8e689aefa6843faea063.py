# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        dim = self.dim
        budget = self.budget

        # CMA-ES implementation
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        
        # Update rates
        mu = lambda_ // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evaluate initial point
        f_mean = count_eval(x_mean)
        
        # Main CMA-ES loop
        while self._call_count < budget - 2 * lambda_ and time.time() < deadline:
            # Generate population
            z = rng.standard_normal((lambda_, dim))
            # Apply transformation using Cholesky of C
            L = np.linalg.cholesky(C)
            population = x_mean + sigma * (z @ L.T)
            
            # Evaluate population
            f_values = np.array([count_eval(x) for x in population])
            
            # Sort by fitness
            indices = np.argsort(f_values)
            y_sorted = z[indices]
            x_sorted = population[indices]
            f_sorted = f_values[indices]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * x_sorted, axis=0)
            
            # Check for convergence or if we've made progress
            if np.all(np.abs(x_new - x_mean) < 1e-14):
                break
            
            # Update step-size control (CSA)
            h_sigma = (np.sqrt((1 - np.exp(-2 / c_c))) + 
                      2 * (1 - (dim + 1) / (dim + 3))) * np.sqrt((mu_eff - 1) / (dim + 1))
            norm_p_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - np.exp(-2 * self._call_count / (dim * c_sigma)))
            
            p_sigma = (1 - c_sigma) * p_sigma + (
                c_sigma / np.sqrt(c_sigma * (2 - c_sigma)) * h_sigma * 
                (np.linalg.norm(x_new - x_mean) / sigma - np.sqrt(mu_eff / 2) * np.sqrt(1 - np.exp(-2 / c_c)))
            )
            
            # Update evolution path
            h_c = (np.sqrt((1 - np.exp(-2 / c_c))) + 
                  2 * (1 - (dim + 1) / (dim + 3))) * np.sqrt((mu_eff - 1) / (dim + 1))
            p_c = (1 - c_c) * p_c + c_c / np.sqrt(c_c * (2 - c_c)) * (x_new - x_mean) / sigma
            
            # Update covariance matrix
            delta_h = (1 - h_c) if np.linalg.norm(p_c) / np.sqrt(1 - np.exp(-2 * self._call_count / c_c)) <= (1 + 5 / c_c) * h_c else 1
            
            outer = np.outer(p_c, p_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * outer + c_mu * np.sum(
                weights[:, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0
            )
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            eigenvalues = np.linalg.eigvalsh(C)
            if np.min(eigenvalues) < 1e-10:
                C += 1e-10 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-15, 100.0)
            
            # Update mean
            x_mean = x_new
            
            # Check convergence
            if np.linalg.norm(sigma * np.sqrt(np.diag(C))) < 1e-15:
                break
        
        # Final polish with a few random perturbations around the best
        if self._best_x is not None and self._call_count < budget - 5 and time.time() < deadline:
            for _ in range(min(10, budget - self._call_count - 2)):
                if self._call_count >= budget - 2 or time.time() > deadline:
                    break
                perturbation = rng.standard_normal(dim) * 0.1 * sigma
                x_candidate = np.clip(self._best_x + perturbation, -5, 5)
                count_eval(x_candidate)
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
