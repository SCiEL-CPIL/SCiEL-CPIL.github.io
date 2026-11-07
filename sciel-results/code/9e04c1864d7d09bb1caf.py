import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0
        
        def wrapped_func(x):
            self.call_count += 1
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > self.deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -self.bound, self.bound)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = np.inf
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # CMA-ES implementation
        n = self.dim
        popsize = max(4, int(4 + 3 * np.log(n)))
        
        # Initial parameters
        x0 = np.random.uniform(-2, 2, size=n)
        sigma = 1.5  # Initial step size
        
        # CMA-ES state
        C = np.eye(n)  # Covariance matrix
        C_inv = np.eye(n)  # Inverse covariance
        C_sqrt_inv = np.eye(n)  # Inverse square root of C
        p_sigma = np.zeros(n)  # Evolution path for sigma
        p_c = np.zeros(n)  # Evolution path for C
        
        # Parameters
        mu = popsize // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial Cholesky decomposition
        L = np.linalg.cholesky(C)
        
        # Evaluate initial point
        wrapped_func(x0)
        
        # Main CMA-ES loop
        while self.call_count < self.budget and time.time() < self.deadline:
            # Generate population
            xs = []
            for i in range(popsize):
                if time.time() > self.deadline or self.call_count >= self.budget:
                    break
                z = np.random.randn(n)
                xi = x0 + sigma * (C_sqrt_inv @ z)
                xs.append(xi)
            
            if len(xs) == 0:
                break
            
            # Evaluate population
            fs = []
            for xi in xs:
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                fi = wrapped_func(xi)
                fs.append(fi)
            
            if len(fs) == 0:
                break
            
            # Sort by fitness
            indices = np.argsort(fs)
            xs = [xs[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update mean
            x_new = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update evolution paths
            delta = (x_new - x0) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / np.sqrt(mu_eff) < 1.4 + 2 / (n + 1))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta
            
            # Update covariance matrix
            delta_C = np.outer(p_c, p_c) - C
            delta_C *= c_c * (2 - c_c)
            delta_C *= h_sigma
            delta_C *= (1 - c_1 - c_mu)
            
            for i in range(mu):
                delta_C += c_mu * weights[i] * (np.outer(xs[i] - x0, xs[i] - x0) - sigma ** 2 * C)
            
            C = C + delta_C
            
            # Ensure positive definite
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break
            
            C_inv = np.linalg.inv(C)
            C_sqrt_inv = np.linalg.inv(L)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x0 = x_new
            
            # Adaptive stopping criterion
            # Check for stagnation
            if len(fs) >= mu:
                improvement = fs[0] - fs[-1]
                if improvement < 1e-10 * max(1, abs(fs[0])):
                    # No significant improvement, try to restart with larger sigma
                    sigma *= 1.5
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt_inv = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    L = np.eye(n)
        
        return self.best_x, self.best_f
