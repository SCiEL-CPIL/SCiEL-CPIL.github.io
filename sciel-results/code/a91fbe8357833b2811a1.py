"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Try multiple random restarts if time permits
        best_x = None
        best_f = np.inf
        total_calls = 0
        calls_per_restart = 80  # calls per restart attempt
        
        while total_calls < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES with random start
            x0 = rng.uniform(-4.5, 4.5, self.dim)
            fx = func(x0)
            total_calls += 1
            if fx < best_f:
                best_f, best_x = fx, x0
            
            # CMA-ES parameters
            sigma = 1.0
            mu = max(1, self.dim // 2)
            lam = 2 * mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c1 = (2 + 2 / (self.dim + 1.3)) / ((self.dim + 1) ** 2 + 2)
            cmu = min(1 - c1, (2 * (mu_eff - 1 + 1 / mu_eff)) / ((self.dim + 2) ** 2 + 2))
            cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
            ds = 1 + cs + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1)
            chi = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
            
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            count_steps = 0
            budget_this_restart = min(calls_per_restart, self.budget - total_calls)
            
            for _ in range(budget_this_restart):
                if time.time() > deadline:
                    break
                
                # Generate offspring
                B = np.linalg.cholesky(C)
                ys = []
                xs = []
                for i in range(lam):
                    z = rng.standard_normal(self.dim)
                    y = B @ z
                    x_new = np.clip(x0 + sigma * y, -5, 5)
                    fy = func(x_new)
                    total_calls += 1
                    ys.append(y)
                    xs.append((x_new, fy))
                
                if time.time() > deadline:
                    break
                
                # Sort by fitness
                xs.sort(key=lambda t: t[1])
                y_sorted = [ys[i] for i in np.argsort([t[1] for t in xs])]
                
                # Update mean
                x_new_mean = sum(w * xs[i][0] for i, w in enumerate(weights))
                
                # Update evolution paths
                y_w = sum(w * y_sorted[i] for i, w in enumerate(weights))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi * y_w
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (count_steps + 1)) / chi) < 1.4 + 2 / (self.dim + 1)
                pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mu_eff) * y_w
                
                # Update covariance
                delta_hsig = (1 - hsig) * c1 * (2 - c1)
                C = ((1 + c1 * delta_hsig - c1 - cmu) * C + 
                     cmu / c1 * (pc @ pc.T - C) + 
                     sum(w * (y_sorted[i] @ y_sorted[i].T) for i, w in enumerate(weights)))
                
                # Symmetrize
                C = 0.5 * (C + C.T)
                
                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
                
                x0 = x_new_mean
                fx = xs[0][1]
                if fx < best_f:
                    best_f, best_x = fx, x0
                count_steps += 1
                
                # Check convergence
                if np.linalg.norm(C - np.trace(C) / self.dim * np.eye(self.dim)) < 1e-10:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
