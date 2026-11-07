"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))  # Standard CMA-ES pop size
        
        # Initialize center point
        # Start near center to quickly find the basin of attraction
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # Initialize step size
        sigma = 2.0
        
        # Initialize covariance matrix (identity)
        C = np.eye(n)
        
        # Eigen decomposition cache
        D_inv = np.ones(n)
        B = np.eye(n)
        
        # Weights for recombination
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # State variables
        ps = np.zeros(n)
        pc = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        calls_made = 1
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Update eigen decomposition of C
            # In a full implementation, we'd update C and decompose every generation.
            # For simplicity and speed in this constrained environment, we approximate
            # or perform the full update if budget allows.
            # Given n=5, full update is cheap.
            
            # Generate population
            ys = []
            xs = []
            fxs = []
            
            for i in range(pop_size):
                # Standard normal in transformed space
                z = rng.standard_normal(n)
                # Transform using B and D_inv (approximation of C^-0.5)
                # Note: We need to update B and D_inv from C periodically.
                # For this implementation, we will update C and decompose every generation.
                
                # Actually, to be robust and correct, let's do the standard CMA-ES update.
                # Since n=5 is small, we can afford the full update.
                
                # First, we need to generate the points. 
                # We need the current B and D_inv. 
                # Let's initialize/update them before the loop.
                
                pass # Placeholder, logic moved below
            
            # Re-structure to update C and decompose properly at the start of the loop
            # or at the end. Let's do it at the start of the iteration if not done.
            
            # To keep the code simple and avoid state sync issues, we will:
            # 1. Decompose C at the start of the generation.
            # 2. Generate points.
            # 3. Evaluate.
            # 4. Update C, ps, pc, sigma.
            
            # Decompose C
            try:
                D_inv, B = np.linalg.eigh(C)
                D_inv = 1.0 / np.sqrt(D_inv + 1e-16)
            except np.linalg.LinAlgError:
                D_inv = np.ones(n)
                B = np.eye(n)
            
            ys = []
            xs = []
            fxs = []
            
            for i in range(pop_size):
                z = rng.standard_normal(n)
                y = B @ (D_inv * z)
                x_new = np.clip(x + sigma * y, -5, 5)
                
                # Check budget before call
                if calls_made >= self.budget:
                    break
                    
                f_new = func(x_new)
                calls_made += 1
                ys.append(y)
                xs.append(x_new)
                fxs.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            if len(xs) == 0:
                break
                
            # Sort by fitness
            order = np.argsort(fxs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fxs = [fxs[i] for i in order]
            
            # Take top mu
            mu_actual = min(mu, len(ys))
            if mu_actual == 0:
                break
            
            y_weighted = np.sum([weights[i] * ys[i] for i in range(mu_actual)], axis=0)
            x_weighted = np.sum([weights[i] * xs[i] for i in range(mu_actual)], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / (sigma) * (B.T @ y_weighted)
            
            # Update evolution path for covariance
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // pop_size)))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * y_weighted
            
            # Update center
            x = x_weighted
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) - (1 - cs) * (1 - 2 * cs) * C) \
                + cmu * np.sum([weights[i] * np.outer(ys[i], ys[i]) for i in range(mu_actual)], axis=0)
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Check time
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
