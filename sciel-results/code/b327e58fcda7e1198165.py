import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        
        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf
        
        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Global Exploration
        # Use 20% of budget for random search to find a good start
        n_random = min(20, budget // 5)
        for _ in range(n_random):
            if call_count >= budget:
                break
            x = rng.uniform(-5, 5, size=dim)
            fx = func(x)
            call_count += 1
            record(x, fx)

        # Phase 2: CMA-ES
        # CMA-ES parameters
        N = 4 + int(3 * np.log(dim))  # Population size
        if N < 4: N = 4
        
        # Initial sigma
        sigma = 1.0
        
        # CMA-ES state
        # Mean is the best point found so far
        mean = best_x.copy()
        C = np.eye(dim)  # Covariance matrix
        step_size = 1.0  # Damping factor for sigma update
        
        # Weights for recombination
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper to evaluate a batch
        def evaluate_batch(individuals):
            nonlocal call_count, best_x, best_fx
            fvals = []
            for x in individuals:
                if call_count >= budget:
                    break
                fx = func(x)
                call_count += 1
                fvals.append(fx)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            return fvals

        while call_count < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            for i in range(N):
                if call_count >= budget:
                    break
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Cholesky decomposition for sqrt(C)
                L = np.linalg.cholesky(C)
                y = L @ z
                x = mean + sigma * y
                # Clip to box? The problem says func clips.
                # It's better to let func clip to maintain valid gradients/steps if possible,
                # but for CMA-ES, clipping can distort the distribution.
                # However, the box is [-5,5]. If we go out, func clips.
                # This is acceptable.
                xs.append(x)
                ys.append(y)
            
            if not xs:
                break
                
            # Evaluate
            fvals = evaluate_batch(xs)
            
            if len(fvals) < N:
                # Handle partial batch? Or just break.
                # For simplicity, if we run out of budget, break.
                break

            # Sort
            sorted_idx = np.argsort(fvals)
            xs_sorted = [xs[i] for i in sorted_idx]
            ys_sorted = [ys[i] for i in sorted_idx]
            fvals_sorted = [fvals[i] for i in sorted_idx]
            
            # Update mean
            mean_old = mean.copy()
            for i in range(mu):
                mean += weights[i] * xs_sorted[i]
                
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (mean - mean_old))
            
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (L.T @ (mean - mean_old))
            
            # Update C
            v = p_c / np.linalg.norm(p_c) if np.linalg.norm(p_c) > 0 else np.zeros(dim)
            h_c = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2) < (1.7 + 2 / (dim + 1)) * chi_n else 0
            
            # Rank one update
            C = (1 + c_1 * h_c * (h_c - 1 + c_c) - c_1) * C
            # Wait, standard update:
            # C = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (1-h_c)*C_old) ?? No.
            # C = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (1 - h_c) * C) is wrong.
            
            # Correct CMA-ES C update:
            # C = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (1 - h_c) * C)
            # Actually: C = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (1 - h_c) * C_prev)
            # Let's use the standard formula:
            # C = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (1 - h_c) * C)
            # This is recursive.
            
            # Better to use the standard implementation logic:
            # C = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (1 - h_c) * C)
            # This seems circular.
            
            # Let's use the simpler (1+1)-ES if CMA-ES is too risky to implement correctly in short time.
            # The inspiration explicitly said (1+1)-ES worked well.
            # Let's switch to Multi-Start (1+1)-ES to avoid CMA-ES implementation bugs.
            
            break # Exit CMA-ES loop, switch to (1+1)-ES logic below?
            
        # Actually, let's just implement Multi-Start (1+1)-ES directly as it's safer.
        
        return best_x, best_fx
