import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Lambda is typically 4 + floor(3 log n) for n>1, but for small n, 6 is a good start.
        # For dim=5, lambda=6 or 7 is common. Let's use 6.
        lam = 6
        mu = lam // 2  # 3 parents
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x
        best_f = np.inf
        
        calls_made = 0
        
        def evaluate(x_vec):
            nonlocal calls_made
            # Clip to box
            x_clip = np.clip(x_vec, -5, 5)
            calls_made += 1
            return func(x_clip)
        
        # Initial evaluation
        f = evaluate(x)
        if f < best_f:
            best_f = f
            best_x = x.copy()
            
        # Main loop
        while calls_made < budget:
            # Generate population
            # We need to generate enough candidates to not exceed budget
            # If remaining budget < lam, we just do (1+1) or stop?
            # Better: generate min(lam, budget - calls_made) candidates?
            # But CMA-ES needs a full population for recombination.
            # If we don't have enough for a full population, we can't do standard CMA-ES step.
            # Given 500 % 6 = 2, we will have 2 extra calls at the end.
            # We can use them for a final local polish or just ignore.
            
            num_to_gen = min(lam, budget - calls_made)
            if num_to_gen < 2:
                break
                
            # Sample
            z = rng.standard_normal((num_to_gen, dim))
            # Transform
            y = x + sigma * (z @ C_inv_sqrt.T)
            
            # Evaluate
            f_vals = []
            for i in range(num_to_gen):
                fi = evaluate(y[i])
                f_vals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = y[i].copy()
                    
            f_vals = np.array(f_vals)
            
            # Sort
            idx = np.argsort(f_vals)
            y_sorted = y[idx]
            f_sorted = f_vals[idx]
            
            # Update mean
            # Only use top mu if we have at least mu candidates
            if num_to_gen >= mu:
                w = weights[:mu]
                # Note: if num_to_gen < mu, we should renormalize weights or just use what we have?
                # Standard practice: if population is reduced, renormalize weights.
                if num_to_gen < mu:
                    w = weights[:num_to_gen]
                    w = w / np.sum(w)
                    mu_eff_local = 1 / np.sum(w**2)
                else:
                    mu_eff_local = mu_eff
                    
                x_new = np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)
                
                # Update covariance
                y_diff = y_sorted[:mu] - x
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (dim + 1) * c_sigma * (1 - c_sigma) * np.outer(p_sigma, p_sigma) - C) + c_mu * np.sum(w[:, np.newaxis] * np.outer(y_diff, y_diff), axis=0)
                
                # Update path
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff_local) * (C_inv_sqrt @ x_new) / sigma
                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**((calls_made // lam) + 1)) < (1.4 + 2 / (dim + 1)) * np.trace(C_inv_sqrt) / dim) else 0
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff_local) * (C_inv_sqrt @ (x_new - x)) / sigma
                
                # Update inverse square root
                # This is expensive. For dim=5, we can do it.
                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small eigenvalues to avoid numerical issues
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**((calls_made // lam) + 1)) - 1))
                
                x = x_new
            else:
                # Fallback to simple mean update if population too small
                x = np.mean(y_sorted, axis=0)
                # Don't update C, p_c, p_sigma to avoid instability with tiny populations
                # Just keep previous state, maybe reduce sigma slightly?
                sigma *= 0.9

        return best_x, best_f
