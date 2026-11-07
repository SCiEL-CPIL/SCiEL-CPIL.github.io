"""CMA-ES implementation for black-box continuous optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with restarts for global convergence."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Pop size lambda
        lam = max(4, int(4 + 3 * np.log(dim + 1)))
        mu = lam // 2
        # Weights: log-linear weights
        weights = np.log(np.arange(mu, 0, -1) + 0.5)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # CMA-ES specific parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # Phase 1: Initial sampling to get a decent starting point
        # We use a fraction of the budget for initialization if it's large, otherwise less
        n_init = min(10, budget // 10)
        init_points = []
        for i in range(n_init):
            x = rng.uniform(-4, 4, dim) # Start slightly inside bounds
            x, fx = eval_point(x)
            init_points.append((fx, x))
            
        if not init_points:
            return best_x, best_f

        # Sort and pick the best as initial mean
        init_points.sort(key=lambda t: t[0])
        mean = init_points[0][1].copy()
        sigma = 2.0
        
        # Phase 2: CMA-ES Loop
        # We run CMA-ES until budget is exhausted or we stagnate, then restart
        while calls_made < budget:
            # Check if we should restart
            # Simple restart criterion: if sigma is very small or no improvement in last 'stagnation' generations
            # For simplicity, we will run one CMA-ES instance until budget is low or stagnation detected
            # Then restart from best_x with new sigma
            
            # Initialize CMA-ES state for this run
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(21*dim**2))
            
            # Run CMA-ES
            stagnation_counter = 0
            last_best_f = best_f
            
            while calls_made < budget:
                # Generate population
                offspring = []
                for i in range(lam):
                    # Cholesky factorization of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback to identity if singular
                        L = np.eye(dim)
                    
                    y = rng.standard_normal(dim)
                    x = mean + sigma * (L @ y)
                    x, fx = eval_point(x)
                    offspring.append((fx, x, y))
                    
                    if calls_made >= budget:
                        break
                
                if calls_made >= budget:
                    break
                
                # Sort offspring
                offspring.sort(key=lambda t: t[0])
                
                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i][1]
                mean = new_mean
                
                # Update paths
                old_mean = mean - sigma * (C_inv_sqrt @ (weights @ np.array([offspring[i][2] for i in range(mu)]).T)).T
                # Note: The vector of weighted y's
                weighted_y = np.zeros(dim)
                for i in range(mu):
                    weighted_y += weights[i] * offspring[i][2]
                    
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (mean - old_mean) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lam + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma
                
                # Update covariance
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu + delta_hsig * c_c * c_c) * C
                C += c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
                
                # Rank-mu update
                y_weighted = np.array([offspring[i][2] for i in range(mu)])
                C += c_mu * (weights @ (y_weighted.T * y_weighted))
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Regularize if necessary
                try:
                    C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Check stagnation
                if best_f < last_best_f - 1e-8:
                    stagnation_counter = 0
                    last_best_f = best_f
                else:
                    stagnation_counter += 1
                    
                # Restart condition
                if sigma < 1e-6 or stagnation_counter > 2 * lam:
                    break
            
            # If budget is still available, restart
            if calls_made < budget:
                # Restart from best point with a larger sigma
                mean = best_x.copy()
                sigma = 2.0
                # Optionally, we could try to infer a better sigma, but 2.0 is a safe bet for [-5,5]^5
                # To avoid immediate re-stagnation, we can randomize the covariance slightly or just reset it
                # Resetting C to identity is standard for restarts in some implementations, 
                # or we can keep C if we believe the local structure is still valid.
                # Given the hidden rotation, resetting C is safer to escape local basins if we got stuck.
                # But if we got stuck because we found the optimum, we don't want to waste budget.
                # The stagnation counter should handle the "found optimum" case by breaking the loop.
                # If we broke due to sigma < 1e-6, we are likely at the optimum.
                # If we broke due to stagnation_counter, we might be in a local minimum.
                # Let's restart with a fresh C.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
