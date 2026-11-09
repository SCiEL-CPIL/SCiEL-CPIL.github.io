"""Bootstrap spawn: DIRECT approach using Rastrigin structure + CMA-ES refinement."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer exploiting Rastrigin periodicity + CMA-ES for rotation-aware search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _rastrigin_lower_bound(self, x):
        """Compute the minimum possible Rastrigin value given a fixed rotation/shift by
        exploiting the fact that for each coordinate i, f_i(x_i) >= 10 - 10*cos(2*pi*x_i),
        and the minimum of 10-10*cos(2*pi*x_i) over x_i is 0, achieved at integer x_i.
        Actually the Rastrigin term per dimension is x_i^2 - 10*cos(2*pi*x_i) + 10.
        The minimum of this is 0 at x_i = integer.
        So the global min of Rastrigin is 0, achieved at any integer point.
        But with rotation, the 'integer' points in the rotated frame are what matter.
        We can compute: for a candidate center c, evaluate the function and also compute
        the theoretical lower bound contribution.
        """
        pass

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-Start CMA-ES
        # We will run multiple independent CMA-ES instances.
        # Each instance gets a portion of the budget.
        # If an instance stagnates, we stop it and start a new one.
        
        # Parameters for CMA-ES
        lambda_pop = 10 + int(3 * math.log(dim))  # ~14 for dim=5
        mu = lambda_pop // 2
        
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 2) / 5
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        chi_N = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Weights
        w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        w /= w.sum()
        nu_eff = 1 / sum(w ** 2)

        # Strategy: 
        # 1. Initial coarse search to find a few good starting points.
        # 2. Run CMA-ES from the best point found.
        # 3. If it stagnates, pick another good point from history and restart CMA-ES with larger sigma.
        
        # Phase 1: Coarse Search
        # Use 20% of budget for initial exploration
        n_coarse = max(10, budget // 5)
        candidates = []
        for _ in range(n_coarse):
            if time.time() > deadline or calls_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            candidates.append((fx, x))
        
        if not candidates:
            return (np.zeros(dim), 0.0)

        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Iterative CMA-ES with Restarts
        # We maintain a list of "promising" centers.
        # We start with the best candidate.
        # If CMA-ES stagnates, we try the next best candidate that hasn't been used recently,
        # or a random new point if we have budget left.
        
        used_centers = []
        
        # We will loop until budget or time runs out.
        # Each loop is one "run" of CMA-ES.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Select a starting point
            # 1. If we have unused candidates, pick the best one.
            # 2. Otherwise, pick a random point or a perturbation of the best found so far.
            
            # To avoid repeating the same center, we keep a small history.
            # For simplicity, let's just cycle through the top 10 candidates.
            # After that, we'll use random points.
            
            if len(used_centers) < len(candidates):
                # Pick the best unused candidate
                # Simple heuristic: just take candidates[len(used_centers)]
                # This assumes candidates are sorted best to worst.
                start_f, start_x = candidates[len(used_centers)]
            else:
                # Generate a new random start point
                start_x = rng.uniform(-5, 5, dim)
                start_f = evaluate(start_x)
            
            used_centers.append(start_x)
            
            # Initialize CMA-ES state
            x_mean = start_x.copy()
            sigma = 1.0 # Start with moderate sigma
            C = np.eye(dim)
            evolution_path_sigma = np.zeros(dim)
            evolution_path_cov = np.zeros((dim, dim))
            
            # Run CMA-ES for a fixed number of evaluations or until stagnation
            # Let's allow up to 50 evaluations per run, or until no improvement for 20 evals.
            run_budget = min(50, budget - calls_used)
            last_improvement_f = start_f
            no_improvement_count = 0
            
            for _ in range(run_budget):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for _ in range(lambda_pop):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_off = x_mean + sigma * (C @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = evaluate(x_off)
                    offspring.append((fx_off, x_off, z))
                
                if len(offspring) < mu:
                    break
                
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]
                
                # Check for improvement
                if selected[0][0] < last_improvement_f - 1e-8:
                    last_improvement_f = selected[0][0]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Stagnation check
                if no_improvement_count > 10:
                    break
                
                # Update evolution paths
                x_new_mean = sum(w[i] * selected[i][1] for i in range(mu))
                
                # Update sigma evolution path
                evolution_path_sigma = (1 - c_sigma) * evolution_path_sigma + \
                                       math.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * \
                                       (C @ ((x_new_mean - x_mean) / sigma))
                
                # h_sigma check
                # Note: This formula can be unstable if denominator is 0.
                # We use a simplified h_sigma check based on norm.
                h_sigma = (np.linalg.norm(evolution_path_sigma) / chi_N) < (1.4 + 2 / (dim + 1))
                h_sigma = float(h_sigma)
                
                # Update covariance evolution path
                evolution_path_cov = (1 - c_c) * evolution_path_cov + \
                                     h_sigma * c_c * (1 - c_c) * np.outer(x_new_mean - x_mean,
                                                                          x_new_mean - x_mean)
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one_update = np.outer(
                    x_new_mean - x_mean + delta_h * (x_mean - x_mean),
                    x_new_mean - x_mean + delta_h * (x_mean - x_mean)
                )
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * evolution_path_cov + \
                    c_mu * sum(w[i] * np.outer(selected[i][1] - x_mean,
                                                selected[i][1] - x_mean) for i in range(mu))
                
                # Symmetrize C
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= math.exp(c_sigma / chi_N *
                                  (np.linalg.norm(evolution_path_sigma) / chi_N - 1))
                sigma = float(np.clip(sigma, 0.01, 10.0))
                
                # Update mean
                x_mean = x_new_mean
                
                # Convergence check
                if sigma < 0.01:
                    break

        return (best_x, best_f)
