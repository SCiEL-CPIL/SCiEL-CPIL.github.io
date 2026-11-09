"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # Multi-start configuration
        # We split the budget into K independent CMA-ES runs.
        # K=5 runs with 100 evaluations each seems reasonable for dim=5.
        # We also do a quick random search phase to find good starting points.
        
        # Phase 1: Random search to find good starting points (10% of budget)
        n_random = max(10, budget // 10)
        candidates = []
        
        # Always include the origin and a few random points
        candidates.append(np.zeros(n))
        for _ in range(n_random - 1):
            x = rng.uniform(-5, 5, n)
            candidates.append(x)
        
        for x in candidates:
            if time.time() > deadline:
                break
            fx = func(x)
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        # Phase 2: Multi-start CMA-ES
        # Use the top 5 candidates as starting points for CMA-ES
        # We need to sort candidates by fx to pick the best ones.
        # Since we didn't store all fx, let's just use the global best and a few random ones.
        # Actually, let's re-evaluate a small set of points to get a sorted list.
        
        # Simpler: Just run CMA-ES from the global best found so far, 
        # and maybe 1-2 other random points if budget allows.
        
        # Let's use a standard CMA-ES implementation.
        # We will run 3 independent CMA-ES instances.
        # Each gets roughly (budget - n_random) // 3 evaluations.
        
        n_runs = 3
        budget_per_run = (budget - n_random) // n_runs
        
        # Starting points for the runs
        start_points = [global_best_x.copy() if global_best_x is not None else np.zeros(n)]
        for _ in range(n_runs - 1):
            start_points.append(rng.uniform(-5, 5, n))
        
        for i in range(n_runs):
            if time.time() > deadline:
                break
                
            self._run_cma_es(func, start_points[i], budget_per_run, rng, deadline, global_best_x, global_best_f)
            # Note: _run_cma_es updates global_best via nonlocal or return. 
            # Since we are in a method, we need to return the best found or update a shared state.
            # Let's refactor to return the best found in each run.
            
        # Actually, the above loop doesn't update global_best_f correctly inside the method call if we don't return it.
        # Let's rewrite the structure to be cleaner.
        
        return global_best_x, global_best_f

    def _run_cma_es(self, func, x0, budget, rng, deadline, best_x, best_f):
        """
        Runs a single CMA-ES instance.
        Returns (best_x, best_f) for this run.
        """
        n = self.dim
        x = x0.copy()
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # CMA-ES parameters
        sigma = 2.0  # Initial step size
        pop_size = max(4, int(4 + 3 * np.log(n)))  # Population size
        
        # Covariance matrix C
        C = np.eye(n)
        # Eigen decomposition cache (simplified: we'll just use random directions if C is identity initially, 
        # but proper CMA-ES needs to update C. 
        # For a "different strategy" that is robust, let's use a simplified CMA-ES 
        # or even just a population-based (mu+lambda)-ES with adaptive sigma and 
        # covariance matrix adaptation if we want to be fancy.
        # Given the constraints and need for simplicity/robustness, 
        # let's implement a basic (mu+lambda)-ES with adaptive sigma and 
        # a simple covariance adaptation (or just isotropic with adaptive sigma).
        # Actually, CMA-ES is the gold standard for this. Let's implement a minimal CMA-ES.
        
        # Minimal CMA-ES
        # State: x (mean), sigma, C (covariance), C^-1/2 (inverse square root)
        # We need to compute C^-1/2. For n=5, this is cheap.
        
        # Initialize
        mean = x0.copy()
        sigma = 2.0
        C = np.eye(n)
        
        # We need to sample from N(0, C). 
        # We can use Cholesky decomposition of C.
        
        calls_made = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Sample lambda vectors from N(0, I)
            z_pop = rng.standard_normal((pop_size, n))
            
            # Transform to y = B * z, where B = C^(1/2)
            # We can use np.linalg.cholesky(C)
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(n)
                B = np.eye(n)
            
            y_pop = z_pop @ B.T
            x_pop = mean + sigma * y_pop
            
            # Clip to bounds
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate population
            f_pop = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                fx_i = func(x_pop[i])
                f_pop.append(fx_i)
                calls_made += 1
                if fx_i < best_f:
                    best_f = fx_i
                    best_x = x_pop[i].copy()
            
            if len(f_pop) < pop_size:
                break
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            f_sorted = np.array(f_pop)[idx]
            x_sorted = x_pop[idx]
            
            # Update mean (using weighted average of top mu individuals)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # C = (1/nu_eff) * sum(w_i * (x_i - mean)(x_i - mean)^T) + (1 - c_1 - mu_eff*c_mu) * C
            # Simplified: Just use the sample covariance of the top mu
            diff = x_sorted[:mu] - new_mean
            C_new = np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            # Mix with old C for stability
            C = 0.9 * C + 0.1 * C_new
            
            # Update sigma (1/5 rule on success rate)
            # Count how many of the top mu are better than the current mean
            # This is a bit complex. Let's use a simple adaptive sigma based on 
            # the distance between the new mean and the old mean.
            # Or just use a fixed schedule? No, adaptive is better.
            # Let's use the standard CMA-ES sigma update, but it's complex.
            # For simplicity, let's just adjust sigma based on whether the best 
            # individual in the population is better than the previous best.
            if f_sorted[0] < best_f:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 5.0)
            
            mean = new_mean
        
        return best_x, best_f

# We need to fix the __call__ method to properly use the return value of _run_cma_es.

class Optimizer:
    """Multi-start CMA-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # Phase 1: Random search to find good starting points (10% of budget)
        n_random = max(10, budget // 10)
        
        # Always include the origin and a few random points
        start_candidates = [np.zeros(n)]
        for _ in range(n_random - 1):
            x = rng.uniform(-5, 5, n)
            start_candidates.append(x)
        
        for x in start_candidates:
            if time.time() > deadline:
                break
            fx = func(x)
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        # Phase 2: Multi-start CMA-ES
        # Use the global best and 2 other random points as starting points
        n_runs = 3
        remaining_budget = budget - n_random
        budget_per_run = remaining_budget // n_runs
        
        start_points = [global_best_x.copy() if global_best_x is not None else np.zeros(n)]
        for _ in range(n_runs - 1):
            start_points.append(rng.uniform(-5, 5, n))
        
        for i in range(n_runs):
            if time.time() > deadline:
                break
                
            best_x_run, best_f_run = self._run_cma_es(func, start_points[i], budget_per_run, rng, deadline)
            
            if best_f_run < global_best_f:
                global_best_f = best_f_run
                global_best_x = best_x_run.copy()
            
        return global_best_x, global_best_f

    def _run_cma_es(self, func, x0, budget, rng, deadline):
        """
        Runs a single CMA-ES instance.
        Returns (best_x, best_f) for this run.
        """
        n = self.dim
        x = x0.copy()
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # CMA-ES parameters
        sigma = 2.0  # Initial step size
        pop_size = max(4, int(4 + 3 * np.log(n)))  # Population size
        
        # Covariance matrix C
        C = np.eye(n)
        
        calls_made = 1 # We already evaluated x0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Sample lambda vectors from N(0, I)
            z_pop = rng.standard_normal((pop_size, n))
            
            # Transform to y = B * z, where B = C^(1/2)
            # We can use np.linalg.cholesky(C)
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(n)
                B = np.eye(n)
            
            y_pop = z_pop @ B.T
            x_pop = x + sigma * y_pop
            
            # Clip to bounds
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate population
            f_pop = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                fx_i = func(x_pop[i])
                f_pop.append(fx_i)
                calls_made += 1
                if fx_i < best_f:
                    best_f = fx_i
                    best_x = x_pop[i].copy()
            
            if len(f_pop) < pop_size:
                break
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            f_sorted = np.array(f_pop)[idx]
            x_sorted = x_pop[idx]
            
            # Update mean (using weighted average of top mu individuals)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # Simplified: Just use the sample covariance of the top mu
            diff = x_sorted[:mu] - new_mean
            C_new = np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            # Mix with old C for stability
            C = 0.9 * C + 0.1 * C_new
            
            # Update sigma (simple adaptive rule)
            if f_sorted[0] < best_f:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 5.0)
            
            x = new_mean
        
        return best_x, best_f
# EVOLVE-BLOCK-END
