"""Rewritten optimizer using a population-based CMA-ES style approach with adaptive step size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based optimizer with adaptive step size and restart mechanism."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial population size
        pop_size = max(4, int(4 + 3 * np.log(dim + 1)))
        
        # Best solution tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to evaluate
        def eval_point(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            return x, f

        # Phase 1: Initial random sampling to find a good starting point
        n_init = min(20, budget // 5)
        candidates = []
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            x, fx = eval_point(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        # Use the best few as seeds for local search
        seeds = [c[1] for c in candidates[:min(3, len(candidates))]]

        # Phase 2: Adaptive population-based local search
        # Use a simple CMA-like update with diagonal covariance (diagonal step sizes)
        # We maintain a population and update the mean based on the best half
        
        sigma = 1.0
        step_sizes = np.ones(dim) * sigma
        
        # Keep a small population
        current_pop = [s.copy() for s in seeds]
        current_fvals = [best_f] * len(current_pop)
        
        # Actually, let's do a proper (lambda, mu)-ES with adaptive sigma
        # lambda = pop_size, mu = pop_size // 2
        mu = max(1, pop_size // 2)
        
        # Initialize mean as the best point found
        mean = best_x.copy()
        
        # We'll use a simple diagonal CMA-ES variant
        # But to keep it simple and robust, let's use a (1+1)-ES with multiple restarts
        # or a (lambda+mu)-ES with fixed sigma adapted by success rate
        
        # Let's try a (lambda, mu)-ES with sigma adaptation
        # lambda = pop_size, mu = pop_size // 2
        # Adapt sigma based on number of improvements
        
        # Reset for the main loop
        sigma = 2.0
        mean = best_x.copy()
        
        # Main loop
        while calls_made < budget:
            # Generate population
            offspring = []
            for i in range(pop_size):
                # Use different step sizes for diversity
                s = sigma * (1.0 + 0.5 * rng.standard_normal())
                y = mean + s * rng.standard_normal(dim)
                y, fy = eval_point(y)
                offspring.append((fy, y))
                
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update mean using the best mu individuals
            if mu > 0:
                weights = np.ones(mu) / mu
                new_mean = np.zeros(dim)
                for i in range(min(mu, len(offspring))):
                    new_mean += weights[i] * offspring[i][1]
                mean = new_mean
            
            # Update sigma based on success rate
            # Count how many offspring are better than the previous mean's function value
            # We need to evaluate the mean or use the best offspring
            # Simple adaptation: if best offspring is better than current best, increase sigma
            # Otherwise decrease
            
            best_offspring_f = offspring[0][0]
            if best_offspring_f < best_f:
                sigma *= 1.2
                best_f = best_offspring_f
                best_x = offspring[0][1].copy()
            else:
                sigma *= 0.9
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # If we've made little progress, restart from the best point with larger sigma
            # Check if we've been stuck
            if calls_made > 100 and calls_made % 100 == 0:
                # Check progress in last 100 calls
                # If best_f hasn't improved much, try a larger step
                if sigma < 0.1:
                    sigma = 1.0
                    mean = best_x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
