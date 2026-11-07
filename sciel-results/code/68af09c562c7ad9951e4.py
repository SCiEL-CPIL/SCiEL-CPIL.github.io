"""Multi-start (1+1)-ES optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES optimizer.
    
    Strategy:
    1. Perform a coarse random search to find promising regions.
    2. Run multiple (1+1)-ES instances from diverse starting points.
    3. Each (1+1)-ES uses adaptive step size based on success rate.
    4. Restart logic: if no improvement for a while, start a new (1+1)-ES 
       from a random point or a perturbed best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_1plus1_es(self, func, x0, sigma0, budget, rng, deadline, best_x, best_f):
        """
        Run a single (1+1)-ES instance.
        Returns the number of calls used.
        """
        dim = self.dim
        x = x0.copy()
        fx = func(x)
        calls = 1
        sigma = sigma0
        
        # Track success for step size adaptation
        # Simple adaptation: increase sigma on success, decrease on failure
        # More robust: use a learning rate
        alpha = 0.2
        
        # Stagnation counter
        no_improve = 0
        max_no_improve = 20  # Restart if stuck for 20 steps
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5.0, 5.0)
            fy = func(y)
            calls += 1
            
            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            # Update local best and step size
            if fy < fx:
                # Success
                x = y
                fx = fy
                no_improve = 0
                # Increase step size slightly
                sigma *= (1 + alpha)
            else:
                # Failure
                no_improve += 1
                # Decrease step size
                sigma *= (1 - alpha)
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Restart condition: if stuck, break and let caller handle restart
            if no_improve > max_no_improve:
                break
        
        return calls

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 15s margin
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Coarse Random Search
        # Use ~20% of budget to find good starting points
        coarse_budget = int(self.budget * 0.2)
        
        for _ in range(coarse_budget):
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Multi-start (1+1)-ES
        # Use remaining budget for local optimization
        remaining_budget = self.budget - total_calls
        
        if best_x is None:
            # Fallback if coarse search failed (unlikely)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            total_calls += 1
            remaining_budget -= 1
        
        # We will run multiple (1+1)-ES instances.
        # Each instance gets a portion of the remaining budget.
        # We restart if an instance stagnates.
        
        num_starts = 10  # Max number of (1+1)-ES starts
        calls_per_start = max(10, remaining_budget // num_starts)
        
        for i in range(num_starts):
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Choose starting point
            if i == 0:
                # Start from the best found in coarse search
                x0 = best_x.copy()
                sigma0 = 0.5
            else:
                # Start from a random point near the best or a new random point
                if rng.random() < 0.5:
                    # Perturb best
                    x0 = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5.0, 5.0)
                    sigma0 = 0.2
                else:
                    # New random point
                    x0 = rng.uniform(-5.0, 5.0, self.dim)
                    sigma0 = 1.0
            
            # Run (1+1)-ES
            used = self._run_1plus1_es(
                func, x0, sigma0, calls_per_start, rng, deadline, best_x, best_f
            )
            total_calls += used
            
            # If we used all the budget for this start, stop
            if total_calls >= self.budget:
                break
        
        # Phase 3: Final Polish
        # If there are calls left, do a very fine-grained (1+1)-ES around the best
        if total_calls < self.budget and best_x is not None:
            remaining = self.budget - total_calls
            x = best_x.copy()
            fx = best_f
            sigma = 0.01  # Very small step
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(self.dim)
                y = np.clip(x + sigma * z, -5.0, 5.0)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Slightly increase sigma if we're still improving
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                
                sigma = np.clip(sigma, 1e-8, 0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
