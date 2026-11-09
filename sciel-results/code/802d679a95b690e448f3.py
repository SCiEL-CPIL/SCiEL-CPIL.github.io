"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start local search with coarse grid initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280.0
        
        # 1. Global Phase: Coarse grid search to find the basin of attraction
        # Weierstrass f16 is multimodal. A coarse grid helps avoid getting stuck
        # in a far local minimum if the shift moves the global min away from origin.
        # Box is [-5, 5]^5.
        # Let's use a grid of 3 points per dimension initially? 3^5 = 243 points.
        # Budget is 500. This leaves ~250 for local refinement.
        # Alternatively, use random sampling for global phase.
        
        # Let's try a hybrid: 
        # Phase 1: 100 random samples to find a good region.
        # Phase 2: Local refinement around the best found point using a (1+1)-ES or Nelder-Mead.
        # Actually, the inspiration suggests "coarse global grid" or "multi-start population".
        
        # Let's implement a simple multi-start (1+1)-ES.
        # Divide budget: 50% for global exploration (multiple starts), 50% for local exploitation.
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Global Exploration
        # Use 100 function evaluations to sample the space.
        n_global = 100
        for i in range(n_global):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: Local Refinement
        # Use the remaining budget for a (1+1)-ES starting from best_x.
        # This mimics the "local refinements from diverse random starts" but focused on the best one.
        # To be more robust, we could do multiple local refinements, but budget is tight.
        # Let's stick to a strong local search from the best global point.
        
        x = best_x
        fx = best_f
        sigma = 1.0  # Start with a moderate step size
        
        remaining_budget = budget - n_global - 1 # -1 for the initial eval in loop? No, we track calls.
        # Actually, let's just loop until deadline or budget.
        
        # We have used n_global calls.
        calls_made = n_global
        
        # (1+1)-ES with 1/5 success rule
        # Success rate counter
        successes = 0
        total_steps = 0
        # We want to adapt sigma.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            # Clip to box
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                successes += 1
            else:
                pass # Keep x, fx
            
            total_steps += 1
            
            # Adapt sigma every 10 steps (or similar)
            # 1/5 rule: if success rate > 0.2, increase sigma; else decrease.
            if total_steps % 10 == 0:
                success_rate = successes / total_steps
                if success_rate > 0.2:
                    sigma *= 1.5
                else:
                    sigma *= 0.7
                # Reset counters for next period
                successes = 0
                total_steps = 0

        return x, fx
# EVOLVE-BLOCK-END
