"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start (1+1)-ES optimizer for rotated Rastrigin.
    
    Strategy:
    1. Diversified exploration using multiple (1+1)-ES runs from different starts.
    2. Intensive exploitation of the best found point using a fine-grained (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin
        
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Multi-Start Exploration ---
        
        # Generate diverse starting points
        # 1. Origin (likely close to global min for many rotations)
        # 2. Random points in the box
        # 3. Deterministic points (corners/midpoints) to ensure coverage
        starts = [np.zeros(dim)]
        
        # Add random starts
        n_random = 8
        for _ in range(n_random):
            starts.append(rng.uniform(-4.0, 4.0, dim))
            
        # Add a few deterministic diverse points
        starts.append(np.full(dim, 3.0))
        starts.append(np.full(dim, -3.0))
        
        # Sort starts? Origin is usually good. Let's prioritize it.
        # We will run a short (1+1)-ES from each start.
        
        # Budget allocation:
        # Total budget: 500
        # Number of starts: ~11
        # Reserve ~150 calls for final refinement.
        # Remaining: 350 calls for 11 starts -> ~30 calls each.
        
        calls_per_run = 30
        refinement_budget = budget - (len(starts) * calls_per_run)
        
        # Ensure refinement budget is at least 100
        if refinement_budget < 100:
            refinement_budget = 100
            calls_per_run = max(10, (budget - refinement_budget) // len(starts))

        for start_point in starts:
            if calls >= budget or time.time() > deadline:
                break
            
            # (1+1)-ES
            x = start_point.copy()
            fx = eval_func(x)
            
            # If this start is already better than global best, update
            # (eval_func handles this)
            
            sigma = 1.5  # Start with larger step for exploration
            window = 10
            recent = [False] * window
            idx = 0
            
            for _ in range(calls_per_run):
                if calls >= budget or time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1
                
                # Adapt sigma
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.15
                    elif sr < 0.2:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.05, 5.0)

        # --- Phase 2: Intensive Refinement ---
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Start with a smaller sigma for fine-tuning
            sigma = 0.2
            window = 10
            recent = [False] * window
            idx = 0
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1
                
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.1
                    elif sr < 0.2:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
