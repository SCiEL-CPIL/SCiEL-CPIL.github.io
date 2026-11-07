import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        
        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Multi-start (1+1)-ES with momentum
        # Lunacek bi-Rastrigin is multimodal. We need to explore widely first,
        # then refine.
        
        # Strategy:
        # 1. Initial random sampling to find a decent basin.
        # 2. Run (1+1)-ES with momentum from the best found so far.
        # 3. If stagnation, restart from a new random point or best point with larger step.
        
        # Phase 1: Random Sampling (10% of budget)
        n_samples = min(50, budget // 10)
        for _ in range(n_samples):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        # Phase 2: (1+1)-ES with momentum
        # We will run this loop until budget or time is exhausted.
        # We implement a simple restart strategy: if no improvement for N steps, restart.
        
        # Parameters for (1+1)-ES
        # Step size sigma
        # Momentum
        # We'll use a fixed number of restarts or a time-based check.
        
        # Let's try a few restarts, each with a fresh momentum and potentially reset step size.
        # Given the landscape, starting from the best found in Phase 1 is good.
        
        # We will alternate between "exploration" (larger sigma, random restarts) and "exploitation" (smaller sigma, local search).
        
        # Simpler approach: Run (1+1)-ES from best_x. If it stagnates, restart from a random point near best_x or a new random point.
        
        # Let's define a helper for one run of (1+1)-ES
        def run_es(start_x, sigma, max_steps):
            nonlocal best_x, best_f, calls
            x = start_x.copy()
            f = best_f if np.allclose(x, best_x) else evaluate(x)
            # If we just evaluated x, f is correct. If x was best_x, we didn't evaluate, so f is best_f.
            # Actually, if start_x is best_x, we already know f.
            # If start_x is new, we need to evaluate.
            if not np.allclose(x, best_x):
                 f = evaluate(x)
            
            momentum = np.zeros(dim)
            # Adaptive step size? Or fixed? 
            # Let's use a simple adaptive step size based on success rate.
            # Or just fixed sigma for simplicity first, maybe decreasing.
            
            # For Lunacek, the local minima are tricky.
            # Let's use a standard (1+1)-ES with step size adaptation (1/5 rule or similar).
            
            sigma_cur = sigma
            success_count = 0
            steps_since_improve = 0
            
            for _ in range(max_steps):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Generate candidate
                z = rng.standard_normal(dim)
                # Apply momentum
                # momentum = 0.9 * momentum + 0.1 * z ? Or just use z?
                # Momentum helps escape local minima if the direction is good.
                # Let's try: direction = momentum + z
                # Then normalize? Or just add?
                # Standard (1+1)-ES doesn't usually use momentum.
                # But "Momentum-based (1+1)-ES" was mentioned in inspirations.
                # Let's try: x_new = x + sigma * (momentum + z)
                # And update momentum: momentum = momentum * decay + z * (1-decay)?
                # Or momentum = z? (No momentum)
                # Let's stick to standard (1+1)-ES first, it's robust.
                
                x_new = x + sigma_cur * z
                
                f_new = evaluate(x_new)
                
                if f_new < f:
                    # Success
                    x = x_new
                    f = f_new
                    success_count += 1
                    steps_since_improve = 0
                    # Increase sigma slightly? Or keep same?
                    # 1/5 rule: if success rate > 0.35, increase sigma.
                    # For simplicity, let's just keep sigma constant or decrease slowly.
                    # Decreasing sigma helps convergence.
                    sigma_cur *= 0.99
                else:
                    steps_since_improve += 1
                    # Decrease sigma
                    sigma_cur *= 0.95
                    
                # Check for stagnation
                if steps_since_improve > 50:
                    break
                    
            return x, f, sigma_cur

        # We will run multiple ES instances.
        # First from the best found.
        # Then from random points if time/budget permits.
        
        # Estimate steps per ES run.
        # Total budget left.
        # Let's divide remaining budget into chunks.
        
        # Restart 1: From best_x
        if calls[0] < budget and time.time() < deadline:
            # Use 50% of remaining budget for first ES run
            remaining = budget - calls[0]
            max_steps = max(10, remaining // 2)
            x, f, sigma = run_es(best_x, 1.0, max_steps)
            
        # Restart 2: From a random point near best_x
        if calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            if remaining > 10:
                # Perturb best_x
                x_start = best_x + rng.uniform(-1, 1, dim)
                x_start = np.clip(x_start, -5, 5)
                max_steps = max(10, remaining // 2)
                x, f, sigma = run_es(x_start, 0.5, max_steps)

        # Restart 3: From another random point
        if calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            if remaining > 10:
                x_start = rng.uniform(-5, 5, dim)
                max_steps = max(10, remaining)
                x, f, sigma = run_es(x_start, 1.0, max_steps)

        # Final fallback: If we haven't made any calls (shouldn't happen), evaluate origin
        if calls[0] == 0:
            evaluate(np.zeros(dim))

        return best_x, best_f
