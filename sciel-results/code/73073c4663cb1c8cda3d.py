import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Structured initial exploration
        # For Katsuura, multimodality is key. Use diverse starting points.
        initial_points = []
        
        # Center
        initial_points.append(np.zeros(dim))
        
        # Random points in full domain
        for _ in range(6):
            initial_points.append(rng.uniform(-5, 5, dim))
        
        # Points near the known minimum range [-4, 4]^5
        for _ in range(6):
            initial_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate initial points
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 10.0:
                break
            x0 = clip(x0)
            evaluate(x0)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float('inf')
            evaluate(best_x)

        # Phase 2: Multi-start (1+1)-ES portfolio
        # Use remaining budget for multiple (1+1)-ES attempts from diverse starts
        remaining_calls = budget - calls_made
        if remaining_calls > 20 and remaining_time() > 10.0:
            # Number of ES attempts based on remaining budget
            num_attempts = min(15, max(3, remaining_calls // 35))
            
            for i in range(num_attempts):
                if calls_made >= budget or remaining_time() < 5.0:
                    break
                
                # Choose starting point
                r = rng.random()
                if i == 0 or r < 0.3:
                    # Start near current best
                    x = clip(best_x + 1.5 * rng.standard_normal(dim))
                elif r < 0.6:
                    # Start from a random point in the domain
                    x = clip(rng.uniform(-5, 5, dim))
                else:
                    # Start from a random point near the expected minimum region
                    x = clip(rng.uniform(-4, 4, dim))
                
                fx = evaluate(x)
                sigma = 0.8
                no_improve = 0
                max_steps = 35
                
                for _ in range(max_steps):
                    if calls_made >= budget or remaining_time() < 2.0:
                        break
                    
                    y = clip(x + sigma * rng.standard_normal(dim))
                    fy = evaluate(y)
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.05
                        no_improve = 0
                    else:
                        sigma *= 0.92
                        no_improve += 1
                    
                    if no_improve >= 20:
                        break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Final fine-tuning from best point
        if calls_made < budget and remaining_time() > 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            
            while calls_made < budget and remaining_time() > 1.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                else:
                    sigma *= 0.93
                    if sigma < 1e-6:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
