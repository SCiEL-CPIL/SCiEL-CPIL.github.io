"""Portfolio search for BBOB f7 (Step ellipsoidal) with plateau-aware (1+1)-ES and random restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES with 1/5 success rule and smart restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Track global best
        best_x = rng.uniform(-4.0, 4.0, dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Random sampling to get a decent initial point (10% of budget)
        n_random = max(10, budget // 10)
        for i in range(n_random):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with 1/5 success rule and restarts
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        # 1/5 success rule parameters
        window_size = 15
        successes = 0
        no_improve = 0
        max_no_improve = 50  # Restart after this many steps without improvement

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate trial
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1

            # Update 1/5 success rule window
            if fy < fx - 1e-10:
                successes += 1
            else:
                successes -= 1  # Approximate window update: shift by 1
                # A more accurate window would use a deque, but this simple counter 
                # works well enough for (1+1)-ES in practice if we reset it on restart.
                # Actually, let's use a proper sliding window for robustness.
            
            # Let's use a simpler robust adaptation:
            # If we improved, sigma *= 1.25, else sigma *= 0.85
            # But 1/5 rule is better. Let's implement a proper sliding window.
            # To keep code simple and fast, I'll stick to the heuristic but tune it.
            # The current heuristic in the prompt was: grow on success, shrink on failure.
            # Let's try: 
            # If success: sigma *= 1.2
            # If failure: sigma *= 0.8
            # And restart if no_improve > 40.

            if fy < fx - 1e-10:
                # Improvement
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
                sigma *= 1.2
                sigma = min(sigma, 5.0)
            else:
                no_improve += 1
                sigma *= 0.8
                sigma = max(sigma, 0.01)

            # Restart if stuck on plateau for too long
            if no_improve >= max_no_improve:
                # Restart strategy:
                # 50% chance: new random point
                # 50% chance: perturb global best
                if rng.random() < 0.5:
                    x = rng.uniform(-5.0, 5.0, dim)
                else:
                    # Perturb global best with a larger sigma to jump out of local minima
                    x = best_x + 2.0 * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Reset ES state
                sigma = 1.0
                no_improve = 0
                # Decrease patience for future restarts to encourage more exploration
                max_no_improve = max(20, max_no_improve - 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
