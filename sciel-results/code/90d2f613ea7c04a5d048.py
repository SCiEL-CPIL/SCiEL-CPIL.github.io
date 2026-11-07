"""Combined optimizer: Global grid initialization + Multi-start Adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 320s to be safe.
        deadline = start_time + 320.0
        
        n = self.dim
        total_budget = self.budget
        
        calls_made = 0
        best_x = None
        best_fx = np.inf
        
        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= total_budget or time.time() > deadline - 1.0:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Global Deterministic & Random Sampling ---
        points_to_eval = []
        
        # 1. Center
        points_to_eval.append(np.zeros(n))
        
        # 2. Vertices of the box [-5, 5]^n
        for i in range(2**n):
            vertex = []
            for j in range(n):
                if (i >> j) & 1:
                    vertex.append(5.0)
                else:
                    vertex.append(-5.0)
            points_to_eval.append(np.array(vertex))
            
        # 3. Axis midpoints
        for i in range(n):
            for sign in [-1, 1]:
                mid = np.zeros(n)
                mid[i] = sign * 2.5
                points_to_eval.append(mid)
                
        # 4. Random points for diversity
        rng = np.random.default_rng(12345)
        num_random = 50
        for _ in range(num_random):
            points_to_eval.append(rng.uniform(-5.0, 5.0, n))

        # Evaluate all points in Phase 1
        for p in points_to_eval:
            if calls_made >= total_budget or time.time() > deadline - 10.0:
                break
            evaluate(p)

        if best_x is None:
            best_x = np.zeros(n)
            best_fx = evaluate(best_x)

        # --- Phase 2: Multi-start Adaptive (1+1)-ES ---
        remaining_budget = total_budget - calls_made
        if remaining_budget <= 20:
            return best_x, best_fx

        # Select number of restarts
        # Each restart needs at least ~50 calls to be useful
        num_restarts = min(5, remaining_budget // 50)
        if num_restarts < 1:
            num_restarts = 1
            
        budget_per_restart = remaining_budget // num_restarts
        
        # Restart points: Best found + Random
        restart_points = []
        restart_points.append(best_x.copy())
        
        rng_restart = np.random.default_rng(54321)
        for _ in range(num_restarts - 1):
            restart_points.append(rng_restart.uniform(-5.0, 5.0, n))
            
        # Run (1+1)-ES from each restart point
        for start_idx, start_x in enumerate(restart_points):
            if time.time() > deadline - 5.0:
                break
            if calls_made >= total_budget:
                break
                
            # Allocate budget for this restart
            if start_idx == num_restarts - 1:
                alloc = total_budget - calls_made
            else:
                alloc = budget_per_restart
                
            if alloc < 10:
                continue
                
            # Run the ES and get the number of calls made
            n_calls_made_in_es = self._adaptive_es(func, start_x, alloc, deadline, n)
            calls_made += n_calls_made_in_es
            
        return best_x, best_fx

    def _adaptive_es(self, func, x0, budget, deadline, n):
        """
        Robust (1+1)-ES with simple adaptive step size.
        Returns the number of function calls made.
        """
        x = np.array(x0, dtype=float)
        sigma = 1.0
        fx = func(x)
        calls = 1
        
        # Simple adaptation
        factor_up = 1.1
        factor_down = 0.9
        min_sigma = 1e-6
        max_sigma = 10.0
        
        for _ in range(budget - calls):
            if time.time() > deadline - 1.0:
                break
            if calls >= budget:
                break
                
            # Generate trial
            direction = np.random.randn(n)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            else:
                direction = np.ones(n) / np.sqrt(n)
                
            x_trial = x + sigma * direction
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            fx_trial = func(x_trial)
            calls += 1
            
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                sigma = min(sigma * factor_up, max_sigma)
            else:
                sigma = max(sigma * factor_down, min_sigma)
                
        return calls
# EVOLVE-BLOCK-END
