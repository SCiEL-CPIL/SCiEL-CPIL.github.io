"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Track global best
        best_x = None
        best_f = np.inf
        calls = [0]  # use list for closure mutation

        def tracked_func(x):
            """Evaluate x, track calls and best."""
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls[0] += 1
            if calls[0] > budget:
                raise TimeoutError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Use a deterministic seed for reproducibility
        rng = np.random.default_rng(12345)

        # Phase 1: Multi-start (1+1)-ES
        # Büche-Rastrigin has many local minima, so we do multiple restarts
        # with different starting points and sigma values.
        
        num_starts = 8
        calls_per_start = max(1, (budget - 10) // num_starts)  # leave some for final refinement
        
        for start_idx in range(num_starts):
            if calls[0] >= budget or time.time() > deadline - 5:
                break
            
            # Different starting points: spread across the domain
            if start_idx == 0:
                x = rng.uniform(-5, 5, dim)
            elif start_idx == 1:
                x = np.zeros(dim)  # center
            elif start_idx == 2:
                x = rng.uniform(-4, 4, dim)
            else:
                # Perturb the current best or random
                if best_x is not None and rng.random() < 0.5:
                    x = np.clip(best_x + rng.standard_normal(dim) * 2.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)
            
            fx = tracked_func(x)
            
            # Adaptive sigma for this start
            sigma = 1.0 + start_idx * 0.3  # larger starts for later explorations
            max_sigma = 4.0
            min_sigma = 0.01
            success_count = 0
            total_count = 0
            target_success = 1.0 / (10 * dim + 2)  # standard 1+1-ES success rate
            
            # Run (1+1)-ES for this start
            start_call_count = calls[0]
            while calls[0] < budget and (calls[0] - start_call_count) < calls_per_start and time.time() < deadline - 2:
                # Adapt sigma
                if total_count >= 10:
                    success_rate = success_count / total_count
                    if success_rate > target_success * 1.5:
                        sigma = min(sigma * 1.2, max_sigma)
                    elif success_rate < target_success * 0.5:
                        sigma = max(sigma * 0.8, min_sigma)
                    success_count = 0
                    total_count = 0
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = tracked_func(y)
                total_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    # Shrink sigma slightly on success for fine-tuning
                    sigma = max(sigma * 0.99, min_sigma)
                else:
                    # On failure, occasionally do a larger jump
                    if rng.random() < 0.03:
                        y2 = x + sigma * 3.0 * rng.standard_normal(dim)
                        y2 = np.clip(y2, -5.0, 5.0)
                        fy2 = tracked_func(y2)
                        total_count += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            success_count += 1
                            sigma = max(sigma * 0.9, min_sigma)
            
            # Update global best if this start found something better
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Fine-tuning around the best point with small sigma
        if best_x is not None and calls[0] < budget and time.time() < deadline - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            min_sigma = 0.001
            success_count = 0
            total_count = 0
            target_success = 1.0 / (10 * dim + 2)
            
            fine_tune_calls = min(budget - calls[0], 50)
            for _ in range(fine_tune_calls):
                if time.time() > deadline - 1 or calls[0] >= budget:
                    break
                
                if total_count >= 10:
                    success_rate = success_count / total_count
                    if success_rate > target_success * 1.5:
                        sigma = min(sigma * 1.1, 0.5)
                    elif success_rate < target_success * 0.5:
                        sigma = max(sigma * 0.7, min_sigma)
                    success_count = 0
                    total_count = 0
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = tracked_func(y)
                total_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.95, min_sigma)

        return best_x, best_f
# EVOLVE-BLOCK-END
