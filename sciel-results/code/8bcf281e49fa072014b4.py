import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 285.0

        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Phase 1: Multi-start Global (1+1)-ES with success-rate adaptation
        # Use 3 starting points for better global coverage
        n_starts = 3
        explore_budget = int(budget * 0.6)
        per_start = explore_budget // n_starts
        
        for start_idx in range(n_starts):
            if calls >= explore_budget or time.time() > deadline:
                break
            
            # Initialize at a random point in [-4, 4]^d for first start, 
            # or near best for subsequent starts
            if start_idx == 0:
                x = rng.uniform(-4, 4, d)
            else:
                x = best_x + rng.normal(0, 1.0, d)
                x = np.clip(x, -5, 5)
            
            try:
                f_x = safe_func(x)
            except StopIteration:
                break
            
            sigma = 1.5
            stall_count = 0
            stall_limit = 40
            window = 15  # success rate window
            
            # Track successes for adaptive step size
            successes = []
            
            for _ in range(per_start):
                if calls >= explore_budget or time.time() > deadline:
                    break
                
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    successes.append(1)
                    stall_count = 0
                else:
                    successes.append(0)
                    stall_count += 1
                
                # Adapt sigma based on recent success rate (target ~0.35)
                if len(successes) >= window:
                    success_rate = sum(successes[-window:]) / window
                    if success_rate > 0.45:
                        sigma *= 1.15
                    elif success_rate < 0.25:
                        sigma *= 0.85
                    successes = successes[-window:]
                
                # Stall handling: try larger step or restart
                if stall_count >= stall_limit:
                    if sigma < 3.0:
                        sigma = min(sigma * 2.0, 5.0)
                    else:
                        # Random restart near best
                        x = best_x + rng.normal(0, 1.5, d)
                        x = np.clip(x, -5, 5)
                        try:
                            f_x = safe_func(x)
                        except StopIteration:
                            break
                        sigma = 2.0
                    stall_count = 0
                    successes = []

        # Phase 2: Local Refinement with finer adaptation
        if best_x is not None and calls < budget:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.3
            
            successes = []
            window = 12
            
            while calls < budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break

                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    successes.append(1)
                else:
                    successes.append(0)
                
                if len(successes) >= window:
                    sr = sum(successes[-window:]) / window
                    if sr > 0.4:
                        sigma *= 1.1
                    elif sr < 0.2:
                        sigma *= 0.8
                    successes = successes[-window:]
                
                if sigma < 1e-8:
                    break

        # Phase 3: Final high-precision polish
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            no_improve = 0
            
            while calls < budget and time.time() < deadline and no_improve < 30:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break

                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 0.95
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

        return best_x, best_f
