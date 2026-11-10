"""Multi-objective optimizer for ZDT6 using direct hypervolume-optimal front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer that directly targets hypervolume maximization.
    
    Strategy:
    - Assume the Pareto front of the specific ZDT6 instance in the evaluator is a
      monotonic curve in objective space where f2 decreases as f1 increases.
    - Instead of blindly searching decision space, we sample f1 values that are
      optimally spaced to maximize the hypervolume under the assumption of a
      convex/concave front shape.
    - We use a dense sweep of x[0] (which typically controls f1 in ZDT variants
      where x[0] is the 'shape' variable) while setting x[1:] to values that
      minimize g (often x[1:] = 0 or x[1:] = 0.5 depending on the exact ZDT6
      variant). Given the reference point [11, 11], and typical ZDT6 scaling,
      we must find the specific x[1:] configuration that places points inside
      the reference box.
      
    Since the standard ZDT6 has f1 >> 11, this task likely uses a scaled version
    or a variant where the front lies within [0, 11]x[0, 11]. 
    A common variant or interpretation is that x[1:] should be set to minimize
    the 'complexity' term, often x[1:] = 0.5 or x[1:] = 0. 
    However, to be robust, we will perform a dense sweep of x[0] from 0 to 1, 
    and for x[1:], we will try a few candidate configurations (0.0, 0.5, 1.0) 
    in a preliminary phase to identify which one yields f-values within the 
    reference box, then focus budget on the best-performing configuration.
    
    Actually, a more direct "hypervolume-optimal" approach:
    1. Use a small fraction of budget to probe the problem structure.
       Evaluate points with x[1:] = 0.0, 0.5, 1.0 at a few x[0] values.
    2. Identify the x[1:] configuration that produces objective values < 11.
    3. Use the remaining budget to densely sample x[0] for that configuration,
       using a spacing that maximizes hypervolume (e.g., uniform or geometric).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        eval_count = 0
        all_points = []
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            eval_count += 1
            x = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x), dtype=float)
                if np.all(np.isfinite(f)):
                    all_points.append((x, f))
                    return f
            except Exception:
                pass
            return None

        # Phase 1: Probe structure
        # Try different settings for x[1:] to see which one yields valid points (f < 11)
        # Common candidates for ZDT6-like problems:
        candidates = [
            np.full(n, 0.5),  # Standard center
            np.zeros(n),      # Zero
            np.ones(n),       # One
            np.full(n, 0.1),  # Low
            np.full(n, 0.9),  # High
        ]
        
        # Set x[0] to a mid-point for probing
        probe_x0 = 0.5
        
        valid_configs = []
        for config in candidates:
            x = config.copy()
            x[0] = probe_x0
            f = evaluate(x)
            if f is not None:
                # Check if it's within the reference box [0, 11] x [0, 11]
                # "Strictly better than reference point" means f1 < 11 and f2 < 11
                if np.all(f < 11.0):
                    valid_configs.append((f, config))
        
        # If no valid config found, try a random search in x[1:]
        if not valid_configs:
            # Fallback: Random search to find any point in the box
            for _ in range(50):
                x = np.random.uniform(lo, hi)
                f = evaluate(x)
                if f is not None and np.all(f < 11.0):
                    valid_configs.append((f, x))
                    if len(valid_configs) >= 5:
                        break
            
            if not valid_configs:
                # If still nothing, return whatever we have (likely empty or out of box)
                return [p[0] for p in all_points]

        # Select the configuration that seems to give the "best" (lowest) objectives
        # to start with, or just pick the first one.
        # Let's pick the one with the lowest sum of objectives to be safe.
        best_config = min(valid_configs, key=lambda t: np.sum(t[0]))[1]
        
        # Phase 2: Dense sweep of x[0] with the best x[1:] configuration
        # We assume x[0] is the variable that controls the trade-off (f1 vs f2).
        # We will sweep x[0] from 0 to 1.
        
        remaining_budget = budget - eval_count
        if remaining_budget <= 0:
            return [p[0] for p in all_points]
        
        # Use a dense grid. 200 points is a lot, so we can afford a fine sweep.
        # Let's use all remaining budget for the sweep.
        n_sweep = remaining_budget
        
        # Create a sweep of x[0]
        # To maximize hypervolume, uniform spacing in x[0] is often good if the front is smooth.
        # If the front is non-uniform, we might want to adapt, but without knowing the exact shape,
        # uniform is a strong baseline.
        
        x0_vals = np.linspace(0.0, 1.0, n_sweep)
        
        for x0 in x0_vals:
            if eval_count >= budget:
                break
            x = best_config.copy()
            x[0] = x0
            evaluate(x)
        
        return [p[0] for p in all_points]
# EVOLVE-BLOCK-END
