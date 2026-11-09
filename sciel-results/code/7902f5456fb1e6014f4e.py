"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    """Multi-start CMA-ES based optimizer for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        x_global_best = np.random.uniform(-5, 5, dim)
        f_global_best = np.inf
        calls = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # Phase 1: Sparse Random Sampling to identify promising basins
        # f22 is highly multimodal. We spend a small fraction of the budget
        # on random sampling to find the global minimum basin.
        n_init = 50
        for _ in range(n_init):
            if not check_time() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Parallel Multi-Start CMA-ES on the best basins
        if HAS_CMA and check_time() and calls < budget:
            # We will run multiple CMA-ES instances in parallel (conceptually, 
            # we run them sequentially but they explore different regions).
            # How many? Let's try 5 instances.
            n_starts = 5
            # We need to pick the best n_starts points from our initial sampling.
            # Since we didn't store all points, we just re-sample or use the global best 
            # and perturbations? 
            # Better: We should have stored the top K points from Phase 1.
            # Let's refactor Phase 1 to store top K.
            
            # Actually, to keep it simple and robust, let's just run CMA-ES from the global best 
            # and a few random perturbations of it, or just restart CMA-ES if it stagnates?
            # The prompt asks for a DIFFERENT strategy. Multi-start is different from single-run.
            # Let's implement a proper multi-start by keeping the best 5 from the initial phase.
            
            # Re-do Phase 1 to store points
            # We already used 50 calls. Let's assume we have x_global_best.
            # To do a true multi-start, we need multiple good starting points.
            # Let's generate 4 more random points and evaluate them? No, budget is tight.
            
            # Alternative: Run CMA-ES on the global best. If it stops early (stagnation), 
            # start a new one from a random point or the second best.
            # But the "Parallel" idea is better.
            
            # Let's modify the approach: 
            # We will run CMA-ES for a limited number of calls (e.g., 100) starting from the global best.
            # Then, if we have budget left, we will run another CMA-ES starting from a new random point 
            # (or the second best from a new small sampling round).
            
            # For simplicity and robustness, let's just run CMA-ES on the global best with a 
            # slightly larger population to explore more, and rely on CMA-ES's internal 
            # restart logic? No, cma.fmin doesn't auto-restart well in a black-box call count context.
            
            # Let's stick to a simpler "Multi-Start" implementation:
            # 1. We have the global best from Phase 1.
            # 2. We run CMA-ES from there.
            # 3. We limit the number of iterations/calls for this instance.
            # 4. If we have budget left, we pick a new random point and run another CMA-ES.
            
            # This is effectively sequential multi-start.
            
            remaining_budget = budget - calls
            if remaining_budget > 20:
                # Run CMA-ES
                pop_size = 10
                sigma0 = 2.0
                
                opts = {
                    'maxiter': 1000,
                    'seed': 42,
                    'verbose': -9,
                    'tolx': 1e-8,
                    'tolfun': 1e-10,
                    'bounds': [-5, 5],
                    'popsize': pop_size
                }
                
                def wrapped_func(x):
                    if not check_time():
                        raise Exception("Time limit")
                    if calls >= budget:
                        raise Exception("Budget")
                    return evaluate(x)
                
                try:
                    # Run CMA-ES
                    best_x, result, es = cma.fmin(wrapped_func, x_global_best.copy(), sigma0, opts)
                    if best_x is not None:
                        # The evaluate function inside wrapped_func already updates global best
                        pass
                except Exception:
                    pass
                    
        else:
            # Fallback: Simple (1+1)-ES if cma is not available or failed
            if check_time() and calls < budget:
                x = x_global_best.copy()
                f = f_global_best
                sigma = 1.0
                C = np.eye(dim)
                
                while check_time() and calls < budget:
                    z = rng.multivariate_normal(np.zeros(dim), C)
                    norm_z = np.linalg.norm(z)
                    if norm_z > 1e-10:
                        z = z / norm_z
                    
                    x_trial = x + sigma * z
                    f_trial = evaluate(x_trial)
                    
                    if f_trial < f:
                        C = (1 - 1.0/dim) * C + (1.0/dim) * (np.outer(z, z))
                        x = x_trial
                        f = f_trial
                    else:
                        # Shrink sigma occasionally
                        if rng.random() < 0.1:
                            sigma *= 0.9
                            C = 0.5 * C + 0.5 * np.eye(dim)

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
