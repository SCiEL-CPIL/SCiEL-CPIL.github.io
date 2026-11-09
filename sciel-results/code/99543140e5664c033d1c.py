"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Powell Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 280.0
        
        d = self.dim
        N = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_fx = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_fx
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            
            self.calls += 1
            fx = func(x)
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # Helper to run Powell with a specific start and optional per-start budget limit
        def run_powell(start_x, max_evals=None):
            nonlocal self
            calls_at_start = self.calls
            
            def eval_func_limited(x):
                if max_evals is not None and (self.calls - calls_at_start) >= max_evals:
                    raise StopIteration("Start budget exceeded")
                return eval_func(x)

            try:
                minimize(
                    eval_func_limited,
                    start_x,
                    method='Powell',
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-12,
                        'xatol': 1e-8,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        # 1. Coarse Global Sampling to identify the best basin
        # We evaluate a small set of diverse points to find the most promising start
        rng = np.random.default_rng(42)
        
        # Define candidate points: Center, Corners (sampled), and Random
        candidates = [np.zeros(d)]
        # Add some structured points
        for i in range(5):
            p = rng.uniform(-4.0, 4.0, size=d)
            candidates.append(p)
        # Add points near the boundaries to catch shifts
        candidates.append(np.array([4.0] * d))
        candidates.append(np.array([-4.0] * d))
        
        # Evaluate all candidates (using 1 call each)
        # This is cheap (12 calls) and helps locate the basin
        for c in candidates:
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            try:
                eval_func(c)
            except StopIteration:
                break
            except Exception:
                pass

        # 2. Focused Local Optimization
        # Run Powell from the single best point found, using the vast majority of the budget
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            run_powell(best_x, None)

        # 3. Final Polish
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            try:
                calls_at_start = self.calls
                def eval_func_polish(x):
                    if self.calls - calls_at_start >= 20:
                        raise StopIteration("Polish budget exceeded")
                    return eval_func(x)
                
                minimize(
                    eval_func_polish,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'ftol': 1e-14,
                        'xatol': 1e-10,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
