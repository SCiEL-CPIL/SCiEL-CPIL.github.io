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

        def run_powell(start_x, max_evals):
            nonlocal self
            calls_at_start = self.calls
            
            def eval_func_limited(x):
                if (self.calls - calls_at_start) >= max_evals:
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
                        'xatol': 1e-10,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        # 1. Global Phase: Sample diverse points
        rng = np.random.default_rng(42)
        candidates = []
        # 20 random points
        for _ in range(20):
            p = rng.uniform(-4.0, 4.0, size=d)
            candidates.append(p)
        
        # Structured points
        candidates.append(np.zeros(d))
        candidates.append(np.array([4.0] * d))
        candidates.append(np.array([-4.0] * d))
        # Mid-points
        candidates.append(np.array([0.0, 4.0, 0.0, 4.0, 0.0]))
        candidates.append(np.array([4.0, 0.0, 4.0, 0.0, 4.0]))
        
        results = []
        for c in candidates:
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            try:
                fx = eval_func(c)
                results.append((fx, c.copy()))
            except StopIteration:
                break
            except Exception:
                pass
        
        if not results:
            return best_x, best_fx

        # Sort by function value
        results.sort(key=lambda x: x[0])
        
        # 2. Local Phase: Run Powell from top 3
        top_k = 3
        max_evals_per_start = 120 # 3 * 120 = 360 evals
        for i in range(min(top_k, len(results))):
            fx, x_start = results[i]
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            run_powell(x_start, max_evals_per_start)
            
        # 3. Polish Phase: Run Powell from the overall best
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            remaining = N - self.calls
            # Use a smaller budget for polish to avoid overshooting
            polish_budget = min(100, remaining)
            run_powell(best_x, polish_budget)

        return best_x, best_fx
