import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        n = self.dim
        total_budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        # Lock for thread safety if needed, though single thread here
        # We use a nonlocal counter via a list or attribute to be safe in closures if any, 
        # but simple nonlocal works in Python 3.
        
        def safe_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls >= total_budget:
                raise StopIteration("Budget exhausted")

            x = np.asarray(x, dtype=np.float64).flatten()
            if x.shape != (n,):
                try:
                    x = x.reshape(n)
                except:
                    return 1e10
            
            # The evaluator clips, but we should ensure our internal state is consistent
            # The contract says: func(x) clips a finite length-5 vector to the box.
            # So we can pass unclipped x, but for our bookkeeping, we clip to know where we are.
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f = float(func(x_clipped))
            except Exception:
                f = 1e10
            
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()

            return f

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        # 1. Origin
        starts = [np.zeros(n)]
        # 2. Random points in the box
        n_starts = 20
        for _ in range(n_starts):
            starts.append(rng.uniform(-4.5, 4.5, size=n))
        
        # 3. Specific points often good for Rosenbrock valley (near 1,1,1,1,1)
        starts.append(np.ones(n))
        starts.append(np.ones(n) * 0.5)
        starts.append(np.ones(n) * -0.5)

        # Shuffle starts to mix deterministic and random
        rng.shuffle(starts)

        # Allocate budget per local optimization run
        # We have ~500 calls. 20 starts. ~25 calls per start on average.
        # L-BFGS-B is efficient.
        
        for x0 in starts:
            if calls >= total_budget or time.time() > deadline:
                break
            
            # If we are very close to the optimum, we might skip or do a tighter search
            # But we don't know f_opt.
            
            try:
                # Use L-BFGS-B with bounds
                # maxiter limits iterations, but we also have the call limit in safe_func
                # ftol/gtol control convergence
                res = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 50,  # Limit iterations to spread budget
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'disp': False
                    }
                )
            except (StopIteration, TimeoutError):
                break
            except Exception:
                continue

        # Final polish if budget remains
        if calls < total_budget and time.time() < deadline:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-15,
                        'gtol': 1e-10,
                        'disp': False
                    }
                )
            except (StopIteration, TimeoutError):
                pass
            except Exception:
                pass

        return best_x, best_f
