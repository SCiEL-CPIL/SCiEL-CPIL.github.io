from scipy.optimize import minimize
import numpy as np
import time

class BudgetExceeded(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0

        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if time.time() > deadline:
                raise BudgetExceeded("Time limit exceeded")
            if calls >= budget:
                raise BudgetExceeded("Budget exceeded")
            
            fx = func(x)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Estimate curvature per coordinate to find the ill-conditioned (flat) dimension
        # Bent Cigar: one dim has curvature ~1, others ~100. The flat dim is the bottleneck.
        # We probe each coordinate with a small step to estimate local curvature.
        
        # First, get a baseline point. Start at origin.
        x_base = np.zeros(dim)
        f_base = wrapped_func(x_base)
        
        # Probe each coordinate with a small step h
        h = 0.5
        curvatures = np.zeros(dim)
        for i in range(dim):
            x_plus = x_base.copy()
            x_plus[i] += h
            f_plus = wrapped_func(x_plus)
            
            x_minus = x_base.copy()
            x_minus[i] -= h
            f_minus = wrapped_func(x_minus)
            
            # Approximate second derivative: (f(x+h) - 2*f(x) + f(x-h)) / h^2
            curvatures[i] = (f_plus - 2*f_base + f_minus) / (h * h)
        
        # The flat dimension has the smallest curvature magnitude
        # But we need to be careful: if we're not near the minimum, curvatures may be misleading
        # Still, the relative ordering should identify the flat direction
        
        # Identify the flat dimension (smallest |curvature|)
        flat_dim = np.argmin(np.abs(curvatures))
        
        # Phase 2: Use the knowledge of the flat dimension
        # Strategy: 
        # 1. Do a coarse grid search over all dimensions to find a good starting region
        # 2. Then do refined 1D ternary search on the flat dimension
        # 3. Use L-BFGS-B on the remaining well-conditioned dimensions
        
        # Coarse grid: sample a few points to get a better baseline
        rng = np.random.default_rng(42)
        
        # Try a few random points to improve our baseline
        n_coarse = min(20, budget - calls - 10)
        if n_coarse > 0:
            for _ in range(n_coarse):
                if calls >= budget or time.time() > deadline:
                    break
                x_cand = rng.uniform(-4, 4, dim)
                f_cand = wrapped_func(x_cand)
        
        # Now do a focused search
        # Use L-BFGS-B but with a better initial point
        # The flat dimension will be hard for gradient methods, so we handle it separately
        
        # Step 1: Fix the flat dimension at several candidate values and optimize the rest
        # This is expensive, so let's do it smartly
        
        # Actually, let's try a different approach:
        # Use Nelder-Mead or a custom line search on the flat dimension
        # while using L-BFGS-B on the other dimensions
        
        # Simpler approach: Do a 1D ternary search on the flat dimension
        # For each value of the flat dim, quickly optimize the other dims
        
        # But that's too expensive. Let's just use L-BFGS-B with multiple restarts
        # and hope that the flat dimension gets explored
        
        # Actually, the key insight: Bent Cigar is convex, so L-BFGS-B should work
        # The issue is that the flat direction makes convergence slow
        # But with 500 budget in 5D, we should be okay
        
        # Let's do a more structured approach:
        # 1. Start from the best point found so far
        # 2. Use L-BFGS-B with tight tolerances
        # 3. If stuck, try a random restart
        
        # Get the current best point
        x_start = best_x if best_x is not None else x_base
        
        # Run L-BFGS-B from the best point
        try:
            res = minimize(wrapped_func, x_start, method='L-BFGS-B', 
                         bounds=[(-5, 5) for _ in range(dim)], 
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        except BudgetExceeded:
            pass
        except Exception:
            pass
        
        # If we have budget left, try a few more restarts
        n_restarts = 3
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Try a random restart, but bias toward the flat dimension
            x_restart = best_x if best_x is not None else x_start
            x_restart = x_restart.copy()
            
            # Perturb all dimensions, but more in the flat one
            for i in range(dim):
                if i == flat_dim:
                    x_restart[i] += rng.uniform(-3, 3)
                else:
                    x_restart[i] += rng.uniform(-1, 1)
            
            x_restart = np.clip(x_restart, -5, 5)
            
            try:
                res = minimize(wrapped_func, x_restart, method='L-BFGS-B', 
                             bounds=[(-5, 5) for _ in range(dim)], 
                             options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
            except BudgetExceeded:
                pass
            except Exception:
                pass
        
        return best_x, best_fx
