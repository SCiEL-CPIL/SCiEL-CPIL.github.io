"""
Optimizer using Multi-Start Nelder-Mead with Coordinate-wise Polish for BBOB f20 (Schwefel).
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s
        deadline = start_time + 290.0
        dim = self.dim
        total_budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Deadline exceeded")
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Multi-Start Nelder-Mead
        rng = np.random.default_rng(42)
        
        # Define starting points
        starts = []
        starts.append(np.zeros(dim))
        # Add some random starts
        for _ in range(5):
            starts.append(rng.uniform(-4.0, 4.0, dim))
        
        # Run Nelder-Mead from each start
        # We limit maxiter to prevent any single run from consuming too much budget
        # Nelder-Mead in 5D uses 6 points in simplex.
        # maxiter is not exactly calls, but a good heuristic.
        # We rely on safe_call to enforce the hard limit.
        
        for x0 in starts:
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            
            try:
                # Use a tight tolerance to encourage convergence within few calls
                # If it doesn't converge, it will just stop after maxiter or budget
                res = minimize(
                    safe_call,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50, # Heuristic limit
                        'xatol': 1e-5,
                        'fatol': 1e-5
                    }
                )
            except RuntimeError:
                pass
            except Exception:
                pass

        # Phase 2: Coordinate-wise Polish (Golden Section Search)
        # This is very effective for the final convergence of Schwefel
        if best_x is not None:
            # We will perform a few rounds of coordinate-wise optimization
            # Each round iterates over all dimensions
            # Golden section search typically takes ~15-20 calls to converge well in 1D
            # With 5 dims, one round is ~75-100 calls.
            # We have ~200-300 calls left after NM (assuming NM used ~200-300).
            # So we can do 2-3 rounds.
            
            max_rounds = 3
            for _ in range(max_rounds):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break
                
                for i in range(dim):
                    if calls_made >= total_budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Golden Section Search for dimension i
                    # We search in a range around the current best_x[i]
                    # Or just the full box [-5, 5] to be safe, but that's expensive.
                    # Let's search in [-5, 5] but we can narrow it if we want.
                    # Actually, for Schwefel, the minimum is often not at the boundary.
                    # Let's use the full range [-5, 5] for the 1D search to ensure we don't miss it.
                    
                    a = -5.0
                    b = 5.0
                    gr = (np.sqrt(5) - 1) / 2  # Golden ratio ~0.618
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    
                    # We need to evaluate c and d
                    # Create helper to evaluate 1D
                    def f_1d(t):
                        x_temp = best_x.copy()
                        x_temp[i] = t
                        return safe_call(x_temp)
                    
                    try:
                        fc = f_1d(c)
                        fd = f_1d(d)
                    except RuntimeError:
                        break
                    except Exception:
                        break
                    
                    # Iterate golden section
                    for _ in range(20): # Max 20 iterations for 1D search
                        if calls_made >= total_budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        if fc < fd:
                            b = d
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            try:
                                fc = f_1d(c)
                            except RuntimeError:
                                break
                            except Exception:
                                break
                        else:
                            a = c
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            try:
                                fd = f_1d(d)
                            except RuntimeError:
                                break
                            except Exception:
                                break
                        if abs(b - a) < 1e-6:
                            break
                    
                    # Update best_x[i] to the midpoint of the final interval
                    if a < b:
                        best_x[i] = (a + b) / 2.0

        # Fallback
        if best_x is None:
            try:
                x = np.zeros(dim)
                best_f = func(x)
                calls_made += 1
                best_x = x
            except:
                best_x = np.zeros(dim)
                best_f = float('nan')

        return best_x, best_f
