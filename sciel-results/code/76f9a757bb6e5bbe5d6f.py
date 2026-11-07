"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Optimizer using Differential Evolution for global search and Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        call_count = 0
        best_x = None
        best_f = np.inf

        # Exception to signal budget or time exceeded
        class SearchStop(Exception):
            pass

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            
            # Time check
            if time.time() > deadline:
                raise SearchStop("Time limit exceeded")
            
            # Budget check
            call_count += 1
            if call_count > self.budget:
                raise SearchStop("Budget exceeded")
            
            # Ensure x is a valid array of correct shape
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
                
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # 1. Global Search Phase: Differential Evolution
        # DE is robust to shifts and rotations and does not rely on gradients.
        # We use a small population to fit within the budget.
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Reserve some budget for local refinement
        # DE with popsize=10 and maxiter=40 uses approx 10 + 40*10 = 410 calls
        # This leaves ~90 calls for local refinement
        de_popsize = 10
        de_maxiter = 40
        
        try:
            # seed=42 for reproducibility
            res = differential_evolution(
                wrapped_func,
                bounds,
                popsize=de_popsize,
                maxiter=de_maxiter,
                seed=42,
                tol=1e-6,
                polish=False  # We will do our own polish
            )
            if res.x is not None:
                # Ensure best_x is updated from DE result if it's better
                # wrapped_func already updates best_x, but res.x is the best found by DE
                # We can evaluate res.x again to be sure, but it might have been evaluated.
                # Actually, differential_evolution calls wrapped_func, so best_x is already updated.
                pass
        except SearchStop:
            pass
        except Exception:
            pass

        if best_x is None:
            # Fallback if no calls were made
            return np.ones(self.dim), np.inf

        # 2. Local Refinement Phase: Nelder-Mead
        # Nelder-Mead is derivative-free, fast in low dimensions, and good for final polishing.
        if call_count < self.budget - 10 and time.time() < deadline:
            try:
                minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except SearchStop:
                pass
            except Exception:
                pass

        # 3. Final Polish with L-BFGS-B if budget remains
        # L-BFGS-B can be very precise if we are close to the minimum.
        if call_count < self.budget - 5 and time.time() < deadline:
            try:
                minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxfun': self.budget - call_count
                    }
                )
            except SearchStop:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
