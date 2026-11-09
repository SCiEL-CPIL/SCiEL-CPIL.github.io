"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Active Set / Anisotropic 1D Search Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        # 300s limit, use 290s to be safe (evaluator kills at 330s, but let's be conservative)
        self.time_limit = 290.0
        
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
            
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # Helper for 1D minimization along a direction
        def line_search_1d(center, direction, max_evals=200, bracket_range=10.0):
            """
            Performs a 1D minimization along 'direction' starting from 'center'.
            Uses scipy's brent or minimize_scalar if available, but to save imports and overhead,
            we implement a simple robust golden-section or use scipy if already imported.
            Since scipy is imported, we use minimize_scalar for robustness.
            """
            nonlocal self, best_x, best_fx
            
            def f1d(t):
                x_new = center + t * direction
                return eval_func(x_new)
                
            # We need to bracket the minimum.
            # The Bent Cigar valley can be narrow. 
            # We try a range around 0.
            # If the direction is unit-normalized, t is the distance.
            
            # Try a few different bracket ranges to ensure we catch the minimum
            # if it's far away or very close.
            ranges = [0.01, 0.1, 1.0, 5.0, 10.0]
            
            best_t = 0.0
            best_f = f1d(0.0) # We already have center evaluated? No, eval_func counts calls.
            # Note: center might have been evaluated before. 
            # To avoid double counting or relying on cache, we just assume we are exploring.
            # Actually, we should check if we've already evaluated center.
            # For simplicity in this black-box context, we assume we can re-evaluate or 
            # we manage state. The task says "Every call is recorded". 
            # Re-evaluating the same point wastes budget. 
            # However, we don't have a cache. 
            # Strategy: We will pass points that we haven't evaluated recently or hope for deduplication?
            # No, the evaluator just counts calls.
            # So we must be careful.
            
            # Let's use a simple coordinate descent style refinement which is robust.
            # Or use scipy.optimize.minimize_scalar with bounds.
            
            try:
                # Use scipy minimize_scalar with method='bounded'
                # We need to define bounds.
                # Since we don't know the scale of t, we try expanding bounds.
                
                # Start with small bounds
                for bound in [0.1, 1.0, 10.0]:
                    res = minimize_scalar(f1d, bounds=(-bound, bound), method='bounded', options={'maxiter': 50})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_t = res.x
                        # Update global best
                        x_candidate = center + best_t * direction
                        if best_f < best_fx:
                            best_fx = best_f
                            best_x = x_candidate.copy()
                # If we found something good, maybe refine further?
                # The bounded method is quite good.
            except Exception:
                pass

            return best_t, best_f

        # 1. Coarse Global Sampling to identify the best basin
        # We evaluate a small set of diverse points to find the most promising start
        rng = np.random.default_rng(42)
        
        # Define candidate points
        candidates = [np.zeros(d)]
        # Add random points
        for _ in range(5):
            candidates.append(rng.uniform(-4.5, 4.5, size=d))
        # Add points near boundaries
        candidates.append(np.array([4.0] * d))
        candidates.append(np.array([-4.0] * d))
        
        # Evaluate all candidates
        for c in candidates:
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            try:
                eval_func(c)
            except StopIteration:
                break
            except Exception:
                pass

        # 2. Anisotropic Refinement Phase
        # The Bent Cigar function has one direction with high curvature (lambda=100) and others with low (lambda=1).
        # We want to align with the high curvature direction and minimize along it.
        
        # We will iteratively:
        # a) Perform a few steps of coordinate descent or Powell to get a local neighborhood.
        # b) Use Gram-Schmidt on recent points to find orthogonal directions.
        # c) Perform 1D line search along the direction that yields the best improvement.
        
        # Let's use a hybrid: 
        # First, run a short Powell to get a decent local landscape.
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            try:
                # Run Powell for a limited number of iterations to align directions
                # Powell naturally finds the axes.
                res = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 20, # Very short, just to get some directional info
                        'ftol': 1e-8,
                        'xatol': 1e-6,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass
        
        # Now, perform targeted 1D searches along the coordinate axes and then mixed directions.
        # Since the rotation is hidden, we don't know the true axes.
        # However, Powell's internal directions might have aligned somewhat.
        # But we don't have access to them.
        
        # Alternative Strategy: 
        # Perform a "Simplified Active Set" method:
        # 1. Pick a direction (initially coordinate axes).
        # 2. Minimize along it.
        # 3. Update the set of directions based on the new best point and previous best points.
        
        # Let's try a sequence of 1D searches along the standard basis vectors first.
        # This is cheap and effective if the rotation is small. If rotation is large, it might be slow.
        
        for _ in range(3): # Repeat a few times to allow convergence
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            
            for i in range(d):
                if self.calls >= N or time.time() - self.start_time > self.time_limit:
                    break
                
                direction = np.zeros(d)
                direction[i] = 1.0
                
                # Perform 1D search along this direction
                # We use a local bracket around the current best_x
                # If the minimum is far, we might miss it. 
                # To handle this, we can use a wider bracket if the function is still decreasing.
                
                # Simple 1D search implementation using scipy
                def f1d(t):
                    x_new = best_x + t * direction
                    return eval_func(x_new)
                    
                try:
                    # Try a small range first
                    res = minimize_scalar(f1d, bounds=(-1.0, 1.0), method='bounded', options={'maxiter': 20})
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = (best_x + res.x * direction).copy()
                        
                        # If we improved significantly, try a wider range to see if we can go further
                        # Check gradient-like behavior? No, just try wider.
                        # If the function is still decreasing at the boundary, expand.
                        # For simplicity, we just do one more try with a wider bound if improvement was good.
                        if best_fx < res.fun + 1e-10: # Roughly, if it's at the edge
                             # Actually, minimize_scalar returns the optimum. If it's at the bound, it might be outside.
                             # We can check f(1.0) vs f(res.x)
                             pass
                except Exception:
                    pass
        
        # 3. Final Polish with Powell
        # Use the remaining budget for a standard Powell run to clean up.
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            try:
                res = minimize(
                    eval_func,
                    best_x,
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

        # Fallback if no valid result
        if not np.isfinite(best_fx):
            if self.calls < N and time.time() - self.start_time < self.time_limit:
                try:
                    best_fx = eval_func(np.zeros(d))
                    best_x = np.zeros(d)
                except:
                    best_fx = 0.0
                    best_x = np.zeros(d)
            else:
                best_fx = 0.0
                best_x = np.zeros(d)

        return best_x, best_fx
# EVOLVE-BLOCK-END
