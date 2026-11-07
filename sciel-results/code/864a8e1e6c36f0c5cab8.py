"""Hybrid optimizer: L-BFGS-B for basin finding, coordinate descent for anisotropic refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f12 (Bent Cigar).
    Strategy:
    1. L-BFGS-B to find the basin (exploits gradient info if available, robust otherwise).
    2. Coordinate Descent with Golden Section Search to refine the narrow valley.
       This is crucial for Bent Cigar because the function is highly ill-conditioned
       (1000x difference in curvature). Gradient methods can struggle to converge
       tightly along the narrow axis without many iterations. CD with 1D search
       converges quadratically (effectively) along each axis and is very efficient.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return np.inf
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global/Local Basin Finding with L-BFGS-B ---
        # Start at origin, which is typically near the center of the search space
        # and often close to the minimum for BBOB functions after shifting.
        
        # Wrapper for L-BFGS-B to enforce budget and time limits
        def lbfgsb_func(x):
            if calls_used >= budget or time.time() > deadline - 10:
                return 1e10
            val = evaluate(x)
            if val == np.inf:
                return 1e10
            return val

        x0 = np.zeros(dim)
        
        # Initial evaluation
        f0 = evaluate(x0)
        if f0 == np.inf:
            # Should not happen, but safe fallback
            return best_x, best_f

        # Run L-BFGS-B
        # We limit maxiter to ensure we don't consume the entire budget,
        # leaving room for the more effective CD refinement on the anisotropic landscape.
        # maxiter=50 is usually enough to get into the basin for dim=5.
        try:
            # bounds are handled by the evaluator's clipping, but we can pass them to scipy
            # to help it avoid going out of bounds unnecessarily (though it might still try).
            # Actually, L-BFGS-B with bounds is good.
            res = minimize(lbfgsb_func, x0, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * dim, 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            pass

        # If L-BFGS-B failed or didn't improve, stick with best so far
        if best_x is None:
            best_x = x0
            best_f = f0

        # --- Phase 2: Coordinate Descent Refinement ---
        # Bent Cigar: f(x) = (x1 - y1)^2 + 1000 * sum((xi - yi)^2 for i>1)
        # The valley is extremely narrow in dimensions 2..d.
        # Coordinate descent is optimal here because it decouples the axes.
        # We use Golden Section Search for each axis to find the minimum efficiently.
        
        current_x = best_x.copy()
        current_f = best_f

        # Number of CD passes. 
        # Each pass costs roughly dim * (evals_per_golden).
        # Golden section search takes ~log(eps/S)/log(0.618) steps.
        # With S=10, eps=1e-6, ~30 steps. 
        # 5 dims * 30 = 150 evals per pass.
        # We have ~500 - (L-BFGS-B usage). L-BFGS-B uses ~50-100.
        # So we can afford 2-3 passes.
        
        max_passes = 3
        pass_count = 0
        
        while pass_count < max_passes:
            if time.time() > deadline - 10:
                break
            if calls_used >= budget - 10:
                break
            
            improved = False
            for i in range(dim):
                if time.time() > deadline - 10:
                    break
                if calls_used >= budget - 10:
                    break
                    
                # Define 1D function along axis i
                def func_1d(t):
                    x_temp = current_x.copy()
                    x_temp[i] = t
                    # Check budget inside evaluate, but we need to handle the return
                    if calls_used >= budget or time.time() > deadline - 5:
                        return 1e10
                    val = evaluate(x_temp)
                    if val == np.inf:
                        return 1e10
                    return val

                # Golden Section Search
                # Initial interval: [current_x[i] - 1.0, current_x[i] + 1.0]
                # If the function is decreasing/increasing, we might need to expand.
                # But since we are refining from L-BFGS-B, we should be close.
                # Let's use a slightly wider interval to be safe, e.g., +/- 2.0
                # But clipped to [-5, 5].
                
                a = max(-5.0, current_x[i] - 2.0)
                b = min(5.0, current_x[i] + 2.0)
                
                # If interval is too small or invalid, skip
                if b - a < 1e-6:
                    continue

                # Golden ratio
                phi = (np.sqrt(5) - 1) / 2
                inv_phi = 1.0 / phi
                inv_phi2 = inv_phi * inv_phi

                # Initial points
                c = b - inv_phi * (b - a)
                d = a + inv_phi * (b - a)
                
                fc = func_1d(c)
                fd = func_1d(d)
                
                # Iterative refinement
                # Max 30 iterations for 1D search
                for _ in range(30):
                    if time.time() > deadline - 5:
                        break
                    if calls_used >= budget - 5:
                        break
                        
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - inv_phi * (b - a)
                        fc = func_1d(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + inv_phi * (b - a)
                        fd = func_1d(d)
                        
                    if abs(b - a) < 1e-8:
                        break

                # Best point is the midpoint of the final interval
                new_val = (a + b) / 2.0
                current_x[i] = new_val
                
                # Evaluate the new point to update best_f and check improvement
                # Note: func_1d already evaluated points, but we need to ensure 
                # the current_x state is evaluated if it changed significantly.
                # Actually, the golden section search evaluates points. 
                # The best point found is roughly (a+b)/2. 
                # We should evaluate this specific point to ensure it's recorded as best.
                f_new = func_1d(new_val)
                
                if f_new < current_f - 1e-10:
                    current_f = f_new
                    improved = True
                    # Update global best if needed (evaluate already does this)
                    # best_x and best_f are updated inside evaluate()
                
            pass_count += 1
            if not improved:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
