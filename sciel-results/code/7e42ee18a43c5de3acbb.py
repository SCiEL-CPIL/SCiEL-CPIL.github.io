"""
Gauss-Newton Optimizer for BBOB f11 (Discus).

Strategy:
1. Phase 1 (Exploration): Evaluate a deterministic set of diverse starting points
   (origin, axis-aligned points, and some random-like deterministic points) to
   identify multiple promising basins. This provides strong early progress for AOCC.
   
2. Phase 2 (Exploitation): For each of the top few starting points, run a 
   Gauss-Newton method. This method estimates the Hessian of the function
   using finite differences. Since f11 is a rotated quadratic function, the
   Hessian is constant (or nearly so in the basin). Gauss-Newton (or Newton's
   method for the quadratic) converges quadratically, which is much faster
   than Nelder-Mead's linear convergence or L-BFGS-B's superlinear convergence.
   
3. Phase 3 (Polish): Continue Gauss-Newton steps until convergence or budget
   exhaustion.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Gauss-Newton Optimizer for BBOB f11 (Discus).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Helper to check budget and time
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Diverse Initialization ---
        # Generate a set of diverse starting points to cover the search space.
        # The minimum is known to be in [-4, 4]^5.
        
        # Start with origin and axis-aligned points
        start_points = [np.zeros(dim)]
        
        # Add points at +/- 3 along each axis (within [-4,4])
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                start_points.append(p)
        
        # Add some deterministic pseudo-random points
        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(42)
        for _ in range(10):
            p = rng.uniform(-4, 4, size=dim)
            start_points.append(p)
        
        # Evaluate all starting points
        start_f_values = []
        for p in start_points:
            if not can_call():
                break
            fx = evaluate(p)
            start_f_values.append((fx, p))
        
        if best_x is None:
            # Fallback
            x0 = np.zeros(dim)
            evaluate(x0)
            if best_x is None:
                return np.zeros(dim)
        
        # Sort starting points by function value, keep top few for multi-start
        start_f_values.sort(key=lambda x: x[0])
        top_starts = start_f_values[:min(3, len(start_f_values))]
        
        # --- Phase 2: Gauss-Newton for each top start ---
        
        def gauss_newton(x_start, f_start, max_iter=20):
            nonlocal calls_made, best_x, best_f
            
            n = dim
            x = x_start.copy()
            f = f_start
            
            # Estimate Hessian using central differences
            # This requires 2*n + 1 function evaluations
            # We do this once per restart, or update it if needed.
            # For a quadratic function, the Hessian is constant.
            
            h = 1e-4
            H = np.zeros((n, n))
            
            # We need to be careful with budget. 
            # If we have spent a lot, we might not have enough for Hessian estimation.
            # But we have 500 calls total. 
            # Phase 1 used ~25 calls. 
            # Each restart can afford ~150 calls.
            # Hessian estimation costs ~11 calls.
            # Each GN step costs 1 call.
            
            # Estimate Hessian at x_start
            # f(x) = 0.5 * x^T H x + g^T x + c
            # We can estimate H directly.
            
            # Check if we have enough budget for Hessian estimation
            if calls_made + 2 * n + 1 > budget:
                # Fallback to simple gradient descent if budget is tight? 
                # Or just skip Hessian and use small steps?
                # Let's try to do it if possible.
                pass
            
            # Estimate Hessian
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        # Diagonal
                        xp = x.copy(); xp[i] += h
                        xm = x.copy(); xm[i] -= h
                        fp = evaluate(xp)
                        fm = evaluate(xm)
                        if not np.isfinite(fp) or not np.isfinite(fm):
                            continue
                        H[i, i] = (fp - 2*f + fm) / (h*h)
                    else:
                        # Off-diagonal
                        # f(x+he_i+he_j) - f(x+he_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)
                        xpp = x.copy(); xpp[i] += h; xpp[j] += h
                        xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                        xmp = x.copy(); xmp[i] -= h; xmp[j] += h
                        xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                        
                        fpp = evaluate(xpp)
                        fpm = evaluate(xpm)
                        fmp = evaluate(xmp)
                        fmm = evaluate(xmm)
                        
                        if not (np.isfinite(fpp) and np.isfinite(fpm) and np.isfinite(fmp) and np.isfinite(fmm)):
                            continue
                        
                        H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h * h)
                        H[j, i] = H[i, j]
            
            # Regularize Hessian to ensure positive definiteness
            # Add a small diagonal term if H is not PD
            try:
                eigvals, eigvecs = np.linalg.eigh(H)
                if np.min(eigvals) < 1e-6:
                    H = H + (1e-6 - np.min(eigvals)) * np.eye(n)
            except Exception:
                H = H + 1e-6 * np.eye(n)
            
            # Now perform Newton steps
            for _ in range(max_iter):
                if not can_call():
                    break
                
                # Estimate Gradient using central differences
                # This costs 2*n calls
                g = np.zeros(n)
                for i in range(n):
                    xp = x.copy(); xp[i] += h
                    xm = x.copy(); xm[i] -= h
                    fp = evaluate(xp)
                    fm = evaluate(xm)
                    if not (np.isfinite(fp) and np.isfinite(fm)):
                        break
                    g[i] = (fp - fm) / (2 * h)
                
                if not np.all(np.isfinite(g)):
                    break
                
                # Solve H dx = -g
                try:
                    dx = np.linalg.solve(H, -g)
                except Exception:
                    # Fallback to steepest descent if solve fails
                    dx = -g * 0.01
                
                # Line search to ensure decrease
                # Simple backtracking line search
                step = 1.0
                f_new = float('inf')
                for _ in range(10):
                    x_trial = x + step * dx
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = evaluate(x_trial)
                    if f_trial < f:
                        f_new = f_trial
                        break
                    step *= 0.5
                
                if not np.isfinite(f_new):
                    break
                
                x = x_trial
                f = f_new
                
                # Convergence check
                if np.linalg.norm(dx) < 1e-6:
                    break
            
            # Update global best
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Run multi-start: Gauss-Newton for each top starting point
        for f_start, x_start in top_starts:
            if not can_call():
                break
            gauss_newton(x_start, f_start, max_iter=10)
        
        return best_x

# EVOLVE-BLOCK-END
