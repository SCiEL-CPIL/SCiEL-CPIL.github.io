"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f11 (Discus):
    1. Deterministic Radial Sweep to align with the anisotropic axis.
    2. Hessian Newton Polishing to exploit the local quadratic structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit
        calls_made = 0
        x_best = np.zeros(dim)
        f_best = np.inf

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def update_best(x, f):
            nonlocal x_best, f_best, calls_made
            if f < f_best:
                x_best = x.copy()
                f_best = f

        # --- Phase 1: Radial Sweep for Axis Alignment ---
        # The Discus function is anisotropic. We need to find the "valley" direction.
        # We sample points along several directions at various radii.
        
        # Directions: Coordinate axes and their negatives, plus diagonals
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            d[i] = -1.0
            directions.append(d)
        
        # Add some diagonal directions for better coverage
        for i in range(dim):
            for j in range(i+1, dim):
                d = np.zeros(dim)
                d[i] = 1.0
                d[j] = 1.0
                directions.append(d)
                d[i] = -1.0
                d[j] = 1.0
                directions.append(d)

        # Radii to check
        radii = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Evaluate Origin first
        f0 = func(np.zeros(dim))
        calls_made += 1
        update_best(np.zeros(dim), f0)

        # Sweep
        for d in directions:
            norm_d = np.linalg.norm(d)
            if norm_d == 0: continue
            d = d / norm_d
            for r in radii:
                if calls_made >= budget - 100: # Reserve budget for Newton
                    break
                x_trial = clip(r * d)
                f_trial = func(x_trial)
                calls_made += 1
                update_best(x_trial, f_trial)
                if time.time() > deadline:
                    return x_best, f_best

        # --- Phase 2: Hessian Newton Polishing ---
        # The Discus function is locally quadratic. We fit a local Hessian and jump to the minimum.
        
        x_curr = x_best.copy()
        f_curr = f_best
        
        # Parameters for finite differences
        h = 0.05
        
        # We will iterate as long as we have budget and time
        # Each Newton iteration costs:
        # Gradient: 2*d calls
        # Hessian: d*(d+1) calls (using central diffs for off-diagonal, 4 calls each, but we can reuse?)
        # Actually, let's estimate Hessian properly.
        # For d=5:
        # Grad: 10 calls
        # Hess: 5 diag (2 calls each = 10) + 10 off-diag (4 calls each = 40) = 50 calls.
        # Total per full Newton step estimation: ~60 calls.
        # With 500 budget, we can afford ~5-8 full Newton steps if we don't waste calls on line search.
        
        newton_round = 0
        while calls_made < budget - 20 and time.time() < deadline:
            newton_round += 1
            
            # 1. Estimate Gradient
            grad = np.zeros(dim)
            for i in range(dim):
                if calls_made + 2 > budget: break
                xp = x_curr.copy(); xp[i] += h
                xm = x_curr.copy(); xm[i] -= h
                xp = clip(xp); xm = clip(xm)
                fpx = func(xp); fxm = func(xm)
                calls_made += 2
                grad[i] = (fpx - fxm) / (2 * h)
            
            if np.linalg.norm(grad) < 1e-8:
                break

            # 2. Estimate Hessian
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    if calls_made + 4 > budget: break
                    if i == j:
                        xp = x_curr.copy(); xp[i] += h
                        xm = x_curr.copy(); xm[i] -= h
                        xp = clip(xp); xm = clip(xm)
                        fpp = func(xp); fxm = func(xm)
                        calls_made += 2
                        H[i, i] = (fpp - 2 * f_curr + fxm) / (h * h)
                    else:
                        xpp = x_curr.copy(); xpp[i] += h; xpp[j] += h
                        xpm = x_curr.copy(); xpm[i] += h; xpm[j] -= h
                        xmp = x_curr.copy(); xmp[i] -= h; xmp[j] += h
                        xmm = x_curr.copy(); xmm[i] -= h; xmm[j] -= h
                        xpp = clip(xpp); xpm = clip(xpm)
                        xmp = clip(xmp); xmm = clip(xmm)
                        fpp = func(xpp); fpm = func(xpm)
                        fmp = func(xmp); fxm = func(xmm)
                        calls_made += 4
                        H[i, j] = (fpp - fpm - fmp + fxm) / (4 * h * h)
                        H[j, i] = H[i, j]
            
            # 3. Compute Newton Step
            # Regularize Hessian to ensure it's positive definite and invertible
            H_reg = H + 1e-4 * np.eye(dim)
            
            try:
                step = np.linalg.solve(H_reg, grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent step if Hessian is singular
                step = -grad * 0.1

            # 4. Line Search
            alpha = 1.0
            improved = False
            for ls_iter in range(10):
                if calls_made >= budget:
                    break
                x_trial = clip(x_curr - alpha * step)
                f_trial = func(x_trial)
                calls_made += 1
                
                if f_trial < f_curr:
                    x_curr = x_trial
                    f_curr = f_trial
                    update_best(x_curr, f_curr)
                    improved = True
                    break
                else:
                    alpha *= 0.5
            
            if not improved:
                # If Newton step didn't work, try a small random perturbation to escape local minima or numerical issues
                # and re-estimate Hessian from the new point
                if calls_made < budget - 10:
                    perturbation = 0.1 * np.random.randn(dim)
                    x_trial = clip(x_curr + perturbation)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < f_curr:
                        x_curr = x_trial
                        f_curr = f_trial
                        update_best(x_curr, f_curr)
                    else:
                        # If even perturbation fails, we might be at the optimum or stuck
                        # Try a smaller step size for next Newton iteration
                        h *= 0.5
                        if h < 1e-4:
                            break

        return x_best, f_best
# EVOLVE-BLOCK-END
