"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free BFGS with multi-start for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def _finite_diff_gradient(x, f0, h=1e-6):
            """Compute gradient using central differences. Uses 2*dim evaluations."""
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                f_plus = _evaluate(x_plus)
                f_minus = _evaluate(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * h)
            return grad

        def _dfbfgs(x0, f0, max_iter=50):
            """
            Derivative-free BFGS local optimizer.
            Uses finite differences for gradients and line search for step size.
            """
            nonlocal call_count, deadline
            
            if call_count >= budget:
                return x0, f0
            
            # Compute initial gradient
            g = _finite_diff_gradient(x0, f0)
            if call_count >= budget:
                return x0, f0
            
            # Check if gradient is small enough to stop
            if np.linalg.norm(g) < 1e-8:
                return x0, f0
            
            # Inverse Hessian approximation (start with identity)
            H = np.eye(dim)
            
            x = x0.copy()
            f = f0
            xk1 = x.copy()
            fk1 = f
            gk1 = g.copy()
            
            for iteration in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Search direction: p = -H * g
                p = -H @ g
                
                # Line search: backtracking to find step size
                # Try alpha values: 1.0, 0.5, 0.25, 0.125, ...
                alpha = 1.0
                found_step = False
                for ls_iter in range(20):
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_trial = x + alpha * p
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = _evaluate(x_trial)
                    
                    # Armijo condition: f(x + alpha*p) <= f(x) + c1 * alpha * g^T * p
                    # c1 = 0.01 is a standard choice
                    if f_trial <= f + 0.01 * alpha * (g @ p):
                        found_step = True
                        break
                    
                    alpha *= 0.5
                
                if not found_step:
                    # If no step found, try a small random perturbation
                    # or break
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try a small step in the negative gradient direction
                    alpha_small = 1e-4
                    x_trial = x + alpha_small * (-g / (np.linalg.norm(g) + 1e-10))
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = _evaluate(x_trial)
                    
                    if f_trial < f:
                        xk1 = x.copy()
                        fk1 = f
                        gk1 = g.copy()
                        x = x_trial
                        f = f_trial
                        # Recompute gradient
                        g = _finite_diff_gradient(x, f)
                        if call_count >= budget:
                            break
                        if time.time() > deadline:
                            break
                    else:
                        break
                
                # Update x and f
                xk1 = x.copy()
                fk1 = f
                gk1 = g.copy()
                x = x + alpha * p
                x = np.clip(x, -5.0, 5.0)
                f = f_trial
                
                # Compute new gradient
                g = _finite_diff_gradient(x, f)
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Check convergence
                if np.linalg.norm(g) < 1e-8:
                    break
                
                # BFGS update
                s = x - xk1
                y = g - gk1
                sy = s @ y
                
                if sy > 1e-10:
                    # Standard BFGS update
                    H = H - (H @ s @ s.T @ H) / (s.T @ H @ s) + np.outer(s, s) / sy
                    # Symmetrize to avoid numerical issues
                    H = (H + H.T) / 2.0
                else:
                    # Skip update if sy is too small
                    pass
            
            return x, f

        # Phase 1: Global Sampling to find diverse starting points
        # Generate a set of candidate starting points
        start_points = []
        
        # Origin
        start_points.append(np.zeros(dim))
        
        # Axis-aligned points
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                start_points.append(pt)
        
        # Random points
        for _ in range(20):
            pt = rng.uniform(-5.0, 5.0, size=dim)
            start_points.append(pt)
        
        # Evaluate all starting points to find the best initial point
        # Limit to a reasonable number to save budget
        max_init = min(len(start_points), 30)
        
        for pt in start_points[:max_init]:
            if call_count >= budget - 100:  # Reserve budget for local optimization
                break
            if time.time() > deadline:
                break
            _evaluate(pt)
        
        # If we haven't found anything, use origin
        if best_x is None:
            best_x = np.zeros(dim)
            _evaluate(best_x)
        
        # Phase 2: Run DF-BFGS from the best point found
        # We could also run from multiple top points, but let's start with the best
        x_opt, f_opt = _dfbfgs(best_x.copy(), best_f)
        
        # If we have remaining budget, try a few more starting points
        # Sort start points by their function values (we need to track them)
        # For simplicity, just try the next few best if budget allows
        # Actually, let's just run one more from a different random point if budget allows
        if call_count < budget - 50 and time.time() < deadline:
            # Pick a random point different from best_x
            for _ in range(5):
                pt = rng.uniform(-5.0, 5.0, size=dim)
                if np.linalg.norm(pt - best_x) > 1.0:  # Ensure it's different
                    break
            else:
                pt = best_x + rng.uniform(-1, 1, size=dim)
            
            if call_count < budget - 50 and time.time() < deadline:
                f_pt = _evaluate(pt)
                if call_count < budget - 30 and time.time() < deadline:
                    x_opt2, f_opt2 = _dfbfgs(pt.copy(), f_pt)
                    if f_opt2 < f_opt:
                        x_opt, f_opt = x_opt2, f_opt2
        
        return best_x, best_f
# EVOLVE-BLOCK-END
