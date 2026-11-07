"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start BFGS with finite-difference gradients for Rosenbrock valley."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _fd_gradient(self, func, x, h=1e-5):
        """Compute finite-difference gradient."""
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad

    def _bfgs_step(self, func, x, fx, budget_remaining, deadline, h=1e-5):
        """
        Run BFGS from x with remaining budget. Returns (x_best, fx_best, calls_used).
        Uses finite-difference gradients and adaptive step size.
        """
        dim = self.dim
        # Initialize inverse Hessian approximation
        H = np.eye(dim)
        x_prev = x.copy()
        fx_prev = fx
        grad_prev = self._fd_gradient(func, x, h)
        calls_used = dim * 2  # 2 evaluations per gradient component

        best_x = x.copy()
        best_fx = fx
        calls_used += 1  # for the initial fx (already counted by caller, but let's track)
        
        # Actually, let's re-think. The caller has already evaluated x.
        # We need to track calls carefully.
        
        # Let's restart the call counting.
        # The caller passes fx which was 1 call.
        # We'll track additional calls from here.
        
        calls_from_here = 0
        max_bfgs_iters = min(20, budget_remaining // (dim * 2 + 1))
        
        for it in range(max_bfgs_iters):
            if time.time() > deadline:
                break
            if calls_from_here >= budget_remaining - 5:
                break
                
            # Compute gradient
            grad = self._fd_gradient(func, x, h)
            calls_from_here += dim * 2
            
            if time.time() > deadline or calls_from_here >= budget_remaining - 5:
                break
                
            # Check convergence
            if np.linalg.norm(grad) < 1e-7:
                break
                
            # BFGS update of H
            s = x - x_prev
            y = grad - grad_prev
            sy = s @ y
            if sy > 1e-12:
                Hs = H @ s
                H = H - (Hs @ s.T) / (s @ s) + np.outer(y, y) / sy
            
            # Search direction
            d = -H @ grad
            
            # Adaptive step size via backtracking line search
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            max_ls = 10
            for ls in range(max_ls):
                x_new = np.clip(x + alpha * d, -5, 5)
                fx_new = func(x_new)
                calls_from_here += 1
                
                if fx_new < fx + c1 * alpha * (grad @ d):
                    break
                alpha *= rho
            
            # Update BFGS state
            s = x_new - x
            y = self._fd_gradient(func, x_new, h) - grad
            calls_from_here += dim * 2
            
            if time.time() > deadline or calls_from_here >= budget_remaining - 5:
                # Still use x_new if it's better
                if fx_new < best_fx:
                    best_x = x_new.copy()
                    best_fx = fx_new
                break
                
            x_prev = x
            x = x_new
            fx_prev = fx
            fx = fx_new
            grad_prev = y
            
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
        
        return best_x, best_fx, calls_from_here

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_fx = np.inf
        
        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_fx  # Return best known value to avoid extra calls
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x)
            calls_made += 1
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val
        
        # Multi-start strategy: use diverse starting points
        # For 5D, we can afford several starts
        # Budget allocation: explore phase + exploit phase
        
        # Phase 1: Random exploration to find good basins
        # Phase 2: BFGS refinement from best points
        
        # Let's use a simple strategy:
        # - Try 5-10 random starts with BFGS from each
        # - Allocate budget dynamically
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        n_starts = 8
        starts = []
        for i in range(n_starts):
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.ones(dim) * 3.0
            elif i == 2:
                x0 = -np.ones(dim) * 3.0
            else:
                x0 = rng.uniform(-4, 4, dim)
            x0 = np.clip(x0, -5, 5)
            starts.append(x0)
        
        # Evaluate all starts
        for x0 in starts:
            if calls_made >= budget - 10 or time.time() > deadline:
                break
            fx0 = safe_func(x0)
        
        # Sort starts by fitness
        start_fxs = []
        for x0 in starts:
            if calls_made >= budget - 10 or time.time() > deadline:
                break
            # We already evaluated some, but let's just re-evaluate to be safe
            # Actually, we can track which ones we evaluated
            fx0 = safe_func(x0)
            start_fxs.append((fx0, x0))
        
        start_fxs.sort(key=lambda t: t[0])
        
        # BFGS refinement from best starts
        for fx0, x0 in start_fxs:
            if calls_made >= budget - 5 or time.time() > deadline:
                break
            remaining = budget - calls_made - 5
            if remaining < dim * 2 + 1:
                break
            
            x_best, fx_best, calls_used = self._bfgs_step(
                safe_func, x0, fx0, remaining, deadline
            )
            calls_made += calls_used
            
            if fx_best < best_fx:
                best_fx = fx_best
                best_x = x_best.copy()
        
        # Final polish: if we have budget left, do one more BFGS from the best point
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline:
            remaining = budget - calls_made - 5
            if remaining >= dim * 2 + 1:
                x_best, fx_best, calls_used = self._bfgs_step(
                    safe_func, best_x, best_fx, remaining, deadline
                )
                calls_made += calls_used
                if fx_best < best_fx:
                    best_fx = fx_best
                    best_x = x_best.copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
